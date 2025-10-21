"""
Compare Superforecaster (SF) vs LLM Delphi performance with maximal clarity.

WHAT THIS SCRIPT DOES (step-by-step):
1) For each Delphi log JSON:
   a) Read the *ground truth* resolution (0/1) for (question_id, resolution_date).
   b) Get the *Superforecaster* (SF) probabilities (no rounds).
      - Select the **median SF by probability**.
      - Compute that median SF's **Brier score**.
   c) Get the *LLM* probabilities **per round** from the Delphi log.
      - For each round, select the **median LLM by probability**.
      - Compute that median LLM's **Brier score** for that round.
   d) Store per-question results.

2) After all questions:
   a) Compute the **average of the median SF Brier** across questions.
   b) For each round, compute the **average of the median LLM Brier** across questions.

KEY PRINCIPLE:
- We pick the **median forecaster by probability** (SF overall; LLM per round)
  and then compute **that forecaster's Brier** against the resolved outcome.

ASSUMPTIONS:
- Resolution is binary (0 or 1).
- Delphi logs follow: delphi_log_<question_id>_<YYYY-MM-DD>.json
- _collect_round_probs(delphi_log) is defined elsewhere and returns either:
  - dict: {round_idx: {expert_id: prob}}
  - or dict: {round_idx: [prob, prob, ...]}
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

import argparse
import json
import os
import re
import pickle
import yaml
from typing import List, Dict, Any, Optional
from dataset.dataloader import ForecastDataLoader

import numpy as np
import matplotlib.pyplot as plt

from utils.config_types import load_typed_experiment_config

# Robustly load question ID sets even if utils.py shadows the utils package
try:
    from utils.sampling import (
        TRAIN_QUESTION_IDS,
        EVALUATION_QUESTION_IDS,
        EVOLUTION_EVALUATION_QUESTION_IDS,
    )
except Exception:
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[1]
    _sampling_fp = _root / "utils" / "sampling.py"
    _spec = _ilu.spec_from_file_location("project_utils_sampling", str(_sampling_fp))
    _mod = _ilu.module_from_spec(_spec)
    assert _spec and _spec.loader, f"Cannot load sampling module at {_sampling_fp}"
    _spec.loader.exec_module(_mod)
    TRAIN_QUESTION_IDS = set(getattr(_mod, "TRAIN_QUESTION_IDS", []))
    EVALUATION_QUESTION_IDS = set(getattr(_mod, "EVALUATION_QUESTION_IDS", []))
    EVOLUTION_EVALUATION_QUESTION_IDS = set(
        getattr(_mod, "EVOLUTION_EVALUATION_QUESTION_IDS", [])
    )

# Optional fallback in case some rounds only stored raw text responses
_PROB_PAT = re.compile(r"FINAL PROBABILITY:\s*(0?\.\d+|1\.0|0|1)", re.IGNORECASE)


def _collect_round_probs(delphi_log: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    """
    Returns {round_idx: {sfid: probability}} using stored numeric probs when available,
    falling back to parsing the response text if needed.
    """
    out: Dict[int, Dict[str, float]] = {}
    rounds = delphi_log.get("rounds", [])
    for r in rounds:
        r_idx = int(r.get("round", 0))
        expert_dict = r.get("experts", {})
        probs: Dict[str, float] = {}
        for sfid, entry in expert_dict.items():
            # Prefer stored numeric prob
            p = entry.get("prob")
            if isinstance(p, (int, float)):
                p = float(p)
            else:
                # Fallback to parse from response text
                p = _extract_prob(entry.get("response"))
            if p is not None:
                probs[sfid] = max(0.0, min(1.0, p))
        out[r_idx] = probs
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _extract_prob(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    matches = _PROB_PAT.findall(text)
    if matches:
        try:
            p = float(matches[-1])
            return max(0.0, min(1.0, p))
        except ValueError:
            pass
    # fallback: last bare number
    nums = re.findall(r"0?\.\d+|1\.0|0|1", text)
    if nums:
        try:
            p = float(nums[-1])
            return max(0.0, min(1.0, p))
        except ValueError:
            pass
    return None


def compute_brier_score(prob: float, outcome: int) -> float:
    """Brier score for a binary outcome in {0,1}."""
    return (prob - outcome) ** 2


def parse_delphi_log_filename(filename: str, file_pattern: str) -> Tuple[str, str]:
    """
    Parse filename based on the config pattern like "prompt_comparison_high_variance_{question_id}_{resolution_date}.json"
    Returns (question_id, date).
    """
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)

    # Find the prefix before {question_id}
    prefix_end = file_pattern.find("{question_id}")
    if prefix_end == -1:
        raise ValueError(f"Pattern {file_pattern} must contain {{question_id}}")

    prefix = file_pattern[:prefix_end]

    # Remove the prefix from the filename
    if not stem.startswith(prefix):
        raise ValueError(f"Filename {filename} doesn't match pattern {file_pattern}")

    remainder = stem[len(prefix) :]

    # The date pattern is YYYY-MM-DD, find it in the remainder
    # Look for pattern like 2025-07-21
    import re

    date_pattern = r"\d{4}-\d{2}-\d{2}"
    date_matches = re.findall(date_pattern, remainder)

    if not date_matches:
        raise ValueError(f"Cannot find date in filename: {filename}")

    # Take the last date match as the resolution date
    date_str = date_matches[-1]

    # The question_id is everything before the last occurrence of _YYYY-MM-DD
    # Find the position of the date in the remainder
    date_pos = remainder.rfind("_" + date_str)
    if date_pos == -1:
        # Date might be directly attached without underscore
        date_pos = remainder.rfind(date_str)
        if date_pos == -1:
            raise ValueError(f"Cannot parse question_id and date from: {filename}")
        question_id = remainder[:date_pos].rstrip("_")
    else:
        question_id = remainder[:date_pos]

    return question_id, date_str


def median_by_probability(probs: List[float]) -> Optional[float]:
    """
    Return the median value from a list of probabilities.
    Empty -> None.
    """
    clean = [p for p in probs if isinstance(p, (int, float))]
    if not clean:
        return None
    clean.sort()
    return clean[len(clean) // 2]


def llm_round_probs_to_list(prob_map: Any) -> List[float]:
    """
    Normalize an LLM round probability container to a plain list[float].
    Accepts dict{id: prob} or list[prob]. Filters to numeric.
    """
    if isinstance(prob_map, dict):
        vals = list(prob_map.values())
    else:
        vals = list(prob_map)
    return [p for p in vals if isinstance(p, (int, float))]


def compute_sf_median_brier(
    loader: ForecastDataLoader,
    question_id: str,
    resolution_date: str,
    resolved_outcome: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Superforecasters do not have rounds.
    1) Collect all SF probabilities (one per SF).
    2) Take the **median probability** (by value).
    3) Compute Brier of that median probability vs resolved outcome.
    Returns: (median_sf_prob, median_sf_brier)
    """
    sf_forecasts = loader.get_super_forecasts(
        question_id=question_id, resolution_date=resolution_date
    )
    sf_probs = [sf.forecast for sf in sf_forecasts]
    median_sf_prob = median_by_probability(sf_probs)
    if median_sf_prob is None:
        return None, None
    return median_sf_prob, compute_brier_score(median_sf_prob, resolved_outcome)


def compute_public_median_brier(
    loader: ForecastDataLoader,
    question_id: str,
    resolution_date: str,
    resolved_outcome: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Public forecasts do not have rounds.
    1) Collect all public probabilities (one per public forecaster).
    2) Take the **median probability** (by value).
    3) Compute Brier of that median probability vs resolved outcome.
    Returns: (median_public_prob, median_public_brier)
    """
    public_forecasts = loader.get_public_forecasts(
        question_id=question_id, resolution_date=resolution_date
    )
    public_probs = [pf.forecast for pf in public_forecasts]
    median_public_prob = median_by_probability(public_probs)
    if median_public_prob is None:
        return None, None
    return median_public_prob, compute_brier_score(median_public_prob, resolved_outcome)


def compute_llm_median_brier_by_round(
    delphi_log: Dict[str, Any], resolved_outcome: int
) -> Dict[int, Dict[str, Optional[float]]]:
    """
    LLMs have multiple rounds recorded in the Delphi log.
    For each round:
      1) Collect all expert probabilities for that round.
      2) Take the **median probability**.
      3) Compute Brier of that median probability vs resolved outcome.
    Returns:
      { round_idx: { "median_llm_prob": float|None, "median_llm_brier": float|None } }
    """
    out: Dict[int, Dict[str, Optional[float]]] = {}
    llm_probs_by_round = _collect_round_probs(delphi_log)

    for round_idx, prob_container in llm_probs_by_round.items():
        probs = llm_round_probs_to_list(prob_container)
        median_llm_prob = median_by_probability(probs)
        if median_llm_prob is None:
            out[round_idx] = {"median_llm_prob": None, "median_llm_brier": None}
            continue
        out[round_idx] = {
            "median_llm_prob": median_llm_prob,
            "median_llm_brier": compute_brier_score(median_llm_prob, resolved_outcome),
        }
    return out


def average_across_questions(values: List[float]) -> float:
    """Mean with explicit float cast; caller should filter out None beforehand."""
    return float(np.mean(values)) if values else float("nan")


def aggregate_llm_rounds_across_questions(
    per_question_llm: Dict[str, Dict[int, Dict[str, Optional[float]]]],
) -> Dict[int, float]:
    """
    For each round r, average the LLM **median Brier** across all questions that have r.
    Input structure:
      per_question_llm[question_id][round_idx]["median_llm_brier"] -> float|None
    Output:
      { round_idx: average_median_llm_brier }
    """
    accumulator: Dict[int, List[float]] = {}
    for qid, per_round in per_question_llm.items():
        for r, payload in per_round.items():
            b = payload.get("median_llm_brier")
            if b is not None:
                accumulator.setdefault(r, []).append(b)

    return {r: average_across_questions(vals) for r, vals in accumulator.items()}


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze Delphi experiment results")
    parser.add_argument(
        "config_path", help="Path to experiment configuration YAML file"
    )
    parser.add_argument(
        "--set",
        default="auto",
        choices=["auto", "train", "eval", "evolution_eval"],
        help="Override YAML sampling method: choose question set explicitly",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_typed_experiment_config(args.config_path)

    # 0) Setup
    loader = ForecastDataLoader()
    results_dir = config["experiment"]["output_dir"]
    initial_forecasts_dir = config["experiment"]["initial_forecasts_dir"]

    # Use the file pattern from config to identify log files
    file_pattern = config["output"]["file_pattern"]
    # Extract the prefix from the pattern up to {question_id}
    pattern_prefix = file_pattern.split("{")[0]  # Gets everything before first {

    # Recursively gather logs under results_dir to support set subfolders (train/eval/evolution_eval)
    delphi_logs = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".json") and f.startswith(pattern_prefix):
                delphi_logs.append(os.path.join(root, f))

    # Storage for per-question results
    per_question_results: Dict[str, Dict[str, Any]] = {}

    # Sampling-method filtering: restrict to the configured question set
    sampling_method = (config.get("data", {}).get("sampling", {}) or {}).get("method")
    allowed_qids = None
    if args.set != "auto":
        if args.set == "train":
            allowed_qids = set(TRAIN_QUESTION_IDS)
        elif args.set == "eval":
            allowed_qids = set(EVALUATION_QUESTION_IDS)
        elif args.set == "evolution_eval":
            allowed_qids = set(EVOLUTION_EVALUATION_QUESTION_IDS)
    else:
        if sampling_method == "evaluation":
            allowed_qids = set(EVALUATION_QUESTION_IDS)
        elif sampling_method == "evolution_evaluation":
            allowed_qids = set(EVOLUTION_EVALUATION_QUESTION_IDS)
        elif sampling_method in ("train", "train_small"):
            allowed_qids = set(TRAIN_QUESTION_IDS)

    # 1) Process each Delphi log (i.e., each question)
    for delphi_log_file in delphi_logs:
        question_id, resolution_date = parse_delphi_log_filename(
            delphi_log_file, file_pattern
        )
        if allowed_qids is not None and question_id not in allowed_qids:
            continue

        # 1a) Ground truth outcome
        resolution = loader.get_resolution(
            question_id=question_id, resolution_date=resolution_date
        )
        y_true = resolution.resolved_to  # expected 0 or 1

        # 1b) Superforecasters (no rounds): median-by-prob -> Brier
        median_sf_prob, median_sf_brier = compute_sf_median_brier(
            loader=loader,
            question_id=question_id,
            resolution_date=resolution_date,
            resolved_outcome=y_true,
        )

        # Public forecasters (no rounds): median-by-prob -> Brier
        median_public_prob, median_public_brier = compute_public_median_brier(
            loader=loader,
            question_id=question_id,
            resolution_date=resolution_date,
            resolved_outcome=y_true,
        )

        # 1c) LLM (per round): median-by-prob -> Brier
        try:
            with open(delphi_log_file, "r") as f:
                delphi_log = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON file {delphi_log_file}: {e}")
            continue

        llm_median_by_round = compute_llm_median_brier_by_round(
            delphi_log=delphi_log, resolved_outcome=y_true
        )

        # load no-example baseline forecasts
        no_example_file = os.path.join(
            initial_forecasts_dir,
            f"collected_fcasts_no_examples_{resolution_date}_{question_id}.json",
        )
        if not os.path.exists(no_example_file):
            print(
                f"Warning: No no-example forecasts found for {question_id} at {no_example_file}"
            )
            no_example_data = {}
        else:
            with open(no_example_file, "r") as f:
                no_example_data = json.load(f)
        no_example_probs = no_example_data.get("forecasts", {})
        no_example_median_prob = median_by_probability(no_example_probs)
        no_example_brier = None
        if no_example_median_prob is not None:
            no_example_brier = compute_brier_score(no_example_median_prob, y_true)

        # 1d) Persist per-question results for later aggregation
        per_question_results[question_id] = {
            "resolution_value": y_true,
            "sf": {
                "median_prob": median_sf_prob,
                "median_brier": median_sf_brier,
            },
            "public": {
                "median_prob": median_public_prob,
                "median_brier": median_public_brier,
            },
            "llm": {
                "median_by_round": llm_median_by_round,
                # e.g., {0: {"median_llm_prob": 0.42, "median_llm_brier": 0.18}, ...}
                "no_example_median_brier": no_example_brier,
            },
        }

    # 2) Aggregate across questions

    # Overall Superforecaster metric: average of per-question median-SF Briers
    sf_median_briers_all_q = [
        qres["sf"]["median_brier"]
        for qres in per_question_results.values()
        if qres["sf"]["median_brier"] is not None
    ]
    avg_sf_median_brier = average_across_questions(sf_median_briers_all_q)
    print(
        f"[SF] Average Brier of the *median SF forecast* across questions: "
        f"{avg_sf_median_brier:.4f}  (n={len(sf_median_briers_all_q)})"
    )

    # Public forecasters overall metric
    public_median_briers_all_q = [
        qres["public"]["median_brier"]
        for qres in per_question_results.values()
        if qres["public"]["median_brier"] is not None
    ]
    avg_public_median_brier = average_across_questions(public_median_briers_all_q)
    print(
        f"[Public] Average Brier of the *median Public forecast* across questions: "
        f"{avg_public_median_brier:.4f}  (n={len(public_median_briers_all_q)})"
    )

    # LLM overall metric: average of per-question no-example LLM Briers
    llm_baseline_briers_all_q = [
        qres["llm"]["no_example_median_brier"]
        for qres in per_question_results.values()
        if qres["llm"]["no_example_median_brier"] is not None
    ]
    avg_llm_baseline_brier = average_across_questions(llm_baseline_briers_all_q)
    print(
        f"[LLM] Average Brier of the no-example *median LLM forecast* across questions: "
        f"{avg_llm_baseline_brier:.4f}  (n={len(llm_baseline_briers_all_q)})"
    )

    # LLM per-round metric: for each round r, average the per-question median-LLM Brier at round r
    per_round_avg_llm_median_brier = aggregate_llm_rounds_across_questions(
        {
            qid: qres["llm"]["median_by_round"]
            for qid, qres in per_question_results.items()
        }
    )

    for r in sorted(per_round_avg_llm_median_brier.keys()):
        avg_brier_r = per_round_avg_llm_median_brier[r]
        # Count how many questions contributed to this round’s average
        n_q = sum(
            1
            for qres in per_question_results.values()
            if qres["llm"]["median_by_round"].get(r, {}).get("median_llm_brier")
            is not None
        )
        print(
            f"[LLM] Average Brier of the *median LLM forecast* at round {r}: "
            f"{avg_brier_r:.4f}  (n={n_q} questions)"
        )
