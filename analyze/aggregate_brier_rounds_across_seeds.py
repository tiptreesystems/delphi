"""
Delphi Brier-curves analysis across seeds (+ SF comparisons).

Inputs
- --config: Eval YAML used to find output_dir and file name pattern.
- --base-dir (optional): Directory containing seed_* subfolders with Delphi logs. Defaults to config.experiment.output_dir
- --out (optional): Output prefix directory for artifacts (defaults to base-dir)

Modes (--mode)
- seeds   (default): Per-seed Delphi trajectories across rounds (one line per seed)
- agg              : Mean ± std across seeds (with SF/Public baseline lines)
- whisker          : Box-and-whisker per round (distribution across seeds)
- perq             : Per-question grid (3xN) showing per-seed trajectories, independent y-axis per question
- sfbar            : One bar per question, grouped by topic; bar = Brier(SF median) − Brier(Delphi final). Green if Delphi better (>0), red if SF better (<0). Error bars = std across seeds

Expert aggregation (--expert-agg)
- median (default) or mean: Aggregates expert probabilities within a round before computing Brier

Superforecaster scope (--sf-scope)
- all  (default): Use all SF forecasts for (question_id, resolution_date)
- used          : Use only SFs referenced in the Delphi log's experts (rounds[*].experts keys)

Outputs
- JSON summary: aggregate_brier_rounds.json (round means, stds, per-seed curves)
- PNG plots with the chosen mode; filenames include the expert-agg suffix, e.g.:
  - seed_brier_rounds_{median|mean}.png
  - aggregate_brier_rounds_{median|mean}.png
  - whisker_brier_rounds_{median|mean}.png
  - per_question_brier_rounds_{median|mean}.png
  - sf_diff_by_topic_{median|mean}.png

Examples
- Per-seed trajectories (median aggregator):
  uv run analyze/aggregate_brier_rounds_across_seeds.py \
    --config configs/evolution_evaluation/delphi_eval_gpt_oss_120b_5_experts_5_examples.yml \
    --base-dir results/evolution_evaluation/gpt_oss_120b_expert_system_mediator_evolved_5_experts_5_examples \
    --mode seeds --expert-agg median

- Aggregated mean ± std with SF/Public baselines:
  uv run analyze/aggregate_brier_rounds_across_seeds.py \
    --config configs/evolution_evaluation/delphi_eval_gpt_oss_120b_3_experts_3_examples.yml \
    --base-dir results/evolution_evaluation/gpt_oss_120b_expert_system_mediator_evolved_3_experts_3_examples \
    --mode agg

- Per-question grid (median, SF scope limited to used experts):
  uv run analyze/aggregate_brier_rounds_across_seeds.py \
    --config configs/evolution_evaluation/delphi_eval_gpt_oss_120b_5_experts_5_examples.yml \
    --base-dir results/evolution_evaluation/gpt_oss_120b_expert_system_mediator_evolved_5_experts_5_examples \
    --mode perq --expert-agg median --sf-scope used

- Per-question improvement bars grouped by topic (green=Delphi better):
  uv run analyze/aggregate_brier_rounds_across_seeds.py \
    --config configs/evolution_evaluation/delphi_eval_gpt_oss_120b_5_experts_5_examples.yml \
    --base-dir results/evolution_evaluation/gpt_oss_120b_expert_system_mediator_evolved_5_experts_5_examples \
    --mode sfbar --expert-agg median --sf-scope all

Suggestions
- Start with seeds + agg to understand overall performance vs. dispersion.
- Use perq to spot question-specific behaviors (e.g., flat vs. improving vs. regressing rounds) and seed sensitivity.
- Try whisker to quickly check stability across rounds and seeds.
- Use sfbar to see where Delphi gains or loses vs. SF, and which topics drive the differences. Consider --sf-scope used to match the Delphi panel exactly.
"""

from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

from dataset.dataloader import ForecastDataLoader
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

    _root = Path(__file__).resolve().parents[1]
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
import yaml


def parse_delphi_log_filename(filename: str, file_pattern: str) -> Tuple[str, str]:
    """Parse (question_id, resolution_date) using the configured file_pattern."""
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    prefix_end = file_pattern.find("{question_id}")
    if prefix_end == -1:
        raise ValueError(f"Pattern {file_pattern} must contain {{question_id}}")
    prefix = file_pattern[:prefix_end]
    if not stem.startswith(prefix):
        raise ValueError(f"Filename {filename} doesn't match pattern {file_pattern}")
    remainder = stem[len(prefix) :]
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    date_matches = re.findall(date_pattern, remainder)
    if not date_matches:
        raise ValueError(f"Cannot find date in filename: {filename}")
    date_str = date_matches[-1]
    date_pos = remainder.rfind("_" + date_str)
    if date_pos == -1:
        date_pos = remainder.rfind(date_str)
        if date_pos == -1:
            raise ValueError(f"Cannot parse question_id and date from: {filename}")
        question_id = remainder[:date_pos].rstrip("_")
    else:
        question_id = remainder[:date_pos]
    return question_id, date_str


def collect_round_probs(delphi_log: Dict[str, Any]) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    rounds = delphi_log.get("rounds", [])
    for r in rounds:
        r_idx = int(r.get("round", 0))
        expert_dict = r.get("experts", {})
        probs: List[float] = []
        for _, entry in (expert_dict or {}).items():
            p = entry.get("prob")
            if isinstance(p, (int, float)):
                probs.append(float(p))
        out[r_idx] = probs
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def median(lst: List[float]) -> Optional[float]:
    vals = [float(x) for x in lst if isinstance(x, (int, float))]
    if not vals:
        return None
    vals.sort()
    return float(vals[len(vals) // 2])


def mean(lst: List[float]) -> Optional[float]:
    vals = [float(x) for x in lst if isinstance(x, (int, float))]
    if not vals:
        return None
    return float(np.mean(vals))


def compute_seed_curves(
    seed_dir: Path,
    file_pattern: str,
    loader: ForecastDataLoader,
    expert_agg: str = "median",
    sf_scope: str = "all",  # 'all' or 'used'
    allowed_qids: Optional[set] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, float], Optional[float], Optional[float]]:
    """
    Returns:
      per_round_mean_brier: {round_idx: mean brier across questions in this seed}
      sf_mean_brier: average Brier of median superforecaster across questions
      public_mean_brier: average Brier of median public across questions
    """
    # Gather logs matching pattern
    # Recursively gather logs under seed_dir to support set subfolders (train/eval/evolution_eval)
    prefix = file_pattern.split("{")[0]
    logs = [
        p for p in seed_dir.rglob("*.json") if p.is_file() and p.name.startswith(prefix)
    ]
    if verbose:
        print(
            f"[seed {seed_dir.name}] Found {len(logs)} candidate log(s) under {seed_dir}"
        )
    per_round_accum: Dict[int, List[float]] = {}
    sf_briers: List[float] = []
    pub_briers: List[float] = []
    expected_rounds: Optional[set[int]] = None

    for fp in logs:
        try:
            qid, res_date = parse_delphi_log_filename(fp.name, file_pattern)
        except Exception:
            continue
        if allowed_qids is not None and qid not in allowed_qids:
            if verbose:
                print(f"  [skip/not-in-set] {fp.name}")
            continue
        try:
            res = loader.get_resolution(qid, res_date)
            if not res or not getattr(res, "resolved", False):
                if verbose:
                    print(f"  [skip/no-resolution] {qid}@{res_date}")
                continue
            actual = float(res.resolved_to)
        except Exception:
            if verbose:
                print(f"  [skip/error-resolution] {qid}@{res_date}")
            continue

        try:
            with fp.open("r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            if verbose:
                print(f"  [skip/json-error] {fp.name}")
            continue

        # Per-round LLM median-of-experts brier
        probs_by_round = collect_round_probs(log)
        rounds_set = set(probs_by_round.keys())
        if expected_rounds is None:
            expected_rounds = rounds_set
        elif rounds_set != expected_rounds:
            raise RuntimeError(
                f"Inconsistent rounds in {seed_dir.name}: expected {sorted(expected_rounds)}, got {sorted(rounds_set)} for {fp.name}"
            )
        for r_idx, probs in probs_by_round.items():
            m = median(probs) if expert_agg == "median" else mean(probs)
            if m is None:
                continue
            b = (m - actual) ** 2
            per_round_accum.setdefault(r_idx, []).append(b)

        # Baselines (median across forecasters)
        try:
            sfs = loader.get_super_forecasts(question_id=qid, resolution_date=res_date)
            if sf_scope == "used":
                # Limit to SF IDs present in the Delphi log's experts
                used_ids = set()
                for rnd in log.get("rounds", []) or []:
                    used_ids.update((rnd.get("experts") or {}).keys())
                sfs = [s for s in sfs if s.user_id in used_ids]
            sf_probs = [float(s.forecast) for s in sfs]
            m_sf = median(sf_probs)
            if m_sf is not None:
                sf_briers.append((m_sf - actual) ** 2)
        except Exception:
            if verbose:
                print(f"  [warn] SF baseline failed for {qid}@{res_date}")
            pass
        try:
            pubs = loader.get_public_forecasts(
                question_id=qid, resolution_date=res_date
            )
            pub_probs = [float(p.forecast) for p in pubs]
            m_pub = median(pub_probs)
            if m_pub is not None:
                pub_briers.append((m_pub - actual) ** 2)
        except Exception:
            if verbose:
                print(f"  [warn] Public baseline failed for {qid}@{res_date}")
            pass

    per_round_mean_brier = {
        r: float(np.mean(v)) for r, v in per_round_accum.items() if v
    }
    sf_mean_brier = float(np.mean(sf_briers)) if sf_briers else None
    public_mean_brier = float(np.mean(pub_briers)) if pub_briers else None
    if verbose:
        counts = {r: len(v) for r, v in per_round_accum.items()}
        print(
            f"[seed {seed_dir.name}] rounds={sorted(list(per_round_mean_brier.keys()))} counts={counts} SFn={len(sf_briers)} Pubn={len(pub_briers)}"
        )
    return per_round_mean_brier, sf_mean_brier, public_mean_brier


def compute_per_question_curves(
    seed_dir: Path,
    file_pattern: str,
    loader: ForecastDataLoader,
    expert_agg: str = "median",
    allowed_qids: Optional[set] = None,
    verbose: bool = False,
) -> Dict[str, Dict[int, float]]:
    """
    For a single seed dir, compute per-question per-round median-of-experts Brier.
    Returns: { question_id: { round_idx: mean_brier_across_logs } }
    """
    # Recursively gather logs under seed_dir to support set subfolders
    prefix = file_pattern.split("{")[0]
    logs = [
        p for p in seed_dir.rglob("*.json") if p.is_file() and p.name.startswith(prefix)
    ]
    if verbose:
        print(f"[perq {seed_dir.name}] scanning {len(logs)} log(s)")
    per_q_round_accum: Dict[str, Dict[int, List[float]]] = {}

    for fp in logs:
        try:
            qid, res_date = parse_delphi_log_filename(fp.name, file_pattern)
        except Exception:
            continue
        if allowed_qids is not None and qid not in allowed_qids:
            continue
        try:
            res = loader.get_resolution(qid, res_date)
            if not res or not getattr(res, "resolved", False):
                continue
            actual = float(res.resolved_to)
        except Exception:
            continue
        try:
            with fp.open("r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            continue
        probs_by_round = collect_round_probs(log)
        for r_idx, probs in probs_by_round.items():
            m = median(probs) if expert_agg == "median" else mean(probs)
            if m is None:
                continue
            b = (m - actual) ** 2
            per_q_round_accum.setdefault(qid, {}).setdefault(r_idx, []).append(b)

    per_q_curves: Dict[str, Dict[int, float]] = {}
    for qid, rmap in per_q_round_accum.items():
        per_q_curves[qid] = {
            r: float(np.mean(vals)) for r, vals in rmap.items() if vals
        }
    if verbose:
        print(f"[perq {seed_dir.name}] questions aggregated: {len(per_q_curves)}")
    return per_q_curves


def compute_seed_question_improvement(
    seed_dir: Path,
    file_pattern: str,
    loader: ForecastDataLoader,
    expert_agg: str = "median",
    sf_scope: str = "all",  # 'all' or 'used'
    allowed_qids: Optional[set] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    For each question in a seed, compute improvement = Brier(SF median) - Brier(Delphi final).
    Positive values mean Delphi outperforms SF. Returns mapping:
      { qid: { 'improvement': float, 'topic': str } }
    """
    # Recursively gather logs under seed_dir to support set subfolders
    prefix = file_pattern.split("{")[0]
    logs = [
        p for p in seed_dir.rglob("*.json") if p.is_file() and p.name.startswith(prefix)
    ]
    if verbose:
        print(f"[sfbar {seed_dir.name}] scanning {len(logs)} log(s)")
    out: Dict[str, Dict[str, Any]] = {}
    for fp in logs:
        try:
            qid, res_date = parse_delphi_log_filename(fp.name, file_pattern)
        except Exception:
            continue
        if allowed_qids is not None and qid not in allowed_qids:
            continue
        # Resolution
        try:
            res = loader.get_resolution(qid, res_date)
            if not res or not getattr(res, "resolved", False):
                continue
            actual = float(res.resolved_to)
        except Exception:
            continue
        # Load log
        try:
            with fp.open("r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            continue
        rounds = log.get("rounds") or []
        if not rounds:
            continue
        final_round = rounds[-1]
        probs = [
            float(entry.get("prob"))
            for entry in (final_round.get("experts") or {}).values()
            if isinstance(entry.get("prob"), (int, float))
        ]
        if not probs:
            continue
        d_pred = median(probs) if expert_agg == "median" else mean(probs)
        delphi_brier = (d_pred - actual) ** 2
        # SF median + Brier
        sfs = loader.get_super_forecasts(question_id=qid, resolution_date=res_date)
        if sf_scope == "used":
            used_ids = set()
            for rnd in log.get("rounds", []) or []:
                used_ids.update((rnd.get("experts") or {}).keys())
            sfs = [s for s in sfs if s.user_id in used_ids]
        sf_probs = [float(s.forecast) for s in sfs]
        if not sf_probs:
            continue
        sf_m = median(sf_probs)
        if sf_m is None:
            continue
        sf_brier = (sf_m - actual) ** 2
        public_forecasts = loader.get_public_forecasts(
            question_id=qid, resolution_date=res_date
        )
        pub_probs = [float(p.forecast) for p in public_forecasts]
        if not pub_probs:
            continue
        try:
            topic = getattr(loader.get_question(qid), "topic", None) or "unknown"
        except Exception:
            topic = "unknown"
        beats_sf = delphi_brier < sf_brier
        out[qid] = {
            "improvement": float(sf_brier - delphi_brier),
            "topic": topic,
            "beats_sf": beats_sf,
        }
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate Delphi Brier curves across seeds"
    )
    ap.add_argument("--config", required=True, help="Path to eval config YAML")
    ap.add_argument(
        "--base-dir",
        default=None,
        help="Override base results dir (defaults to config experiment.output_dir)",
    )
    ap.add_argument(
        "--out", default=None, help="Override output prefix (defaults to base-dir)"
    )
    ap.add_argument(
        "--mode",
        default="seeds",
        choices=["seeds", "agg", "whisker", "sfbar"],
        help="Plot mode: per-seed lines, aggregated mean±std, whisker, or SF diff bar by topic",
    )
    ap.add_argument(
        "--perq",
        action="store_true",
        help="Render per-question subplots (3xN grid) for the selected mode",
    )
    ap.add_argument(
        "--perq-ylims",
        default="auto",
        choices=["auto", "01"],
        help="Per-question y-axis limits: auto (fit data) or 01 (fixed [0,1])",
    )
    ap.add_argument(
        "--expert-agg",
        default="median",
        choices=["median", "mean"],
        help="Aggregate expert probabilities per round with median (default) or mean",
    )
    ap.add_argument(
        "--sf-scope",
        default="all",
        choices=["all", "used"],
        help="Use all SFs at resolution date (all) or only those referenced in the Delphi log's experts (used)",
    )
    ap.add_argument(
        "--set",
        default="auto",
        choices=["auto", "train", "eval", "evolution_eval"],
        help="Override YAML sampling method: choose question set explicitly",
    )
    ap.add_argument(
        "--ignore-qid",
        nargs="+",
        default=None,
        help="List of question IDs to ignore in validation and results",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    cfg = load_typed_experiment_config(args.config)
    base_dir = Path(args.base_dir or cfg["experiment"]["output_dir"])
    file_pattern = cfg["output"]["file_pattern"]
    # Sampling-method alignment or CLI override
    sampling_method = (cfg.get("data", {}).get("sampling", {}) or {}).get("method")
    allowed_qids: Optional[set] = None
    set_name = "auto"
    if args.set != "auto":
        set_name = args.set
        if args.set == "train":
            allowed_qids = set(TRAIN_QUESTION_IDS)
        elif args.set == "eval":
            allowed_qids = set(EVALUATION_QUESTION_IDS)
        elif args.set == "evolution_eval":
            allowed_qids = set(EVOLUTION_EVALUATION_QUESTION_IDS)
    else:
        if sampling_method == "evaluation":
            allowed_qids = set(EVALUATION_QUESTION_IDS)
            set_name = "eval"
        elif sampling_method == "evolution_evaluation":
            allowed_qids = set(EVOLUTION_EVALUATION_QUESTION_IDS)
            set_name = "evolution_eval"
        elif sampling_method in ("train", "train_small"):
            allowed_qids = set(TRAIN_QUESTION_IDS)
            set_name = "train"

    # Build ignore set and apply to allowed set if provided
    ignore_set = set(args.ignore_qid) if args.ignore_qid else set()
    if allowed_qids is not None:
        allowed_qids = allowed_qids - ignore_set

    # Discover seed dirs
    seed_dirs = [
        d for d in base_dir.iterdir() if d.is_dir() and re.match(r"^seed_\d+$", d.name)
    ]
    if not seed_dirs:
        print(f"No seed_* directories under {base_dir}")
        return
    seed_dirs.sort(key=lambda p: p.name)

    loader = ForecastDataLoader()
    # If only ignoring without a defined set, enforce filtering by universe minus ignore_set
    if allowed_qids is None and ignore_set:
        try:
            all_qids = set(getattr(loader, "questions", {}).keys())
        except Exception:
            all_qids = set()
        if all_qids:
            allowed_qids = all_qids - ignore_set
    if args.verbose:
        print(f"Loaded config: {cfg}")
        if ignore_set:
            print(
                f"Ignoring {len(ignore_set)} question IDs (will be excluded from validation and results)"
            )

    # --- Strict validation of question IDs per seed ---
    # Gather qids present in each seed for the selected set only and cross-check
    prefix = file_pattern.split("{")[0]
    seed_qids: Dict[str, set[str]] = {}
    # Determine preferred subfolder to scan when a set is selected
    preferred_subdir = None
    if allowed_qids is not None and set_name in {"eval", "evolution_eval", "train"}:
        preferred_subdir = set_name
    for sd in seed_dirs:
        # Limit scan to the preferred subdir if present (avoids mixing sets)
        scan_roots = [sd]
        if preferred_subdir and (sd / preferred_subdir).is_dir():
            scan_roots = [sd / preferred_subdir]

        logs = []
        for root in scan_roots:
            logs.extend(
                [
                    p
                    for p in root.rglob("*.json")
                    if p.is_file() and p.name.startswith(prefix)
                ]
            )
        qid_counts: Dict[str, int] = {}
        qids: set[str] = set()
        for fp in logs:
            try:
                qid, _ = parse_delphi_log_filename(fp.name, file_pattern)
            except Exception:
                continue
            # When a set is selected, ignore logs whose qid is outside that set
            if allowed_qids is not None and qid not in allowed_qids:
                continue
            qids.add(qid)
            qid_counts[qid] = qid_counts.get(qid, 0) + 1

        # Duplicates within a seed
        dups = [qid for qid, cnt in qid_counts.items() if cnt > 1]
        if dups:
            raise RuntimeError(
                f"Found duplicate logs for questions in {sd}: {dups[:10]} (+{max(0, len(dups) - 10)} more)"
            )

        # Allowed set alignment: no extraneous, and (if provided) should not exceed allowed set
        if allowed_qids is not None:
            extraneous = sorted(list(qids - allowed_qids))
            missing = sorted(list(allowed_qids - qids))
            if extraneous:
                raise RuntimeError(
                    f"Seed {sd.name} contains questions outside selected set {len(extraneous)} examples. Example IDs: {extraneous[:10]}"
                )
            if missing:
                raise RuntimeError(
                    f"Seed {sd.name} is missing {len(missing)} questions from selected set. Example IDs: {missing[:10]}"
                )
        seed_qids[sd.name] = qids

    # Cross-seed alignment: require identical qid sets across seeds
    if seed_qids:
        seed_items = list(seed_qids.items())
        base_seed, base_set = seed_items[0]
        for name, s in seed_items[1:]:
            if s != base_set:
                only_in_base = sorted(list(base_set - s))
                only_in_this = sorted(list(s - base_set))
                raise RuntimeError(
                    f"Question sets differ between seeds. Seed {base_seed} vs {name}. "
                    f"Only in {base_seed}: {only_in_base[:10]} | Only in {name}: {only_in_this[:10]}"
                )
    per_seed_curves: Dict[str, Dict[int, float]] = {}
    sf_list: List[float] = []
    pub_list: List[float] = []

    # Accumulate per round across seeds
    round_to_seed_values: Dict[int, List[float]] = {}
    for sd in seed_dirs:
        curve, sf_b, pub_b = compute_seed_curves(
            sd,
            file_pattern,
            loader,
            expert_agg=args.expert_agg,
            sf_scope=args.sf_scope,
            allowed_qids=allowed_qids,
        )
        per_seed_curves[sd.name] = curve
        if sf_b is not None:
            sf_list.append(sf_b)
        if pub_b is not None:
            pub_list.append(pub_b)
        for r, v in curve.items():
            round_to_seed_values.setdefault(r, []).append(v)

    rounds_sorted = sorted(round_to_seed_values.keys())
    mean_curve = (
        [float(np.mean(round_to_seed_values[r])) for r in rounds_sorted]
        if rounds_sorted
        else []
    )
    std_curve = (
        [float(np.std(round_to_seed_values[r])) for r in rounds_sorted]
        if rounds_sorted
        else []
    )

    sf_mean = float(np.mean(sf_list)) if sf_list else None
    sf_std = float(np.std(sf_list)) if sf_list else None
    pub_mean = float(np.mean(pub_list)) if pub_list else None
    pub_std = float(np.std(pub_list)) if pub_list else None

    # Save JSON summary
    out_prefix = Path(args.out) if args.out else base_dir
    out_prefix = out_prefix / f"set_{set_name}"
    out_prefix.mkdir(parents=True, exist_ok=True)
    summary = {
        "rounds": rounds_sorted,
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "sf_baseline": {"mean": sf_mean, "std": sf_std},
        "public_baseline": {"mean": pub_mean, "std": pub_std},
        "per_seed": {
            seed: {str(r): v for r, v in curve.items()}
            for seed, curve in per_seed_curves.items()
        },
    }
    out_json = (
        Path(out_prefix)
        / f"aggregate_brier_rounds_{args.expert_agg}_{args.sf_scope}_{set_name}.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON -> {out_json}")

    # Plot
    # Per-question toggle: render grid for selected mode (except sfbar)
    if args.perq and args.mode != "sfbar":
        # Build per-question curves per seed
        per_seed_perq: Dict[str, Dict[str, Dict[int, float]]] = {}
        all_qids: set[str] = set()
        for sd in seed_dirs:
            perq = compute_per_question_curves(
                sd,
                file_pattern,
                loader,
                expert_agg=args.expert_agg,
                allowed_qids=allowed_qids,
                verbose=args.verbose,
            )
            per_seed_perq[sd.name] = perq
            all_qids.update(perq.keys())

        if not all_qids:
            print("No per-question data found to plot.")
            return

        qids_sorted = sorted(all_qids)

        # Compute per-question SF/Public Brier baselines for annotation
        qid_to_sf_brier: Dict[str, Optional[float]] = {}
        qid_to_pub_brier: Dict[str, Optional[float]] = {}
        # Build a mapping qid -> resolution_date by scanning logs (restricted to selected set)
        qid_to_res_date: Dict[str, str] = {}
        prefix = file_pattern.split("{")[0]
        for sd in seed_dirs:
            # Prefer set-specific subfolder to avoid mixing
            scan_roots = [sd]
            preferred_subdir = None
            if allowed_qids is not None and set_name in {
                "eval",
                "evolution_eval",
                "train",
            }:
                preferred_subdir = set_name
            if preferred_subdir and (sd / preferred_subdir).is_dir():
                scan_roots = [sd / preferred_subdir]
            for root in scan_roots:
                for fp in root.rglob("*.json"):
                    if not (fp.is_file() and fp.name.startswith(prefix)):
                        continue
                    try:
                        qid, res_date = parse_delphi_log_filename(fp.name, file_pattern)
                    except Exception:
                        continue
                    if allowed_qids is not None and qid not in allowed_qids:
                        continue
                    if qid not in qid_to_res_date:
                        qid_to_res_date[qid] = res_date
            # If we have all questions mapped, we can stop early
            if len(qid_to_res_date) >= len(qids_sorted):
                break

        for qid in qids_sorted:
            res_date = qid_to_res_date.get(qid)
            if not res_date:
                # Fallback to config date
                res_date = cfg.get("data", {}).get("resolution_date")
            try:
                res = loader.get_resolution(qid, res_date)
                actual = (
                    float(res.resolved_to)
                    if res and getattr(res, "resolved", False)
                    else None
                )
            except Exception:
                actual = None
            # SF baseline
            sf_brier = None
            pub_brier = None
            if actual is not None:
                try:
                    sfs = loader.get_super_forecasts(
                        question_id=qid, resolution_date=res_date
                    )
                    sf_probs = [float(s.forecast) for s in sfs]
                    m_sf = np.median(sf_probs) if sf_probs else None
                    if m_sf is not None:
                        sf_brier = float((m_sf - actual) ** 2)
                except Exception:
                    pass
                try:
                    pubs = loader.get_public_forecasts(
                        question_id=qid, resolution_date=res_date
                    )
                    pub_probs = [float(p.forecast) for p in pubs]
                    m_pub = np.median(pub_probs) if pub_probs else None
                    if m_pub is not None:
                        pub_brier = float((m_pub - actual) ** 2)
                except Exception:
                    pass
            qid_to_sf_brier[qid] = sf_brier
            qid_to_pub_brier[qid] = pub_brier
        n_q = len(qids_sorted)
        n_cols = 6
        n_rows = math.ceil(n_q / n_cols) if n_q > 0 else 1
        # Independent Y axes per question
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 3.2, n_rows * 2.4),
            sharex=True,
            sharey=False,
        )
        axes_grid = axes if n_rows > 1 else np.array(axes).reshape(1, -1)
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

        for idx, qid in enumerate(qids_sorted):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes_grid[r, c]

            if args.mode == "seeds":
                seed_items = sorted(per_seed_perq.items(), key=lambda kv: kv[0])
                for i, (seed_name, perq_map) in enumerate(seed_items):
                    curve = perq_map.get(qid, {})
                    if not curve:
                        continue
                    rs = sorted(curve.keys())
                    xs = [ri + 1 for ri in rs]
                    ys = [curve[ri] for ri in rs]
                    color = color_cycle[i % len(color_cycle)] if color_cycle else None
                    ax.plot(xs, ys, label=seed_name, color=color, linewidth=1.4)
            else:
                # Aggregate across seeds per round for this question
                r_to_vals: Dict[int, List[float]] = {}
                for seed_name, perq_map in per_seed_perq.items():
                    curve = perq_map.get(qid, {})
                    for ri, v in curve.items():
                        r_to_vals.setdefault(ri, []).append(v)
                rs = sorted(r_to_vals.keys())
                xs = [ri + 1 for ri in rs]
                if args.mode == "agg":
                    means = [float(np.mean(r_to_vals[ri])) for ri in rs]
                    stds = [float(np.std(r_to_vals[ri])) for ri in rs]
                    ax.plot(xs, means, color="C0", label="mean across seeds")
                    ax.fill_between(
                        xs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color="C0",
                        alpha=0.2,
                    )
                elif args.mode == "whisker":
                    data = [r_to_vals[ri] for ri in rs]
                    bp = ax.boxplot(
                        data,
                        positions=xs,
                        widths=0.5,
                        showmeans=False,
                        patch_artist=True,
                    )
                    for box in bp["boxes"]:
                        box.set(facecolor="#6baed6", alpha=0.4)
                    for median_line in bp["medians"]:
                        median_line.set(color="#08519c", linewidth=2)

            # Add per-question baseline Briers on each subplot
            sf_q = qid_to_sf_brier.get(qid)
            pub_q = qid_to_pub_brier.get(qid)
            if sf_q is not None:
                ax.axhline(sf_q, color="C2", linestyle="--", linewidth=0.9)
            if pub_q is not None:
                ax.axhline(pub_q, color="C3", linestyle="-.", linewidth=0.9)
            # Title includes per-question baseline Briers when available
            title = qid[:8]
            parts = []
            if sf_q is not None:
                parts.append(f"SF {sf_q:.3f}")
            if pub_q is not None:
                parts.append(f"Pub {pub_q:.3f}")
            if parts:
                title = f"{title} | " + "  ".join(parts)
            ax.set_title(title, fontsize=9)
            # Apply y-limits policy for per-question plots
            if args.perq_ylims == "01":
                ax.set_ylim(-0.02, 1.02)
            ax.grid(True, axis="y", alpha=0.3)
            if r == n_rows - 1:
                ax.set_xlabel("Round")
            if c == 0:
                ax.set_ylabel("Brier")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_png = (
            Path(out_prefix)
            / f"per_question_brier_rounds_{args.mode}_{args.expert_agg}_{args.sf_scope}_{set_name}.png"
        )
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot -> {out_png}")
        return

    if args.mode == "seeds":
        # Plot each seed’s curve separately
        plt.figure(figsize=(8, 4))
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, (seed, curve) in enumerate(
            sorted(per_seed_curves.items(), key=lambda kv: kv[0])
        ):
            if not curve:
                continue
            rs = sorted(curve.keys())
            xs = [r + 1 for r in rs]
            ys = [curve[r] for r in rs]
            color = color_cycle[i % len(color_cycle)] if color_cycle else None
            # line with markers at each round
            plt.plot(xs, ys, label=seed, color=color, marker="o")
        # Baselines as flat lines
        if sf_mean is not None:
            plt.axhline(
                sf_mean,
                color="C2",
                linestyle="--",
                label=f"SF median: {sf_mean:.3f}±{(sf_std or 0):.3f}",
            )
        if pub_mean is not None:
            plt.axhline(
                pub_mean,
                color="C3",
                linestyle="-.",
                label=f"Public median: {pub_mean:.3f}±{(pub_std or 0):.3f}",
            )
        plt.xlabel("Round")
        plt.ylabel("Brier (lower is better)")
        plt.title("Delphi median Brier across rounds (per-seed trajectories)")
        plt.grid(True, axis="y", alpha=0.3)
        # X-axis shows only integer rounds
        if rounds_sorted:
            plt.xticks([r + 1 for r in rounds_sorted])
        plt.legend(ncol=2)
        out_png = (
            Path(out_prefix)
            / f"seed_brier_rounds_{args.expert_agg}_{args.sf_scope}_{set_name}.png"
        )
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot -> {out_png}")
    elif args.mode == "agg":
        xs = [r + 1 for r in rounds_sorted]  # 1-indexed for display
        plt.figure(figsize=(8, 4))
        # Mean curve with vertical error bars (std) and markers at each round
        plt.errorbar(
            xs,
            mean_curve,
            yerr=std_curve,
            fmt="-o",
            color="C0",
            capsize=3,
            label="Delphi median (mean±std across seeds)",
        )
        # Baselines as flat lines
        if sf_mean is not None:
            plt.axhline(
                sf_mean,
                color="C2",
                linestyle="--",
                label=f"SF median (mean±std): {sf_mean:.3f}±{(sf_std or 0):.3f}",
            )
        if pub_mean is not None:
            plt.axhline(
                pub_mean,
                color="C3",
                linestyle="-.",
                label=f"Public median (mean±std): {pub_mean:.3f}±{(pub_std or 0):.3f}",
            )
        plt.xlabel("Round")
        plt.ylabel("Brier (lower is better)")
        plt.title("Delphi median Brier across rounds (aggregated across seeds)")
        plt.grid(True, axis="y", alpha=0.3)
        # X-axis shows only integer rounds
        if rounds_sorted:
            plt.xticks(xs)
        plt.legend()
        out_png = (
            Path(out_prefix)
            / f"aggregate_brier_rounds_{args.expert_agg}_{args.sf_scope}_{set_name}.png"
        )
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot -> {out_png}")
    elif args.mode == "whisker":
        # Whisker/box plot across seeds per round
        if not rounds_sorted:
            print("No round data to plot.")
            return
        xs = [r + 1 for r in rounds_sorted]
        data = [round_to_seed_values[r] for r in rounds_sorted]
        plt.figure(figsize=(8, 4))
        bp = plt.boxplot(
            data, positions=xs, widths=0.5, showmeans=False, patch_artist=True
        )
        # Light styling
        for box in bp["boxes"]:
            box.set(facecolor="#6baed6", alpha=0.4)
        for whisker in bp["whiskers"]:
            whisker.set(color="#3182bd")
        for cap in bp["caps"]:
            cap.set(color="#3182bd")
        for median_line in bp["medians"]:
            median_line.set(color="#08519c", linewidth=2)
        # Baselines as flat lines
        if sf_mean is not None:
            plt.axhline(
                sf_mean,
                color="C2",
                linestyle="--",
                label=f"SF median: {sf_mean:.3f}±{(sf_std or 0):.3f}",
            )
        if pub_mean is not None:
            plt.axhline(
                pub_mean,
                color="C3",
                linestyle="-.",
                label=f"Public median: {pub_mean:.3f}±{(pub_std or 0):.3f}",
            )
        plt.xlabel("Round")
        plt.ylabel("Brier (lower is better)")
        plt.title("Delphi median Brier across rounds (box-and-whisker across seeds)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.xticks(xs)
        if sf_mean is not None or pub_mean is not None:
            plt.legend()
        out_png = (
            Path(out_prefix)
            / f"whisker_brier_rounds_{args.expert_agg}_{args.sf_scope}_{set_name}.png"
        )
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot -> {out_png}")
    elif args.mode == "sfbar":
        # Per-question bar chart: improvement = Brier(SF median) - Brier(Delphi final), grouped by topic
        # Positive bars (green) => Delphi better; negative (red) => SF better
        q_to_vals: Dict[str, List[float]] = {}
        q_to_topic: Dict[str, str] = {}
        for sd in seed_dirs:
            perq = compute_seed_question_improvement(
                sd,
                file_pattern,
                loader,
                expert_agg=args.expert_agg,
                sf_scope=args.sf_scope,
                allowed_qids=allowed_qids,
                verbose=args.verbose,
            )
            for qid, payload in perq.items():
                q_to_vals.setdefault(qid, []).append(payload["improvement"])
                q_to_topic[qid] = payload.get("topic", "unknown")
        if not q_to_vals:
            print("No data for sfbar mode.")
            return
        # Compute mean improvement per question
        qids = sorted(q_to_vals.keys())
        q_mean = {qid: float(np.mean(vals)) for qid, vals in q_to_vals.items()}
        q_std = {qid: float(np.std(vals)) for qid, vals in q_to_vals.items()}
        # Group by topic
        topic_to_qs: Dict[str, List[str]] = {}
        for qid in qids:
            topic = q_to_topic.get(qid, "unknown")
            topic_to_qs.setdefault(topic, []).append(qid)
        topics = sorted(topic_to_qs.keys())
        # Build positions with gaps between topics
        positions: List[float] = []
        labels: List[str] = []
        colors: List[str] = []
        heights: List[float] = []
        yerr: List[float] = []
        topic_centers: Dict[str, float] = {}
        separators: List[
            float
        ] = []  # x-positions for vertical dashed lines between topics
        x = 0.0
        gap = 0.6
        for t in topics:
            qs = sorted(topic_to_qs[t])
            start = x
            for qid in qs:
                positions.append(x)
                labels.append(qid[:8])
                h = q_mean[qid]
                heights.append(h)
                yerr.append(q_std[qid])
                colors.append("#2ca02c" if h >= 0 else "#d62728")
                x += 1.0
            end = x - 1.0
            topic_centers[t] = (start + end) / 2.0 if end >= start else start
            # record a separator half-way through the gap (except after last topic)
            x += gap  # gap between topics
            separators.append(x - gap / 2.0)
        if separators:
            separators = separators[:-1]  # remove trailing separator after last topic
        plt.figure(figsize=(max(10, 0.5 * len(positions)), 5))
        plt.bar(positions, heights, yerr=yerr, capsize=2, color=colors, alpha=0.9)
        plt.axhline(0.0, color="#444", linewidth=1)
        plt.xticks(positions, labels, rotation=45, ha="right", fontsize=8)
        # Determine y-limits with padding
        if heights:
            y_upper = (
                max((h + e) for h, e in zip(heights, yerr)) if yerr else max(heights)
            )
            y_lower = (
                min((h - e) for h, e in zip(heights, yerr)) if yerr else min(heights)
            )
            pad = 0.1 * max(abs(y_upper), abs(y_lower), 1e-3)
            plt.ylim(y_lower - pad, y_upper + pad)
            label_y = y_upper + 0.2 * pad
        else:
            label_y = 0.05
        # Vertical dashed separators between topic groups
        for sx in separators:
            plt.axvline(sx, color="#999", linestyle="--", linewidth=1, alpha=0.6)
        # Topic labels above groups
        for t, cx in topic_centers.items():
            plt.text(cx, label_y, t, ha="center", va="bottom", fontsize=9)
        plt.ylabel("Improvement: Brier(SF) − Brier(Delphi)")
        plt.title(
            "Per-question improvement by topic (green: Delphi better, red: SF better)"
        )
        plt.tight_layout()
        out_png = (
            Path(out_prefix)
            / f"sf_diff_by_topic_{args.expert_agg}_{args.sf_scope}_{set_name}.png"
        )
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot -> {out_png}")
    # (per-question grid handled above when args.perq is set)


if __name__ == "__main__":
    main()
