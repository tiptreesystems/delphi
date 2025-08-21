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
from typing import List, Dict, Any, Optional
from dataset.dataloader import ForecastDataLoader

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import debugpy
print("Waiting for debugger attach...")
debugpy.listen(5679)
debugpy.wait_for_client()
print("Debugger attached.")

results_dir_with_examples = "outputs_initial_delphi_flexible_retry_test_set"
results_dir_no_examples = "outputs_initial_delphi_flexible_retry_no_examples_test_set"
super_agent_dir_with_examples = "outputs_superagent_delphi_forecasts_flexible_retry_test_set"
results_dir_with_1_expert = "outputs_initial_delphi_flexible_retry_1_experts_test_set"
results_dir_with_2_experts = "outputs_initial_delphi_flexible_retry_2_experts_test_set"
results_dir_with_3_experts = "outputs_initial_delphi_flexible_retry_3_experts_test_set"
results_dir_with_10_experts = "outputs_initial_delphi_flexible_retry_10_experts_test_set"

initial_forecasts_dir = 'outputs_initial_forecasts_flexible_retry_test_set'

# Optional fallback in case some rounds only stored raw text responses
_PROB_PAT = re.compile(r'FINAL PROBABILITY:\s*(0?\.\d+|1\.0|0|1)', re.IGNORECASE)

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
    nums = re.findall(r'0?\.\d+|1\.0|0|1', text)
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

def parse_delphi_log_filename(filename: str) -> Tuple[str, str]:
    """
    Parse "delphi_log_<question_id>_<YYYY-MM-DD>.json" -> (question_id, date).
    Safe against '_' within the question_id.
    """
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    if not stem.startswith("delphi_log_with_examples_"):
        raise ValueError(f"Unexpected filename format: {filename}")
    parts = stem[len("delphi_log_with_examples_"):].split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse <question_id> and <date> from: {filename}")
    question_id = "_".join(parts[:-1])
    date_str = parts[-1]
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
    resolved_outcome: int
) -> Tuple[Optional[float], Optional[float]]:
    """
    Superforecasters do not have rounds.
    1) Collect all SF probabilities (one per SF).
    2) Take the **median probability** (by value).
    3) Compute Brier of that median probability vs resolved outcome.
    Returns: (median_sf_prob, median_sf_brier)
    """
    sf_forecasts = loader.get_super_forecasts(question_id=question_id, resolution_date=resolution_date)
    sf_probs = [sf.forecast for sf in sf_forecasts]
    median_sf_prob = median_by_probability(sf_probs)
    if median_sf_prob is None:
        return None, None
    return median_sf_prob, compute_brier_score(median_sf_prob, resolved_outcome)

def compute_public_median_brier(
    loader: ForecastDataLoader,
    question_id: str,
    resolution_date: str,
    resolved_outcome: int
) -> Tuple[Optional[float], Optional[float]]:
    """
    Public forecasts do not have rounds.
    1) Collect all public probabilities (one per public forecaster).
    2) Take the **median probability** (by value).
    3) Compute Brier of that median probability vs resolved outcome.
    Returns: (median_public_prob, median_public_brier)
    """
    public_forecasts = loader.get_public_forecasts(question_id=question_id, resolution_date=resolution_date)
    public_probs = [pf.forecast for pf in public_forecasts]
    median_public_prob = median_by_probability(public_probs)
    if median_public_prob is None:
        return None, None
    return median_public_prob, compute_brier_score(median_public_prob, resolved_outcome)

def compute_llm_median_brier_by_round(
    delphi_log: Dict[str, Any],
    resolved_outcome: int
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
            "median_llm_brier": compute_brier_score(median_llm_prob, resolved_outcome)
        }
    return llm_probs_by_round, out


def average_across_questions(values: List[float]) -> float:
    """Mean with explicit float cast; caller should filter out None beforehand."""
    return float(np.mean(values)) if values else float("nan")

def aggregate_llm_rounds_across_questions(
    per_question_llm: Dict[str, Dict[int, Dict[str, Optional[float]]]]
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


# Match logs by question_id and resolution_date, ignoring prefixes
def extract_id_and_date(filename: str) -> Tuple[str, str]:
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    # Remove known prefixes
    if stem.startswith("delphi_log_with_examples_"):
        stem = stem[len("delphi_log_with_examples_"):]
    elif stem.startswith("delphi_log_no_examples_"):
        stem = stem[len("delphi_log_no_examples_"):]
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse <question_id> and <date> from: {filename}")
    question_id = "_".join(parts[:-1])
    date_str = parts[-1]
    return question_id, date_str

if __name__ == "__main__":

    # 0) Setup
    loader = ForecastDataLoader()

    delphi_logs_with_examples = [
        os.path.join(results_dir_with_examples, f)
        for f in os.listdir(results_dir_with_examples)
        if f.startswith("delphi_log_with_examples") and f.endswith(".json")
    ]

    delphi_logs_no_examples = [
        os.path.join(results_dir_no_examples, f)
        for f in os.listdir(results_dir_no_examples)
        if f.startswith("delphi_log_no_examples") and f.endswith(".json")
    ]

    delphi_logs_super_agent = [
        os.path.join(super_agent_dir_with_examples, f)
        for f in os.listdir(super_agent_dir_with_examples)
        if f.startswith("delphi_log_with_examples") and f.endswith(".json")
    ]

    delphi_logs_1_experts = [
        os.path.join(results_dir_with_1_expert, f)
        for f in os.listdir(results_dir_with_1_expert)
        if f.startswith("delphi_log_with_examples") and f.endswith(".json")
    ]

    delphi_logs_two_experts = [
        os.path.join(results_dir_with_2_experts, f)
        for f in os.listdir(results_dir_with_2_experts)
        if f.startswith("delphi_log_with_examples") and f.endswith(".json")
    ]

    delphi_logs_three_experts = [
        os.path.join(results_dir_with_3_experts, f)
        for f in os.listdir(results_dir_with_3_experts)
        if f.startswith("delphi_log_with_examples") and f.endswith(".json")
    ]

    delphi_logs_ten_experts = [
        os.path.join(results_dir_with_10_experts, f)
        for f in os.listdir(results_dir_with_10_experts)
        if f.startswith("delphi_log_with_examples") and f.endswith(".json")
    ]


    # Build mapping from (question_id, date) to file for both sets
    logs_with_examples_map = {
        extract_id_and_date(f): f for f in delphi_logs_with_examples
    }
    logs_no_examples_map = {
        extract_id_and_date(f): f for f in delphi_logs_no_examples
    }

    logs_super_agent_map = {
        extract_id_and_date(f): f for f in delphi_logs_super_agent
    }

    logs_one_expert_map = {
        extract_id_and_date(f): f for f in delphi_logs_1_experts
    }

    logs_two_experts_map = {
        extract_id_and_date(f): f for f in delphi_logs_two_experts
    }

    logs_three_experts_map = {
        extract_id_and_date(f): f for f in delphi_logs_three_experts
    }

    logs_ten_experts_map = {
        extract_id_and_date(f): f for f in delphi_logs_ten_experts
    }


    # Only keep pairs present in both sets
    matched_logs = [
        (logs_with_examples_map[key], logs_no_examples_map[key], logs_super_agent_map[key], logs_one_expert_map[key], logs_two_experts_map[key], logs_three_experts_map[key], logs_ten_experts_map[key])
        for key in logs_with_examples_map.keys() & logs_no_examples_map.keys() & logs_super_agent_map.keys() & logs_one_expert_map.keys() & logs_two_experts_map.keys() & logs_three_experts_map.keys() & logs_ten_experts_map.keys()
    ]

    # Storage for per-question outputs
    per_question_results: Dict[str, Dict[str, Any]] = {}

    # 1) Process each Delphi log (i.e., each question)
    for delphi_log_file_with_examples, delphi_log_file_no_examples, delphi_log_file_superagent, delphi_log_file_one_expert, delphi_log_file_two_experts, delphi_log_file_three_experts, delphi_log_file_ten_experts in matched_logs:
        question_id, resolution_date = parse_delphi_log_filename(delphi_log_file_with_examples)

        # 1a) Ground truth outcome
        resolution = loader.get_resolution(question_id=question_id, resolution_date=resolution_date)
        y_true = resolution.resolved_to  # expected 0 or 1

        # 1b) Superforecasters (no rounds): median-by-prob -> Brier
        median_sf_prob, median_sf_brier = compute_sf_median_brier(
            loader=loader,
            question_id=question_id,
            resolution_date=resolution_date,
            resolved_outcome=y_true
        )

        # Public forecasters (no rounds): median-by-prob -> Brier
        median_public_prob, median_public_brier = compute_public_median_brier(
            loader=loader,
            question_id=question_id,
            resolution_date=resolution_date,
            resolved_outcome=y_true
        )

        # 1c) LLM (per round): median-by-prob -> Brier
        with open(delphi_log_file_with_examples, "r") as f:
            delphi_log_with_examples = json.load(f)

        llm_median_prob_by_round_with_examples, llm_median_brier_by_round_with_examples = compute_llm_median_brier_by_round(
            delphi_log=delphi_log_with_examples,
            resolved_outcome=y_true
        )

        with open(delphi_log_file_no_examples, "rb") as f:
            delphi_log_no_examples = json.load(f)

        # 1d) LLM (no examples): median-by-prob -> Brier
        _, llm_median_by_round_no_examples = compute_llm_median_brier_by_round(
            delphi_log=delphi_log_no_examples,
            resolved_outcome=y_true
        )

        with open(delphi_log_file_superagent, "r") as f:
            delphi_log_superagent = json.load(f)
        _, llm_superagent_median_by_round = compute_llm_median_brier_by_round(
            delphi_log=delphi_log_superagent,
            resolved_outcome=y_true
        )

        with open(delphi_log_file_one_expert, "r") as f:
            delphi_log_one_expert = json.load(f)
        _, llm_one_expert_median_by_round = compute_llm_median_brier_by_round(
            delphi_log=delphi_log_one_expert,
            resolved_outcome=y_true
        )

        with open(delphi_log_file_two_experts, "r") as f:
            delphi_log_two_experts = json.load(f)
        _, llm_two_experts_median_by_round = compute_llm_median_brier_by_round(
            delphi_log=delphi_log_two_experts,
            resolved_outcome=y_true
        )

        with open(delphi_log_file_three_experts, "r") as f:
            delphi_log_three_experts = json.load(f)
        _, llm_three_experts_median_by_round = compute_llm_median_brier_by_round(
            delphi_log=delphi_log_three_experts,
            resolved_outcome=y_true
        )

        with open(delphi_log_file_ten_experts, "r") as f:
            delphi_log_ten_experts = json.load(f)
        _, llm_ten_experts_median_by_round = compute_llm_median_brier_by_round(
            delphi_log=delphi_log_ten_experts,
            resolved_outcome=y_true
        )


        # load no-example baseline forecasts
        no_example_file = os.path.join(initial_forecasts_dir, f"collected_fcasts_no_examples_{resolution_date}_{question_id}.pkl")
        if not os.path.exists(no_example_file):
            print(f"Warning: No no-example forecasts found for {question_id} at {no_example_file}")
            no_example_data = {}
        else:
            with open(no_example_file, 'rb') as f:
                no_example_data = pickle.load(f)[0]
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
                "median_by_round": llm_median_brier_by_round_with_examples,
                "median_by_round_prob": llm_median_prob_by_round_with_examples,
                "median_by_round_no_examples": llm_median_by_round_no_examples,  # to be filled later
                "median_by_round_superagent": llm_superagent_median_by_round,
                "median_by_round_one_expert": llm_one_expert_median_by_round,
                "median_by_round_two_experts": llm_two_experts_median_by_round,
                "median_by_round_three_experts": llm_three_experts_median_by_round,
                "median_by_round_ten_experts": llm_ten_experts_median_by_round,
                # e.g., {0: {"median_llm_prob": 0.42, "median_llm_brier": 0.18}, ...}
                "no_example_median_brier": no_example_brier
            },
        }

        print(f"Results for question {question_id}:")
        sf = per_question_results[question_id]["sf"]
        llm = per_question_results[question_id]["llm"]["median_by_round"]
        print(f"  superforecaster_prob: {sf['median_prob']}")
        print(f"  superforecaster_brier: {sf['median_brier']}")
        print(f"  llm_median_by_round_3: {json.dumps(llm_median_brier_by_round_with_examples[3], indent=2)}")
        # print(f"  llm_median_by_round_prob: {json.dumps(llm_median_prob_by_round_with_examples, indent=2)}")

    # 2) Aggregate across questions

    import numpy as np
    import matplotlib.pyplot as plt

    def plot_brier_diff_sf_vs_llm_round3(
        per_question_results: dict,
        *,
        loader,                         # NEW: used for topic lookup
        sort_within_topic: bool = True, # optional: sort questions by diff within each topic
        save_path: str | None = None
    ):
        """
        Single bar per question showing Δ Brier = LLM(round 3) - SF, grouped by topic.
        Positive (LLM worse) => red; Negative (LLM better) => green.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        # Build topic -> list[(qid, diff)] using loader.get_topic(qid)
        topic_to_items = defaultdict(list)
        for qid, rec in per_question_results.items():
            sf_brier = rec["sf"]["median_brier"]
            try:
                llm_r3_brier = rec["llm"]["median_by_round"][3]["median_llm_brier"]
            except KeyError as e:
                raise KeyError(f"Missing round 3 LLM median for question_id={qid}") from e
            diff = llm_r3_brier - sf_brier
            topic = loader.get_topic(qid)
            topic_to_items[topic].append((qid, diff))

        # Order topics (alphabetical) and optionally sort within topic by diff
        ordered_topics = sorted(topic_to_items.keys())
        ordered_qids, ordered_diffs, topic_spans = [], [], []  # spans: (start_idx, end_idx, topic)
        idx = 0
        for topic in ordered_topics:
            items = topic_to_items[topic]
            if sort_within_topic:
                items.sort(key=lambda kv: kv[1])  # ascending by diff
            start = idx
            for qid, d in items:
                ordered_qids.append(qid)
                ordered_diffs.append(d)
                idx += 1
            end = idx - 1
            topic_spans.append((start, end, topic))

        x = np.arange(len(ordered_qids))
        colors = ["red" if d > 0 else "green" for d in ordered_diffs]

        fig, ax = plt.subplots(figsize=(max(10, len(ordered_qids) * 0.6), 5))
        ax.bar(x, ordered_diffs, color=colors)
        ax.axhline(0, color="black", linewidth=1)

        ax.set_ylabel("Δ Brier (LLM - SF)")
        ax.set_xlabel("Question ID (grouped by topic)")
        ax.set_title("Per-question Δ Brier grouped by topic — LLM Round 3 vs Superforecasters")
        ax.set_xticks(x, ordered_qids, rotation=60, ha="right")
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.margins(x=0.01)

        # Draw topic separators and labels
        for start, end, topic in topic_spans:
            if start > 0:
                ax.axvline(start - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
            mid = (start + end) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.97, topic, ha="center", va="top", fontsize=9)

        # Simple legend proxy (since bars are colored per sign)
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(facecolor="green", label="LLM better (Δ<0)"),
                    Patch(facecolor="red",   label="LLM worse (Δ>0)")],
            loc="best"
        )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
        return fig, ax

    plot_brier_diff_sf_vs_llm_round3(per_question_results, save_path='temp_plot_brier_sf_vs_llm_round3.png', loader=loader)

    # Overall Superforecaster metric: average of per-question median-SF Briers
    sf_median_briers_all_q = [
        qres["sf"]["median_brier"]
        for qres in per_question_results.values()
        if qres["sf"]["median_brier"] is not None
    ]
    avg_sf_median_brier = average_across_questions(sf_median_briers_all_q)
    print(f"[SF] Average Brier of the *median SF forecast* across questions: "
          f"{avg_sf_median_brier:.4f}  (n={len(sf_median_briers_all_q)})")

    # Public forecasters overall metric
    public_median_briers_all_q = [
        qres["public"]["median_brier"]
        for qres in per_question_results.values()
        if qres["public"]["median_brier"] is not None
    ]
    avg_public_median_brier = average_across_questions(public_median_briers_all_q)
    print(f"[Public] Average Brier of the *median Public forecast* across questions: "
          f"{avg_public_median_brier:.4f}  (n={len(public_median_briers_all_q)})")

    # LLM overall metric: average of per-question no-example LLM Briers
    llm_baseline_briers_all_q = [
        qres["llm"]["no_example_median_brier"]
        for qres in per_question_results.values()
        if qres["llm"]["no_example_median_brier"] is not None
    ]
    avg_llm_baseline_brier = average_across_questions(llm_baseline_briers_all_q)
    print(f"[LLM] Average Brier of the no-example *median LLM forecast* across questions: "
          f"{avg_llm_baseline_brier:.4f}  (n={len(llm_baseline_briers_all_q)})")

    # LLM per-round metric: for each round r, average the per-question median-LLM Brier at round r
    per_round_avg_llm_median_brier_with_examples = aggregate_llm_rounds_across_questions(
        {qid: qres["llm"]["median_by_round"] for qid, qres in per_question_results.items()}
    )

    for r in sorted(per_round_avg_llm_median_brier_with_examples.keys()):
        avg_brier_r = per_round_avg_llm_median_brier_with_examples[r]
        # Count how many questions contributed to this round’s average
        n_q = sum(
            1 for qres in per_question_results.values()
            if qres["llm"]["median_by_round"].get(r, {}).get("median_llm_brier") is not None
        )
        print(f"[LLM] Average Brier of the *median LLM forecast* at round {r}: "
              f"{avg_brier_r:.4f}  (n={n_q} questions)")

    # LLM per-round metric (no examples): for each round r, average the per-question median-LLM Brier at round r
    per_round_avg_llm_median_brier_no_examples = aggregate_llm_rounds_across_questions(
        {qid: qres["llm"]["median_by_round_no_examples"] for qid, qres in per_question_results.items()}
    )


    for r in sorted(per_round_avg_llm_median_brier_no_examples.keys()):
        avg_brier_r = per_round_avg_llm_median_brier_no_examples[r]
        # Count how many questions contributed to this round’s average
        n_q = sum(
            1 for qres in per_question_results.values()
            if qres["llm"]["median_by_round_no_examples"].get(r, {}).get("median_llm_brier") is not None
        )
        print(f"[LLM No Examples] Average Brier of the *median LLM forecast* at round {r}: "
              f"{avg_brier_r:.4f}  (n={n_q} questions)")

    per_round_avg_llm_median_brier_superagent = aggregate_llm_rounds_across_questions(
        {qid: qres["llm"]["median_by_round_superagent"] for qid, qres in per_question_results.items()}
    )

    for r in sorted(per_round_avg_llm_median_brier_superagent.keys()):
        avg_brier_r = per_round_avg_llm_median_brier_superagent[r]
        # Count how many questions contributed to this round’s average
        n_q = sum(
            1 for qres in per_question_results.values()
            if qres["llm"]["median_by_round_superagent"].get(r, {}).get("median_llm_brier") is not None
        )
        print(f"[LLM Superagent] Average Brier of the *median LLM forecast* at round {r}: "
              f"{avg_brier_r:.4f}  (n={n_q} questions)")

    per_round_avg_llm_median_brier_one_expert = aggregate_llm_rounds_across_questions(
        {qid: qres["llm"]["median_by_round_one_expert"] for qid, qres in per_question_results.items()}
    )

    for r in sorted(per_round_avg_llm_median_brier_one_expert.keys()):
        avg_brier_r = per_round_avg_llm_median_brier_one_expert[r]
        # Count how many questions contributed to this round’s average
        n_q = sum(
            1 for qres in per_question_results.values()
            if qres["llm"]["median_by_round_one_expert"].get(r, {}).get("median_llm_brier") is not None
        )
        print(f"[LLM One Expert] Average Brier of the *median LLM forecast* at round {r}: "
              f"{avg_brier_r:.4f}  (n={n_q} questions)")

    per_round_avg_llm_median_brier_two_experts = aggregate_llm_rounds_across_questions(
        {qid: qres["llm"]["median_by_round_two_experts"] for qid, qres in per_question_results.items()}
    )

    for r in sorted(per_round_avg_llm_median_brier_two_experts.keys()):
        avg_brier_r = per_round_avg_llm_median_brier_two_experts[r]
        # Count how many questions contributed to this round’s average
        n_q = sum(
            1 for qres in per_question_results.values()
            if qres["llm"]["median_by_round_two_experts"].get(r, {}).get("median_llm_brier") is not None
        )
        print(f"[LLM Two Experts] Average Brier of the *median LLM forecast* at round {r}: "
              f"{avg_brier_r:.4f}  (n={n_q} questions)")

    per_round_avg_llm_median_brier_three_experts = aggregate_llm_rounds_across_questions(
        {qid: qres["llm"]["median_by_round_three_experts"] for qid, qres in per_question_results.items()}
    )

    for r in sorted(per_round_avg_llm_median_brier_three_experts.keys()):
        avg_brier_r = per_round_avg_llm_median_brier_three_experts[r]
        # Count how many questions contributed to this round’s average
        n_q = sum(
            1 for qres in per_question_results.values()
            if qres["llm"]["median_by_round_three_experts"].get(r, {}).get("median_llm_brier") is not None
        )
        print(f"[LLM Three Experts] Average Brier of the *median LLM forecast* at round {r}: "
              f"{avg_brier_r:.4f}  (n={n_q} questions)")

    per_round_avg_llm_median_brier_ten_experts = aggregate_llm_rounds_across_questions(
        {qid: qres["llm"]["median_by_round_ten_experts"] for qid, qres in per_question_results.items()}
    )

    for r in sorted(per_round_avg_llm_median_brier_ten_experts.keys()):
        avg_brier_r = per_round_avg_llm_median_brier_ten_experts[r]
        # Count how many questions contributed to this round’s average
        n_q = sum(
            1 for qres in per_question_results.values()
            if qres["llm"]["median_by_round_ten_experts"].get(r, {}).get("median_llm_brier") is not None
        )
        print(f"[LLM Ten Experts] Average Brier of the *median LLM forecast* at round {r}: "
              f"{avg_brier_r:.4f}  (n={n_q} questions)")