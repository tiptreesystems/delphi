#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import debugpy
import psutil

# def _is_debugpy_running(port=5679):
#     """Check if debugpy is already listening on the given port."""
#     for proc in psutil.process_iter(attrs=["cmdline"]):
#         try:
#             cmdline = proc.info["cmdline"]
#             if cmdline and any("debugpy" in arg for arg in cmdline) and str(port) in " ".join(cmdline):
#                 return True
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             continue
#     return False

# if not _is_debugpy_running():
#     import debugpy
#     print("Waiting for debugger attach...")
#     debugpy.listen(5679)
#     debugpy.wait_for_client()
#     print("Debugger attached.")


PATTERN = re.compile(r"FINAL\s+PROBABILITY:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)


def compute_stats(obj: Dict[str, Any]):
    """
    Returns:
      per_round: list of dicts with keys: round, num_experts, num_finals, success_rate
      totals: dict with keys: total_experts, total_finals, overall_success_rate
    """
    per_round = []
    total_experts = 0
    total_finals = 0

    rounds = obj.get("rounds", [])
    for rnd in rounds:
        round_idx = rnd.get("round")
        experts = rnd.get("experts", {})
        num_experts = len(experts)

        num_finals = 0
        for exp_payload in experts.values():
            # Happy-path extraction of text content
            if isinstance(exp_payload, str):
                content = exp_payload
            elif isinstance(exp_payload, dict):
                resp = exp_payload.get("response")
                if isinstance(resp, str):
                    content = resp
                elif isinstance(resp, dict):
                    content = resp.get("content")
                else:
                    content = None
            else:
                content = None

            if isinstance(content, str) and PATTERN.search(content):
                num_finals += 1

        total_experts += num_experts
        total_finals += num_finals
        success_rate = (num_finals / num_experts) if num_experts > 0 else 0.0
        per_round.append(
            {
                "round": round_idx,
                "num_experts": num_experts,
                "num_finals": num_finals,
                "success_rate": success_rate,
            }
        )

    overall_success_rate = (total_finals / total_experts) if total_experts > 0 else 0.0
    totals = {
        "total_experts": total_experts,
        "total_finals": total_finals,
        "overall_success_rate": overall_success_rate,
    }
    return per_round, totals


def main():
    parser = argparse.ArgumentParser(
        description="Count expert responses ending with FINAL PROBABILITY: <float> in a JSON file."
    )
    parser.add_argument("jsonpath", type=Path, help="Path to input JSON file")
    args = parser.parse_args()

    with args.jsonpath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    experts_per_round = 0
    # Support a single object or a list of objects in the file
    if isinstance(data, dict):
        per_round, totals = compute_stats(data)
        experts_counts = [r["num_experts"] for r in per_round]
        unique_counts = sorted(set(experts_counts))
        if len(unique_counts) != 1:
            # VERY LOUD failure: show per-round counts and abort
            details = ", ".join(
                f"round {r['round']}: {r['num_experts']}" for r in per_round
            )
            raise RuntimeError(
                "INCONSISTENT EXPERT COUNTS ACROSS ROUNDS! "
                f"Found counts={unique_counts}. Per-round details: {details}"
            )
        experts_per_round = unique_counts[0]

    elif isinstance(data, list):
        # Aggregate across a list of objects
        all_rounds = []
        agg_experts = agg_finals = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            pr, t = compute_stats(item)
            # --- Consistency check: experts count must be identical across rounds ---
            experts_counts = [r["num_experts"] for r in pr]
            unique_counts = sorted(set(experts_counts))
            if len(unique_counts) != 1:
                # VERY LOUD failure: show per-round counts and abort
                details = ", ".join(
                    f"round {r['round']}: {r['num_experts']}" for r in pr
                )
                raise RuntimeError(
                    "INCONSISTENT EXPERT COUNTS ACROSS ROUNDS! "
                    f"Found counts={unique_counts}. Per-round details: {details}"
                )
            experts_per_round = unique_counts[0]

            all_rounds.extend(pr)
            agg_experts += t["total_experts"]
            agg_finals += t["total_finals"]
        # Collapse per-round reporting by index if present; otherwise just print sequentially
        per_round = all_rounds
        totals = {
            "total_experts": agg_experts,
            "total_finals": agg_finals,
            "overall_success_rate": (agg_finals / agg_experts)
            if agg_experts > 0
            else 0.0,
        }
    else:
        raise ValueError("Top-level JSON must be an object or a list of objects.")

    # Prints for downstream aggregation
    print(f"file: {args.jsonpath}")
    print(f"experts_per_round: {experts_per_round}")
    for r in per_round:
        rlabel = r["round"] if r["round"] is not None else "unknown"
        print(
            f"round {rlabel}: finals={r['num_finals']}/{r['num_experts']} success_rate={r['success_rate']:.4f}"
        )
    print(f"finals_total: {totals['total_finals']}")
    print(f"overall_success_rate: {totals['overall_success_rate']:.4f}")


if __name__ == "__main__":
    main()
