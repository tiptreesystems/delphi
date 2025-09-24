#!/usr/bin/env python3
"""
Plot mean±std curves from multiple JSON result files (one line per file)
and overlay superforecaster & public baselines.

JSON schema expected (per file):
{
  "rounds": [0, 1, 2, 3],
  "mean_curve": [...],
  "std_curve": [...],
  "sf_baseline": {"mean": 0.1356, "std": 0.0},
  "public_baseline": {"mean": 0.1653, "std": 0.0},
  "per_seed": {...}  # unused here
}

USAGE
-----
python plot_brier_progress.py \
  --paths path/to/runA.json path/to/runB.json ... \
  --labels "1 Agent" "3 Agents" ... \
  --title "Average Brier Score" \
  --out /path/to/figure.png

Notes
-----
- Baseline lines are taken from the FIRST file by default. If you prefer the
  average of baselines across files, add: --baseline-mode average
- X ticks/labels are inferred from 'rounds' and formatted like the provided figure.
- Y label and legend text follow the example figure.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


def load_result(path: Path):
    with open(path, "r") as f:
        d = json.load(f)
    # minimal validation
    for key in ["rounds", "mean_curve", "std_curve", "sf_baseline", "public_baseline"]:
        if key not in d:
            raise ValueError(f"{path} is missing key: {key}")
    return d


def format_round_labels(rounds: List[int]) -> List[str]:
    # 0 -> "initial forecasts", 1.. -> "update i"
    labels = []
    for r in rounds:
        if r == 0:
            labels.append("initial forecasts")
        else:
            labels.append(f"update {r}")
    return labels


def compute_baselines(results: List[dict], mode: str) -> Tuple[float, float]:
    """Return (public_mean, sf_mean) according to mode."""
    if mode == "first":
        public = results[0]["public_baseline"]["mean"]
        sf = results[0]["sf_baseline"]["mean"]
        return public, sf
    elif mode == "average":
        publics = [r["public_baseline"]["mean"] for r in results]
        sfs = [r["sf_baseline"]["mean"] for r in results]
        return float(np.mean(publics)), float(np.mean(sfs))
    else:
        raise ValueError("baseline-mode must be 'first' or 'average'")


def main():
    p = argparse.ArgumentParser(
        description="Plot Brier mean±std vs. round for multiple runs."
    )
    # p.add_argument("--paths", nargs="+", required=True, help="JSON files to plot (one line per file).")
    # p.add_argument("--labels", nargs="+", required=True, help="Legend labels (same length as --paths).")
    p.add_argument(
        "--out",
        type=str,
        default="/home/williaar/projects/delphi/results/compare_across_configs/brier_progress.pdf",
        help="Output image path (PDF).",
    )
    p.add_argument(
        "--title", type=str, default="", help="Optional plot title (small, above axes)."
    )
    p.add_argument(
        "--baseline-mode",
        choices=["first", "average"],
        default="first",
        help="Use baselines from the first file (default) or the average across files.",
    )
    p.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    args = p.parse_args()

    # paths = [Path(s) for s in args.paths]
    # labels = args.labels

    # paths = [
    #     '/home/williaar/projects/delphi/results/evolution_evaluation/gpt_oss_120b_expert_system_mediator_evolved_1_experts_3_examples_no_system_prompt/set_eval/aggregate_brier_rounds_median_all_eval.json', # 1 agent
    #     '/home/williaar/projects/delphi/results/evolution_evaluation/gpt_oss_120b_expert_system_mediator_evolved_2_experts_3_examples_no_system_prompt/set_eval/aggregate_brier_rounds_median_all_eval.json', # 2 agents
    #     '/home/williaar/projects/delphi/results/evolution_evaluation/gpt_oss_120b_expert_system_mediator_evolved_3_experts_3_examples_no_system_prompt/set_eval/aggregate_brier_rounds_median_all_eval.json', # 3 agents
    #     '/home/williaar/projects/delphi/results/evolution_evaluation/gpt_oss_120b_expert_system_mediator_evolved_5_experts_3_examples_no_system_prompt/set_eval/aggregate_brier_rounds_median_all_eval.json', # 5 agents
    # ]

    # labels = [
    #     '1 Agent', '2 Agents', '3 Agents', '5 Agents'
    # ]

    paths = [
        "/home/williaar/projects/delphi/results/evolution_evaluation/gpt_oss_120b_expert_system_mediator_evolved_3_experts_3_examples_no_system_prompt/set_eval/aggregate_brier_rounds_median_all_eval.json",  # oss-120b
        "/home/williaar/projects/delphi/results/evolution_evaluation/gpt_oss_20b_expert_system_mediator_evolved_3_experts_3_examples_no_system_prompt/set_eval/aggregate_brier_rounds_median_all_eval.json",  # oss-20b
        "/home/williaar/projects/delphi/results/evolution_evaluation/gpt_oss_20b_expert_system_mediator_evolved_3_experts_no_examples_no_system_prompt/set_eval/aggregate_brier_rounds_median_all_eval.json",  # oss-20b no examples
        "/home/williaar/projects/delphi/results/evolution_evaluation/o3_expert_system_mediator_evolved_3_experts_3_examples_no_system_prompt/set_eval/aggregate_brier_rounds_median_all_eval.json",  # o3
        "/home/williaar/projects/delphi/results/evolution_evaluation/o3_expert_system_mediator_evolved_3_experts_no_examples_no_system_prompt/set_eval/aggregate_brier_rounds_median_all_eval.json",  # o3 no examples
    ]

    labels = [
        "gpt-oss-120b",
        "gpt-oss-20b",
        "gpt-oss-20b-no-examples",
        "o3",
        "o3-no-examples",
    ]

    if len(paths) != len(labels):
        raise SystemExit("ERROR: --paths and --labels must have the same length.")

    results = [load_result(pth) for pth in paths]

    # Ensure consistent rounds
    rounds0 = results[0]["rounds"]
    for i, r in enumerate(results[1:], start=1):
        if r["rounds"] != rounds0:
            raise SystemExit(
                f"ERROR: rounds differ in file index {i}: {r['rounds']} vs {rounds0}"
            )
    x = np.array(rounds0, dtype=float)
    xticklabels = format_round_labels(rounds0)

    # Baselines
    public_mean, sf_mean = compute_baselines(results, args.baseline_mode)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(15, 7))
    # Y-axis label matches the example figure
    ax.set_ylabel("Average Brier Score\n(mean ± std, ↓ lower is better)", fontsize=18)
    ax.set_xlabel("Round", fontsize=18)
    ax.set_xticks(x, xticklabels, rotation=0, fontsize=18)
    ax.tick_params(axis="y", labelsize=12)

    # use the "viridis" colormap for the plot color cycle
    n_colors = len(paths)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_colors))[::-1]
    ax.set_prop_cycle(color=colors)

    # assign distinct markers per line (no cycler import)
    markers = ["o", "s", "^", "D", "v", "p", "X", "*", "h", "+"]
    marker_list = [markers[i % len(markers)] for i in range(n_colors)]

    # ensure color cycle remains
    ax.set_prop_cycle(color=colors)

    # wrap ax.errorbar so we can inject a different marker per call without changing the loop below
    _marker_iter = iter(marker_list)
    _orig_errorbar = ax.errorbar

    def _errorbar_with_markers(*args, **kwargs):
        # remove any fmt provided (e.g. "o-") and apply our own marker while preserving line style
        fmt = kwargs.pop("fmt", None)
        marker = next(_marker_iter)
        if fmt is not None and "-" in fmt:
            kwargs.setdefault("linestyle", "-")
        kwargs.setdefault("marker", marker)
        return _orig_errorbar(*args, **kwargs)

    ax.errorbar = _errorbar_with_markers

    # Plot each run: mean ± std as error bars + line
    for res, label in zip(results, labels):
        y = np.array(res["mean_curve"], dtype=float)
        yerr = np.array(res["std_curve"], dtype=float)
        # errorbar returns a container; use default line/marker cycling
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            capsize=3,
            linewidth=2,
            elinewidth=1.2,
            label=label,
        )

    # Baseline horizontal lines (match example legend wording)
    ax.axhline(
        public_mean,
        linestyle=(0, (5, 5)),
        linewidth=2,
        color="#ff7f0e",
        label="Median of Public",
    )
    ax.axhline(
        sf_mean,
        linestyle=(0, (3, 6)),
        linewidth=2,
        color="#2ca02c",
        label="Median of Superforecasters",
    )

    # Legend
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=18,
        ncol=1,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.6,
        borderpad=0.3,
    )

    # Optional small title
    if args.title:
        ax.set_title(args.title, fontsize=18, pad=8)

    # Tight layout and save
    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
