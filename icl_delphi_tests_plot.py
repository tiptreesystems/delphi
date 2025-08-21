# plot_delphi_distribution.py
# Usage:
#   python plot_delphi_distribution.py --log /path/to/delphi_log_...json --out outputs/delphi_distribution.png
#
# Produces a violin + scatter plot showing the distribution of expert probabilities per round.

import argparse
import json
import os
import re
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


import os
import textwrap

output_dir = "outputs_initial_delphi_flexible_retry_test_set_forecast_evolutions"


def parse_delphi_log_filename(filename: str):
    """
    Given a filename like:
        delphi_log_with_examples_<unique_id>_<YYYY-MM-DD>.json
        delphi_log_no_examples_<unique_id>_<YYYY-MM-DD>.json
    return (unique_id, date_str)
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)

    if name.startswith("delphi_log_with_examples_"):
        prefix = "delphi_log_with_examples_"
    elif name.startswith("delphi_log_no_examples_"):
        prefix = "delphi_log_no_examples_"
    else:
        raise ValueError(f"Unexpected filename format: {filename}")

    parts = name[len(prefix):].split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse unique_id and date from: {filename}")

    unique_id = "_".join(parts[:-1])  # join in case ID has underscores
    date_str = parts[-1]

    return unique_id, date_str


# Optional fallback in case some rounds only stored raw text responses
_PROB_PAT = re.compile(r'FINAL PROBABILITY:\s*(0?\.\d+|1\.0|0|1)', re.IGNORECASE)

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

def _collect_round_probs(delphi_log: Dict[str, Any]) -> Dict[int, List[float]]:
    """
    Returns {round_idx: [probabilities]} using stored numeric probs when available,
    falling back to parsing the response text if needed.
    """
    out: Dict[int, List[float]] = {}
    rounds = delphi_log.get("rounds", [])
    for r in rounds:
        r_idx = int(r.get("round", 0))
        expert_dict = r.get("experts", {})
        probs = []
        for sfid, entry in expert_dict.items():
            # Prefer stored numeric prob
            p = entry.get("prob")
            if isinstance(p, (int, float)):
                p = float(p)
            else:
                # Fallback to parse from response text
                p = _extract_prob(entry.get("response"))
            if p is not None:
                probs.append(max(0.0, min(1.0, p)))
        out[r_idx] = probs
    return dict(sorted(out.items(), key=lambda kv: kv[0]))

def plot_distribution_by_round(
    round_probs: Dict[int, List[float]],
    real_superforecaster_forecasts: List[float],
    title: str = "Delphi: Distribution of Forecasts by Round",
    save_path: Optional[str] = None,
    show_points: bool = True,
    resolution = None
) -> None:
    """
    Makes a violin plot across rounds, with optional jittered points and per-round medians.
    Adds real superforecaster forecasts as dots on the right edge of the plot.
    """
    if not round_probs:
        raise ValueError("No round probabilities found to plot.")

    rounds = list(round_probs.keys())
    data = [round_probs[r] for r in rounds]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Violin plot of distributions
    parts = ax.violinplot(
        data, positions=rounds,
        showmeans=False, showextrema=False, showmedians=False
    )

    # Style violins lightly
    for pc in parts['bodies']:
        pc.set_alpha(0.4)

    # Overlay per-round medians and IQR
    medians = [np.median(d) if len(d) else np.nan for d in data]
    q1 = [np.percentile(d, 25) if len(d) else np.nan for d in data]
    q3 = [np.percentile(d, 75) if len(d) else np.nan for d in data]

    ax.plot(rounds, medians, marker='o', linewidth=1.5, label='Median')
    ax.vlines(rounds, q1, q3, linewidth=2, label='IQR')

    # Optional jittered scatter of individual expert probs
    if show_points:
        rng = np.random.default_rng(42)
        for x, d in zip(rounds, data):
            if not d:
                continue
            jitter = (rng.random(len(d)) - 0.5) * 0.15
            ax.scatter(np.full(len(d), x) + jitter, d, s=18, alpha=0.7)

    # --- Add real superforecaster forecasts as dots (to the right of the violins) ---
    if real_superforecaster_forecasts:
        x_pos = max(rounds) + 1.2  # place them just to the right of last round
        jitter = (np.random.default_rng(123).random(len(real_superforecaster_forecasts)) - 0.5) * 0.15
        ax.scatter(
            np.full(len(real_superforecaster_forecasts), x_pos) + jitter,
            real_superforecaster_forecasts,
            s=30, c="black", marker="x", label="Superforecasters"
        )
        # Extend x ticks and add label
        ax.set_xticks(rounds + [x_pos])
        ax.set_xticklabels([str(r) for r in rounds] + ["Superforecasters"])

    # Wrap the title if it's too long
    wrapped_title = "\n".join(textwrap.wrap(title, width=120))
    ax.set_title(wrapped_title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)

    if resolution:
        if resolution.resolved_to == 1:
            plt.suptitle("Resolved: YES ↑", color='green', fontsize=14, fontweight='bold')
        elif resolution.resolved_to == 0:
            plt.suptitle("Resolved: NO ↓", color='red', fontsize=14, fontweight='bold')

    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='best')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()
def main():
    parser = argparse.ArgumentParser(description="Plot evolution of Delphi forecast distributions across rounds.")
    parser.add_argument("--logdir", required=True, help="Directory containing delphi_log_...json files.")
    parser.add_argument("--output-dir", default=output_dir, help="Directory to save output plots.")
    parser.add_argument("--title", default=None, help="Custom plot title.")
    parser.add_argument("--no-points", action="store_true", help="Disable showing individual expert points.")
    args = parser.parse_args()


    resolution_date = '2025-07-21'

    logdir = args.logdir
    json_files = [f for f in os.listdir(logdir) if f.startswith("delphi_log_") and f.endswith(".json")]
    if not json_files:
        print(f"No delphi_log_*.json files found in {logdir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    loader = ForecastDataLoader()

    for json_file in json_files:
        json_path = os.path.join(logdir, json_file)
        with open(json_path, "r") as f:
            delphi_log = json.load(f)

        json_basename = os.path.splitext(os.path.basename(json_file))[0]
        out_path = os.path.join(args.output_dir, f"{json_basename}.png")

        round_probs = _collect_round_probs(delphi_log)

        sf_ids = delphi_log['rounds'][0]['experts'].keys()
        qid = delphi_log.get('question', None)

        real_superforecaster_forecast_objects = [
            loader.get_super_forecasts(question_id=qid, user_id=sf_id, resolution_date=resolution_date)[0]
            for sf_id in sf_ids
        ]

        real_superforecaster_forecasts = [
            obj.forecast for obj in real_superforecaster_forecast_objects
        ]

        try:
            question_id, resolution_date = parse_delphi_log_filename(json_file)
            resolution = loader.get_resolution(question_id=question_id, resolution_date=resolution_date)
        except Exception as e:
            print(f"Skipping {json_file}: {e}")
            continue

        question = loader.get_question(question_id=qid)
        title = args.title or f"{question.question} \n\n topic: {question.topic}"

        plot_distribution_by_round(
            round_probs,
            real_superforecaster_forecasts=real_superforecaster_forecasts,
            title=title,
            save_path=out_path,
            show_points=not args.no_points,
            resolution=resolution
        )

        print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
