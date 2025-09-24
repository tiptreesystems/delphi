#!/usr/bin/env python
"""
Script to run analysis on all expert comparison folders and plot with variance-based error bars.
"""

import json
import sys
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from dataset.dataloader import ForecastDataLoader


def extract_model_name(folder_name):
    """Extract a clean model name from the folder name."""
    name = folder_name.replace("results_experts_comparison_", "")
    name = name.replace("results_prompt_comparison_", "")

    name_map = {
        "claude37_sonnet": "Claude 3.7 Sonnet",
        "deepseek_r1": "DeepSeek R1",
        "gpt_oss_120b": "GPT OSS 120B",
        "gpt_oss_20b": "GPT OSS 20B",
        "gpt5": "GPT-5",
        "llama_maverick": "Llama Maverick",
        "o1": "O1",
        "o3": "O3",
        "qwen3_32b": "Qwen3 32B",
        "baseline": "Baseline",
        "baseline_with_examples": "Baseline w/ Examples",
        "base_rate": "Base Rate",
        "deep_analytical": "Deep Analytical",
        "frequency_based": "Frequency Based",
        "high_variance": "High Variance",
        "opinionated": "Opinionated",
        "short_focused": "Short Focused",
    }

    return name_map.get(name, name.replace("_", " ").title())


def compute_brier_score(prob, outcome):
    """Compute Brier score for a binary outcome."""
    return (prob - outcome) ** 2


def get_expert_brier_scores_from_json(json_path):
    """Extract individual expert Brier scores from a JSON file for each round."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Get the resolution
        resolution = data.get("resolution", {})
        y_true = resolution.get("resolved_to", 0)  # Binary outcome

        # Extract expert probabilities by round
        round_briers = {}
        rounds = data.get("rounds", [])

        for round_data in rounds:
            round_num = round_data.get("round", 0)
            experts = round_data.get("experts", {})

            # Calculate Brier score for each expert
            expert_briers = []
            for expert_id, expert_data in experts.items():
                if "prob" in expert_data:
                    prob = expert_data["prob"]
                    brier = compute_brier_score(prob, y_true)
                    expert_briers.append(brier)

            if expert_briers:
                round_briers[round_num] = expert_briers

        return round_briers, y_true

    except Exception as e:
        print(f"Error processing {json_path}: {e}", file=sys.stderr)
        return {}, None


def get_sf_and_public_briers(question_id, resolution_date):
    """Get SF and public Brier scores for a question."""
    loader = ForecastDataLoader()

    # Get resolution
    resolution = loader.get_resolution(
        question_id=question_id, resolution_date=resolution_date
    )
    y_true = resolution.resolved_to

    # Get SF forecasts
    sf_forecasts = loader.get_super_forecasts(
        question_id=question_id, resolution_date=resolution_date
    )
    sf_briers = [compute_brier_score(sf.forecast, y_true) for sf in sf_forecasts]

    # Get public forecasts
    public_forecasts = loader.get_public_forecasts(
        question_id=question_id, resolution_date=resolution_date
    )
    public_briers = [
        compute_brier_score(pf.forecast, y_true) for pf in public_forecasts
    ]

    return sf_briers, public_briers


def analyze_output_directory(output_dir):
    """Analyze all JSON files in an output directory."""
    output_dir = Path(output_dir)
    json_files = sorted([f for f in output_dir.glob("*.json") if f.is_file()])

    # Aggregate Brier scores across questions for each round
    all_round_briers = defaultdict(
        list
    )  # round -> list of all expert briers across questions
    all_sf_briers = []
    all_public_briers = []

    for json_file in json_files:
        # Get expert Brier scores from JSON first
        round_briers, y_true = get_expert_brier_scores_from_json(json_file)

        # Add to aggregate
        for round_num, briers in round_briers.items():
            all_round_briers[round_num].extend(briers)

        # Try to get SF/public data from the JSON itself if possible
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            question_id = data.get("question_id", "")
            resolution_date = data.get("resolution_date", "2025-07-21")

            # Only try to get SF/public if we have a valid question_id
            if question_id and not question_id.startswith("experts_"):
                try:
                    sf_briers, public_briers = get_sf_and_public_briers(
                        question_id, resolution_date
                    )
                    all_sf_briers.extend(sf_briers)
                    all_public_briers.extend(public_briers)
                except Exception as e:
                    # Silently skip - SF/public data might not be available
                    pass
        except Exception as e:
            # Silently skip
            pass

    # Calculate statistics for each round
    round_stats = {}
    for round_num in sorted(all_round_briers.keys()):
        briers = all_round_briers[round_num]
        if briers:
            round_stats[round_num] = {
                "median": np.median(briers),
                "mean": np.mean(briers),
                "std": np.std(briers),
                "q25": np.percentile(briers, 25),
                "q75": np.percentile(briers, 75),
                "n": len(briers),
            }

    # Calculate SF and public stats
    sf_stats = {
        "median": np.median(all_sf_briers) if all_sf_briers else None,
        "mean": np.mean(all_sf_briers) if all_sf_briers else None,
        "std": np.std(all_sf_briers) if all_sf_briers else None,
    }

    public_stats = {
        "median": np.median(all_public_briers) if all_public_briers else None,
        "mean": np.mean(all_public_briers) if all_public_briers else None,
        "std": np.std(all_public_briers) if all_public_briers else None,
    }

    return round_stats, sf_stats, public_stats


def plot_results_with_variance(
    all_results,
    title="Brier Scores Across Delphi Rounds",
    include_models=None,
    exclude_models=["GPT-5"],
):
    """Create a plot with error bars based on actual variance."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Filter models
    if include_models:
        filtered_results = {k: v for k, v in all_results.items() if k in include_models}
    else:
        filtered_results = {
            k: v for k, v in all_results.items() if k not in exclude_models
        }

    # Color palette - distinct colors for the selected models
    color_map = {"O3": "blue", "Claude 3.7 Sonnet": "green", "GPT OSS 120B": "red"}
    colors = [
        color_map.get(name, plt.cm.tab20(i))
        for i, name in enumerate(filtered_results.keys())
    ]

    # Track SF and Public baselines
    all_sf_medians = []
    all_public_medians = []

    for idx, (model_name, (round_stats, sf_stats, public_stats)) in enumerate(
        filtered_results.items()
    ):
        if not round_stats:
            print(f"Skipping {model_name}: no round data", file=sys.stderr)
            continue

        # Get rounds and statistics
        rounds = sorted(round_stats.keys())
        medians = [round_stats[r]["median"] for r in rounds]

        # Use IQR for error bars (25th to 75th percentile)
        lower_errors = [
            round_stats[r]["median"] - round_stats[r]["q25"] for r in rounds
        ]
        upper_errors = [
            round_stats[r]["q75"] - round_stats[r]["median"] for r in rounds
        ]

        # Plot with asymmetric error bars
        ax.errorbar(
            rounds,
            medians,
            yerr=[lower_errors, upper_errors],
            fmt="o-",
            label=f"{model_name} (n={round_stats[rounds[0]]['n']})",
            color=colors[idx],
            linewidth=2,
            markersize=8,
            alpha=0.8,
            capsize=5,
            capthick=1.5,
        )

        # Collect SF and Public values
        if sf_stats["median"] is not None:
            all_sf_medians.append(sf_stats["median"])
        if public_stats["median"] is not None:
            all_public_medians.append(public_stats["median"])

    # Add horizontal lines for SF and Public baselines
    # Use known values from icl_delphi_results.py
    sf_median = 0.116  # From actual data
    public_median = 0.169  # From actual data

    ax.axhline(
        y=sf_median,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"SF Median: {sf_median:.3f}",
        alpha=0.7,
    )
    ax.axhline(
        y=public_median,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Public Median: {public_median:.3f}",
        alpha=0.7,
    )

    # Formatting
    ax.set_xlabel("Delphi Round", fontsize=12)
    ax.set_ylabel("Brier Score (lower is better)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)

    # Set x-axis to show integer rounds
    if rounds:
        ax.set_xticks(rounds)
        ax.set_xticklabels([str(r) for r in rounds])

    # Set y-axis limits
    ax.set_ylim(bottom=0, top=max(0.3, ax.get_ylim()[1]))

    plt.tight_layout()
    return fig


def main():
    # Find all output directories
    expert_dirs = sorted(
        [
            d
            for d in Path("..").glob("results_experts_comparison_*")
            if d.is_dir() and "_initial" not in d.name
        ]
    )
    prompt_dirs = sorted(
        [
            d
            for d in Path("..").glob("results_prompt_comparison_*")
            if d.is_dir() and "_initial" not in d.name
        ]
    )

    if not expert_dirs and not prompt_dirs:
        print("No output directories found")
        sys.exit(1)

    print(f"Found {len(expert_dirs)} expert comparison directories")
    print(f"Found {len(prompt_dirs)} prompt comparison directories")
    print("=" * 60)

    # Process expert comparisons
    expert_results = {}
    if expert_dirs:
        print("\nProcessing Expert Comparisons:")
        print("-" * 40)
        for output_dir in expert_dirs:
            print(f"Processing {output_dir.name}...")
            round_stats, sf_stats, public_stats = analyze_output_directory(output_dir)

            if round_stats:
                model_name = extract_model_name(output_dir.name)
                expert_results[model_name] = (round_stats, sf_stats, public_stats)

                # Print summary
                rounds_str = ", ".join(
                    [
                        f"R{r}: {stats['median']:.3f}±{stats['std']:.3f}"
                        for r, stats in sorted(round_stats.items())
                    ]
                )
                print(f"  Results: {rounds_str}")
            else:
                print(f"  No results found")
            print()

    # Process prompt comparisons
    prompt_results = {}
    if prompt_dirs:
        print("\nProcessing Prompt Comparisons:")
        print("-" * 40)
        for output_dir in prompt_dirs:
            print(f"Processing {output_dir.name}...")
            round_stats, sf_stats, public_stats = analyze_output_directory(output_dir)

            if round_stats:
                model_name = extract_model_name(output_dir.name)
                prompt_results[model_name] = (round_stats, sf_stats, public_stats)

                # Print summary
                rounds_str = ", ".join(
                    [
                        f"R{r}: {stats['median']:.3f}±{stats['std']:.3f}"
                        for r, stats in sorted(round_stats.items())
                    ]
                )
                print(f"  Results: {rounds_str}")
            else:
                print(f"  No results found")
            print()

    # Create plots
    print("=" * 60)

    # Plot expert comparisons - only selected models
    if expert_results:
        print("Creating expert comparison plot with variance...")
        selected_models = ["O3", "Claude 3.7 Sonnet", "GPT OSS 120B"]
        fig = plot_results_with_variance(
            expert_results,
            title="Model Comparison: O3 vs Claude 3.7 Sonnet vs GPT OSS 120B",
            include_models=selected_models,
        )
        output_file = "expert_comparison_selected.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Expert comparison plot saved to {output_file}")

        if "--no-show" not in sys.argv:
            plt.show()

    # Plot prompt comparisons
    if prompt_results:
        print("Creating prompt comparison plot with variance...")
        fig = plot_results_with_variance(
            prompt_results, title="Prompt Strategy Comparison: Brier Scores with IQR"
        )
        output_file = "prompt_comparison_variance.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Prompt comparison plot saved to {output_file}")

        if "--no-show" not in sys.argv:
            plt.show()


if __name__ == "__main__":
    main()
