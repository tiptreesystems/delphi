#!/usr/bin/env python3
"""
Topic-based analysis of sweep results using DataFrames.

This script analyzes sweep results grouped by question topics, creating separate plots for each topic.
It uses pandas DataFrames with flat column structure for all statistics.
"""

import json
import sys
import subprocess
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from dataset.dataloader import ForecastDataLoader


def load_question_topics(sweep_dir):
    """Load question topics using ForecastDataLoader for questions in sweep."""
    loader = ForecastDataLoader()
    question_to_topic = {}

    # Get question IDs from the sweep output directories
    sweep_dir = Path(sweep_dir)
    question_ids = set()

    output_dirs = [
        d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith("results_")
    ]
    for output_dir in output_dirs:
        json_files = list(output_dir.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    qid = data.get("question_id", "")
                    if qid:
                        question_ids.add(qid)
            except:
                continue

    # Load topics for each question ID found
    for qid in question_ids:
        try:
            question = loader.get_question(qid)
            if question and hasattr(question, "topic"):
                question_to_topic[qid] = question.topic
            else:
                question_to_topic[qid] = "Unknown"
        except:
            question_to_topic[qid] = "Unknown"

    return question_to_topic


def extract_params_from_config_path(config_path):
    """Extract parameters from config file path."""
    params = {}

    # Extract from filename like config_n_experts_3_seed_42.yml or config_n_experts_3.yml
    filename = Path(config_path).stem

    # Parse n_experts
    n_experts_match = re.search(r"n_experts_(\d+)", filename)
    if n_experts_match:
        params["n_experts"] = int(n_experts_match.group(1))

    # Parse seed (optional)
    seed_match = re.search(r"seed_(\d+)", filename)
    if seed_match:
        params["seed"] = int(seed_match.group(1))
    else:
        # Default seed if not specified
        params["seed"] = 42

    return params


def load_results_as_dataframe(sweep_dir):
    """Load sweep results and organize them into a pandas DataFrame."""
    sweep_dir = Path(sweep_dir)

    # Load question topics
    print("Loading question topics...")
    question_to_topic = load_question_topics(sweep_dir)

    # Initialize ForecastDataLoader for getting resolutions
    loader = ForecastDataLoader()

    # Find all config files
    config_files = list(sweep_dir.glob("config_*.yml"))
    print(f"Found {len(config_files)} configuration files")

    # Process each config and collect results
    all_results = []

    for config_path in config_files:
        print(f"Processing {config_path.name}...")

        # Extract parameters from config filename
        params = extract_params_from_config_path(config_path)

        if not params:
            print(f"  Could not parse parameters from {config_path.name}, skipping...")
            continue

        # Check if corresponding output directory exists
        # Try different naming patterns
        if params["seed"] != 42:
            output_dir_name = (
                f"results_n_experts_{params['n_experts']}_seed_{params['seed']}"
            )
        else:
            # Try simple format first
            output_dir_name = f"results_n_experts_{params['n_experts']}"
            output_dir = sweep_dir / output_dir_name
            if not output_dir.exists():
                # Fallback to explicit seed format
                output_dir_name = (
                    f"results_n_experts_{params['n_experts']}_seed_{params['seed']}"
                )

        output_dir = sweep_dir / output_dir_name

        if not output_dir.exists():
            print(f"  Output directory {output_dir_name} not found, skipping...")
            continue

        # Process each JSON file and calculate Brier scores
        json_files = list(output_dir.glob("*.json"))
        if not json_files:
            print(f"  No JSON files in {output_dir_name}, skipping...")
            continue

        # Get resolution date from config (assuming it's 2025-07-21 based on filenames)
        resolution_date = "2025-07-21"

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                qid = data.get("question_id", "")
                if not qid or qid not in question_to_topic:
                    continue

                # Get resolution for this question
                try:
                    resolution = loader.get_resolution(qid, resolution_date)
                    if resolution is None or not hasattr(resolution, "resolved_to"):
                        continue
                    resolution_value = resolution.resolved_to
                except:
                    continue

                # Calculate Brier scores for each round
                row = {
                    "question_id": qid,
                    "topic": question_to_topic[qid],
                    "n_experts": params["n_experts"],
                    "seed": params["seed"],
                }

                # Process rounds data
                if "rounds" in data and isinstance(data["rounds"], list):
                    round_briers = {}

                    for round_idx, round_data in enumerate(data["rounds"]):
                        if "experts" in round_data and isinstance(
                            round_data["experts"], dict
                        ):
                            # Calculate aggregate forecast for this round
                            forecasts = []
                            for expert_id, expert_data in round_data["experts"].items():
                                if (
                                    "prob" in expert_data
                                    and expert_data["prob"] is not None
                                ):
                                    forecasts.append(expert_data["prob"])

                            if forecasts:
                                # Calculate mean forecast and Brier score
                                mean_forecast = sum(forecasts) / len(forecasts)
                                brier = (mean_forecast - resolution_value) ** 2
                                round_briers[round_idx] = brier

                    # Add round scores as separate columns
                    for round_num in range(5):
                        col_name = f"round_{round_num}_brier"
                        row[col_name] = round_briers.get(round_num, np.nan)

                    # Add final round score
                    if round_briers:
                        final_round = max(round_briers.keys())
                        row["final_brier"] = round_briers[final_round]
                        row["final_round_num"] = final_round
                    else:
                        row["final_brier"] = np.nan
                        row["final_round_num"] = np.nan

                    all_results.append(row)

            except Exception as e:
                continue

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Ensure consistent column order
    column_order = ["question_id", "topic", "n_experts", "seed"]
    for i in range(5):
        column_order.append(f"round_{i}_brier")
    column_order.extend(["final_brier", "final_round_num"])

    # Reorder columns (only include columns that exist)
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]

    return df


def plot_topic_progression_from_df(df, topic, output_path):
    """Create round progression plot for a specific topic from DataFrame."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Filter data for this topic
    topic_df = df[df["topic"] == topic].copy()

    if topic_df.empty:
        ax.text(
            0.5,
            0.5,
            f"No data available for topic: {topic}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(f"Topic: {topic} (No Data)", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # Get unique n_experts values
    n_experts_values = sorted(topic_df["n_experts"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_experts_values)))

    # Plot for each n_experts value
    for idx, n_experts in enumerate(n_experts_values):
        # Filter for this n_experts value
        expert_df = topic_df[topic_df["n_experts"] == n_experts]

        # Collect round data
        round_cols = [
            col
            for col in expert_df.columns
            if col.startswith("round_") and col.endswith("_brier")
        ]
        rounds = []
        mean_briers = []
        std_errors = []

        for col in sorted(round_cols):
            round_num = int(col.split("_")[1])
            values = expert_df[col].dropna()

            if len(values) > 0:
                rounds.append(round_num)
                mean_briers.append(values.mean())
                if len(values) > 1:
                    std_errors.append(values.std() / np.sqrt(len(values)))
                else:
                    std_errors.append(0)

        if rounds:
            ax.errorbar(
                rounds,
                mean_briers,
                yerr=std_errors,
                fmt="o-",
                label=f"n_experts={n_experts}",
                color=colors[idx],
                linewidth=2,
                markersize=8,
                alpha=0.8,
                capsize=3,
                capthick=1.5,
                elinewidth=1.5,
            )

    ax.set_xlabel("Delphi Round", fontsize=14)
    ax.set_ylabel("Brier Score (lower is better)", fontsize=14)
    ax.set_title(
        f"Topic: {topic} - Brier Score Progression", fontsize=16, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_topic_final_scores_from_df(df, topic, output_path):
    """Create final score comparison plot for a specific topic from DataFrame."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Filter data for this topic
    topic_df = df[df["topic"] == topic].copy()

    if topic_df.empty or "final_brier" not in topic_df.columns:
        ax.text(
            0.5,
            0.5,
            f"No data available for topic: {topic}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(
            f"Topic: {topic} - Final Scores (No Data)", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # Group by n_experts and calculate statistics
    stats = (
        topic_df.groupby("n_experts")["final_brier"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Calculate standard error
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["se"] = stats["se"].fillna(0)

    # Plot
    ax.errorbar(
        stats["n_experts"],
        stats["mean"],
        yerr=stats["se"],
        fmt="o-",
        linewidth=2.5,
        markersize=10,
        capsize=5,
        capthick=2,
        elinewidth=2,
        label="Mean ± SE",
        color="blue",
        alpha=0.8,
    )

    # Add sample size annotations
    for _, row in stats.iterrows():
        ax.annotate(
            f"n={int(row['count'])}",
            (row["n_experts"], row["mean"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel("Number of Experts", fontsize=14)
    ax.set_ylabel("Final Brier Score (lower is better)", fontsize=14)
    ax.set_title(
        f"Topic: {topic} - Final Score by Number of Experts",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_topic_summary_from_df(df, topic):
    """Create a summary for a specific topic from DataFrame."""
    topic_df = df[df["topic"] == topic]

    print(f"\n{topic}:")
    print(f"  Total data points: {len(topic_df)}")
    print(f"  Unique questions: {topic_df['question_id'].nunique()}")
    print(f"  n_experts values: {sorted(topic_df['n_experts'].unique())}")
    print(f"  Seeds: {sorted(topic_df['seed'].unique())}")

    if "final_brier" in topic_df.columns:
        # Calculate average final scores by n_experts
        stats = topic_df.groupby("n_experts")["final_brier"].agg(["mean", "count"])
        print("  Average final Brier scores:")
        for n_experts, row in stats.iterrows():
            print(
                f"    n_experts={n_experts}: {row['mean']:.3f} (n={int(row['count'])})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sweep results by question topic using DataFrames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_sweep_by_topic_df.py results/20250821_072517
  python analyze_sweep_by_topic_df.py results/20250821_072517 --output topic_plots_df
  python analyze_sweep_by_topic_df.py results/20250821_072517 --save-df results.csv
  
This script creates separate plots for each question topic and uses pandas DataFrames.
        """,
    )

    parser.add_argument("sweep_dir", help="Path to sweep results directory")

    parser.add_argument(
        "--output",
        default="topic_plots_df",
        help="Output directory for topic plots (default: topic_plots_df)",
    )

    parser.add_argument("--save-df", help="Save the DataFrame to a CSV file")

    parser.add_argument(
        "--no-plots", action="store_true", help="Skip creating plots, only process data"
    )

    args = parser.parse_args()

    # Load sweep results into DataFrame
    print(f"Loading sweep results from {args.sweep_dir}...")
    df = load_results_as_dataframe(args.sweep_dir)

    if df.empty:
        print("No results found!")
        return

    print(f"\nLoaded {len(df)} rows of data")
    print(f"Topics found: {sorted(df['topic'].unique())}")
    print(f"Questions: {df['question_id'].nunique()}")
    print(f"Configurations: {df[['n_experts', 'seed']].drop_duplicates().shape[0]}")

    # Save DataFrame if requested
    if args.save_df:
        df.to_csv(args.save_df, index=False)
        print(f"\nDataFrame saved to {args.save_df}")

        # Print DataFrame info
        print("\nDataFrame structure:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        print("\nDataFrame shape:", df.shape)

    if args.no_plots:
        print("\nSkipping plot creation (--no-plots specified)")
        return

    # Create output directory
    sweep_path = Path(args.sweep_dir)
    output_dir = sweep_path / args.output
    output_dir.mkdir(exist_ok=True)

    print(f"\nCreating plots in {output_dir}")
    print("=" * 80)
    print("TOPIC SUMMARIES")
    print("=" * 80)

    # Process each topic
    for topic in sorted(df["topic"].unique()):
        # Create summary
        create_topic_summary_from_df(df, topic)

        # Create safe filename from topic name
        safe_topic_name = re.sub(r"[^\w\s-]", "", topic)
        safe_topic_name = re.sub(r"[-\s]+", "_", safe_topic_name)

        # Create progression plot
        progression_path = output_dir / f"{safe_topic_name}_progression.png"
        print(f"  Creating progression plot: {progression_path.name}")
        plot_topic_progression_from_df(df, topic, progression_path)

        # Create final scores plot
        final_path = output_dir / f"{safe_topic_name}_final_scores.png"
        print(f"  Creating final scores plot: {final_path.name}")
        plot_topic_final_scores_from_df(df, topic, final_path)

    print("\n" + "=" * 80)
    print(f"Created plots for {df['topic'].nunique()} topics in {output_dir}")

    # Create summary statistics CSV
    summary_stats = (
        df.groupby(["topic", "n_experts"])
        .agg(
            {
                "final_brier": ["mean", "std", "count"],
                "round_0_brier": "mean",
                "round_1_brier": "mean",
                "round_2_brier": "mean",
                "round_3_brier": "mean",
            }
        )
        .round(3)
    )

    summary_path = output_dir / "summary_statistics.csv"
    summary_stats.to_csv(summary_path)
    print(f"Summary statistics saved to {summary_path}")


if __name__ == "__main__":
    main()
