#!/usr/bin/env python3
"""
Analyze results from sequential learning experiments.

This script loads saved results from sequential learning runs and generates
visualizations and statistics about bias evolution and learning patterns.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import pandas as pd


def load_results(file_path: str) -> Dict:
    """Load results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def analyze_bias_evolution(results: Dict) -> pd.DataFrame:
    """Analyze how bias evolves over time."""
    data = []

    for i, result in enumerate(results["results"]):
        # Expert biases
        for j, expert_result in enumerate(result["expert_results"]):
            if expert_result.get("bias") is not None:
                data.append(
                    {
                        "question_idx": i,
                        "model": f"expert_{j}",
                        "predicted": expert_result["predicted_prob"],
                        "actual": expert_result.get("resolution"),
                        "bias": expert_result["bias"],
                        "sf_median": expert_result.get("sf_median"),
                    }
                )

        # Mediator bias
        if (
            result.get("mediator_result")
            and result["mediator_result"].get("bias") is not None
        ):
            data.append(
                {
                    "question_idx": i,
                    "model": "mediator",
                    "predicted": result["mediator_result"]["synthesized_prob"],
                    "actual": result["mediator_result"].get("resolution"),
                    "bias": result["mediator_result"]["bias"],
                    "sf_median": result["mediator_result"].get("sf_median"),
                }
            )

    return pd.DataFrame(data)


def plot_bias_evolution(df: pd.DataFrame, output_dir: Path):
    """Plot how bias changes over sequential predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Bias over time for each model
    ax = axes[0, 0]
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        ax.plot(
            model_df["question_idx"],
            model_df["bias"],
            label=model,
            marker="o",
            alpha=0.7,
        )
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Question Index")
    ax.set_ylabel("Bias (Predicted - Actual)")
    ax.set_title("Bias Evolution Over Sequential Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Rolling mean bias
    ax = axes[0, 1]
    window = 3
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        if len(model_df) >= window:
            rolling_bias = model_df["bias"].rolling(window=window, center=True).mean()
            ax.plot(
                model_df["question_idx"],
                rolling_bias,
                label=f"{model} (rolling mean)",
                linewidth=2,
            )
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Question Index")
    ax.set_ylabel("Rolling Mean Bias")
    ax.set_title(f"Smoothed Bias Trend (window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Absolute error over time
    ax = axes[1, 0]
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        abs_error = model_df["bias"].abs()
        ax.plot(model_df["question_idx"], abs_error, label=model, marker="s", alpha=0.7)
    ax.set_xlabel("Question Index")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Absolute Prediction Error Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Calibration plot
    ax = axes[1, 1]
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        if not model_df.empty:
            ax.scatter(
                model_df["predicted"], model_df["actual"], label=model, alpha=0.6
            )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Outcome")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = output_dir / "bias_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved bias evolution plot to: {output_path}")
    plt.close()


def print_summary_statistics(results: Dict, df: pd.DataFrame):
    """Print summary statistics from the sequential learning."""
    print("\n" + "=" * 60)
    print("SEQUENTIAL LEARNING ANALYSIS")
    print("=" * 60)

    # Overall statistics
    print(f"\nTotal questions processed: {len(results['results'])}")
    print(f"Questions with resolutions: {len(df[df['bias'].notna()])}")

    # Model-specific statistics
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        resolved_df = model_df[model_df["bias"].notna()]

        if not resolved_df.empty:
            print(f"\n{model.upper()} Statistics:")
            print(f"  Mean bias: {resolved_df['bias'].mean():.3f}")
            print(f"  Std bias: {resolved_df['bias'].std():.3f}")
            print(f"  Mean absolute error: {resolved_df['bias'].abs().mean():.3f}")
            print(
                f"  Min/Max bias: {resolved_df['bias'].min():.3f} / {resolved_df['bias'].max():.3f}"
            )

            # Check for improvement over time
            if len(resolved_df) > 1:
                first_half = (
                    resolved_df.iloc[: len(resolved_df) // 2]["bias"].abs().mean()
                )
                second_half = (
                    resolved_df.iloc[len(resolved_df) // 2 :]["bias"].abs().mean()
                )
                improvement = (first_half - second_half) / first_half * 100
                print(f"  Improvement (first vs second half): {improvement:.1f}%")

    # Memory statistics
    if "expert_memory" in results:
        expert_memory = results["expert_memory"]
        print(f"\nExpert Memory Statistics:")
        print(f"  Entries stored: {len(expert_memory['entries'])}")
        if expert_memory["bias_stats"]["mean_bias"] != 0:
            print(
                f"  Learned mean bias: {expert_memory['bias_stats']['mean_bias']:.3f}"
            )
            print(
                f"  Overconfidence count: {expert_memory['bias_stats']['overconfidence_count']}"
            )
            print(
                f"  Underconfidence count: {expert_memory['bias_stats']['underconfidence_count']}"
            )

    if "mediator_memory" in results and any(
        r.get("mediator_result") for r in results["results"]
    ):
        mediator_memory = results["mediator_memory"]
        print(f"\nMediator Memory Statistics:")
        print(f"  Entries stored: {len(mediator_memory['entries'])}")
        if mediator_memory["bias_stats"]["mean_bias"] != 0:
            print(
                f"  Learned mean bias: {mediator_memory['bias_stats']['mean_bias']:.3f}"
            )

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze sequential learning results")
    parser.add_argument("results_file", type=str, help="Path to the results JSON file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis",
        help="Directory to save analysis outputs",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze bias evolution
    df = analyze_bias_evolution(results)

    # Print summary statistics
    print_summary_statistics(results, df)

    # Generate plots
    if not args.no_plots and not df.empty:
        plot_bias_evolution(df, output_dir)

        # Save processed data
        csv_path = output_dir / "bias_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved bias data to: {csv_path}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
