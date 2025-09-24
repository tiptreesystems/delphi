#!/usr/bin/env python3
"""
Analyze reasoning effort sweep results to determine optimal settings.

This script computes aggregate statistics to answer:
1. Which expert reasoning effort level (low/medium/high) performs best?
2. Which mediator reasoning effort level (low/medium/high) performs best?
"""

import json
import sys
import subprocess
import re
import argparse
from pathlib import Path
from collections import defaultdict
import statistics


def run_icl_delphi_results(config_path):
    """Run icl_delphi_results.py and capture the output."""
    try:
        result = subprocess.run(
            ["python", "icl_delphi_results.py", config_path],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout
    except Exception as e:
        print(
            f"Error running icl_delphi_results.py with {config_path}: {e}",
            file=sys.stderr,
        )
        return None


def parse_final_brier_score(output_text):
    """Parse the final round Brier score from icl_delphi_results.py output."""
    if not output_text:
        return None

    # Look for the highest round number LLM result
    round_matches = re.findall(
        r"\[LLM\].*?median LLM forecast.*?at round (\d+): ([\d.]+|nan)", output_text
    )

    if not round_matches:
        return None

    # Get the final (highest) round
    final_round = max(int(r[0]) for r in round_matches)

    for round_num, brier in round_matches:
        if int(round_num) == final_round and brier != "nan":
            return float(brier)

    return None


def load_and_analyze_sweep(sweep_dir):
    """Load sweep results and analyze Brier scores."""
    sweep_dir = Path(sweep_dir)

    # Load results summary
    results_summary_path = sweep_dir / "results_summary.json"
    if not results_summary_path.exists():
        print(f"Error: No results_summary.json found in {sweep_dir}")
        return None

    with open(results_summary_path, "r") as f:
        results = json.load(f)

    # Process each successful experiment
    experiment_data = []

    for result in results:
        if result["success"]:
            config_path = result["config_file"]
            value_parts = result["value"].split(" + ")

            if len(value_parts) == 2:
                expert_reasoning = value_parts[0]
                mediator_reasoning = value_parts[1]

                print(
                    f"Processing {expert_reasoning} expert + {mediator_reasoning} mediator..."
                )

                # Get Brier score
                output = run_icl_delphi_results(config_path)
                final_brier = parse_final_brier_score(output)

                if final_brier is not None:
                    experiment_data.append(
                        {
                            "expert_reasoning": expert_reasoning,
                            "mediator_reasoning": mediator_reasoning,
                            "final_brier": final_brier,
                            "config_file": config_path,
                        }
                    )
                    print(f"  Final Brier score: {final_brier:.4f}")
                else:
                    print(f"  Could not extract Brier score")

    return experiment_data


def compute_aggregate_statistics(experiment_data):
    """Compute aggregate statistics for expert and mediator reasoning levels."""

    # Group by expert reasoning level
    expert_groups = defaultdict(list)
    for exp in experiment_data:
        expert_groups[exp["expert_reasoning"]].append(exp["final_brier"])

    # Group by mediator reasoning level
    mediator_groups = defaultdict(list)
    for exp in experiment_data:
        mediator_groups[exp["mediator_reasoning"]].append(exp["final_brier"])

    print(f"\n{'=' * 80}")
    print(f"REASONING EFFORT AGGREGATE ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Total experiments analyzed: {len(experiment_data)}")

    # Expert reasoning analysis
    print(f"\n📊 EXPERT REASONING EFFORT ANALYSIS")
    print(f"-" * 50)
    print(
        f"{'Level':<10} {'Count':<8} {'Mean':<10} {'Median':<10} {'StdDev':<10} {'Min':<10} {'Max':<10}"
    )
    print(f"-" * 50)

    expert_stats = {}
    for level in ["low", "medium", "high"]:
        if level in expert_groups:
            values = expert_groups[level]
            stats = {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
            }
            expert_stats[level] = stats

            print(
                f"{level:<10} {stats['count']:<8} {stats['mean']:<10.4f} {stats['median']:<10.4f} "
                f"{stats['stdev']:<10.4f} {stats['min']:<10.4f} {stats['max']:<10.4f}"
            )

    # Mediator reasoning analysis
    print(f"\n📊 MEDIATOR REASONING EFFORT ANALYSIS")
    print(f"-" * 50)
    print(
        f"{'Level':<10} {'Count':<8} {'Mean':<10} {'Median':<10} {'StdDev':<10} {'Min':<10} {'Max':<10}"
    )
    print(f"-" * 50)

    mediator_stats = {}
    for level in ["low", "medium", "high"]:
        if level in mediator_groups:
            values = mediator_groups[level]
            stats = {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
            }
            mediator_stats[level] = stats

            print(
                f"{level:<10} {stats['count']:<8} {stats['mean']:<10.4f} {stats['median']:<10.4f} "
                f"{stats['stdev']:<10.4f} {stats['min']:<10.4f} {stats['max']:<10.4f}"
            )

    # Determine best settings
    print(f"\n🏆 BEST SETTINGS (Lower Brier = Better)")
    print(f"-" * 50)

    # Best expert reasoning level (by mean)
    if expert_stats:
        best_expert_mean = min(expert_stats.items(), key=lambda x: x[1]["mean"])
        best_expert_median = min(expert_stats.items(), key=lambda x: x[1]["median"])
        print(
            f"🎯 Best Expert Reasoning (by mean):   {best_expert_mean[0]} ({best_expert_mean[1]['mean']:.4f})"
        )
        print(
            f"🎯 Best Expert Reasoning (by median): {best_expert_median[0]} ({best_expert_median[1]['median']:.4f})"
        )

    # Best mediator reasoning level (by mean)
    if mediator_stats:
        best_mediator_mean = min(mediator_stats.items(), key=lambda x: x[1]["mean"])
        best_mediator_median = min(mediator_stats.items(), key=lambda x: x[1]["median"])
        print(
            f"🎯 Best Mediator Reasoning (by mean):   {best_mediator_mean[0]} ({best_mediator_mean[1]['mean']:.4f})"
        )
        print(
            f"🎯 Best Mediator Reasoning (by median): {best_mediator_median[0]} ({best_mediator_median[1]['median']:.4f})"
        )

    # Individual experiment ranking
    print(f"\n📋 INDIVIDUAL EXPERIMENT RANKING")
    print(f"-" * 50)
    print(f"{'Rank':<6} {'Expert':<8} {'Mediator':<10} {'Brier Score':<12}")
    print(f"-" * 50)

    # Sort by Brier score (lower is better)
    sorted_experiments = sorted(experiment_data, key=lambda x: x["final_brier"])

    for i, exp in enumerate(sorted_experiments, 1):
        print(
            f"{i:<6} {exp['expert_reasoning']:<8} {exp['mediator_reasoning']:<10} {exp['final_brier']:<12.4f}"
        )

    # Statistical significance tests (basic comparison)
    print(f"\n📈 PERFORMANCE DIFFERENCES")
    print(f"-" * 50)

    # Expert reasoning differences
    expert_levels = ["low", "medium", "high"]
    if len(expert_stats) > 1:
        print("Expert Reasoning Level Differences (mean):")
        for i, level1 in enumerate(expert_levels):
            if level1 in expert_stats:
                for level2 in expert_levels[i + 1 :]:
                    if level2 in expert_stats:
                        diff = (
                            expert_stats[level1]["mean"] - expert_stats[level2]["mean"]
                        )
                        direction = "better" if diff < 0 else "worse"
                        print(
                            f"  {level1} vs {level2}: {abs(diff):.4f} ({level1} is {direction})"
                        )

    # Mediator reasoning differences
    if len(mediator_stats) > 1:
        print("\nMediator Reasoning Level Differences (mean):")
        for i, level1 in enumerate(expert_levels):
            if level1 in mediator_stats:
                for level2 in expert_levels[i + 1 :]:
                    if level2 in mediator_stats:
                        diff = (
                            mediator_stats[level1]["mean"]
                            - mediator_stats[level2]["mean"]
                        )
                        direction = "better" if diff < 0 else "worse"
                        print(
                            f"  {level1} vs {level2}: {abs(diff):.4f} ({level1} is {direction})"
                        )

    print(f"\n{'=' * 80}")

    return expert_stats, mediator_stats, sorted_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Analyze reasoning effort sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_reasoning_effort_sweep.py results/20250818_223127
        """,
    )

    parser.add_argument("sweep_dir", help="Path to sweep results directory")

    args = parser.parse_args()

    # Load and analyze sweep results
    print(f"Loading sweep results from {args.sweep_dir}...")
    experiment_data = load_and_analyze_sweep(args.sweep_dir)

    if not experiment_data:
        print("Failed to load experiment data")
        sys.exit(1)

    if len(experiment_data) == 0:
        print("No valid experiment data found")
        sys.exit(1)

    # Compute statistics
    expert_stats, mediator_stats, sorted_experiments = compute_aggregate_statistics(
        experiment_data
    )


if __name__ == "__main__":
    main()
