#!/usr/bin/env python3
"""
Quick script to calculate Brier statistics from a results directory.
"""

import json
import sys
from pathlib import Path
import numpy as np


def calculate_brier_from_json(json_path):
    """Calculate Brier score from a single JSON file."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract final round predictions
        if not data.get("rounds"):
            return None

        # Get final round (highest round number)
        final_round = max(data["rounds"], key=lambda x: x["round"])

        brier_scores = []
        for expert_id, expert_data in final_round["experts"].items():
            pred_prob = expert_data["prob"]

            # Look for resolution in question data or infer from question_id
            question_id = data["question_id"]

            # For now, we'll need actual resolution data
            # This is a placeholder - you'd need to add resolution loading
            print(f"Question {question_id}: Predicted prob = {pred_prob:.3f}")

        return None  # Would return brier_scores if we had resolutions

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python get_brier_stats.py <results_directory>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    # Find all JSON files
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files in {results_dir}")
    print("=" * 60)

    # Process each file
    all_predictions = []

    for json_path in sorted(json_files):
        print(f"\nProcessing: {json_path.name}")

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            question_id = data.get("question_id", "unknown")
            question_text = data.get("question_text", "No question text")[:80] + "..."

            print(f"Question ID: {question_id}")
            print(f"Question: {question_text}")

            if not data.get("rounds"):
                print("  No rounds found")
                continue

            # Show progression through rounds
            for round_data in data["rounds"]:
                round_num = round_data["round"]
                experts = round_data["experts"]

                if round_num == 0:
                    print(f"  Round {round_num} (Initial): {len(experts)} experts")
                else:
                    print(f"  Round {round_num}: {len(experts)} experts")

                # Calculate median probability for this round
                probs = [expert["prob"] for expert in experts.values()]
                median_prob = np.median(probs)
                mean_prob = np.mean(probs)
                std_prob = np.std(probs)

                print(
                    f"    Median prob: {median_prob:.3f}, Mean: {mean_prob:.3f} ± {std_prob:.3f}"
                )

                # Store final round data
                if round_num == max([r["round"] for r in data["rounds"]]):
                    all_predictions.append(
                        {
                            "question_id": question_id,
                            "median_prob": median_prob,
                            "mean_prob": mean_prob,
                            "std_prob": std_prob,
                            "n_experts": len(experts),
                        }
                    )

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if all_predictions:
        final_medians = [p["median_prob"] for p in all_predictions]
        final_means = [p["mean_prob"] for p in all_predictions]

        print(f"Total questions processed: {len(all_predictions)}")
        print(f"Overall median of medians: {np.median(final_medians):.3f}")
        print(
            f"Overall mean of medians: {np.mean(final_medians):.3f} ± {np.std(final_medians):.3f}"
        )
        print(
            f"Overall mean of means: {np.mean(final_means):.3f} ± {np.std(final_means):.3f}"
        )
        print(f"Range: [{min(final_medians):.3f}, {max(final_medians):.3f}]")

        # Show distribution
        print(f"\nProbability distribution:")
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(final_medians, bins=bins)
        for i in range(len(bins) - 1):
            print(f"  {bins[i]:.1f}-{bins[i + 1]:.1f}: {hist[i]:2d} questions")
    else:
        print("No predictions found")

    print("\nNote: To calculate Brier scores, you need actual resolutions.")
    print("The script would need to load resolution data to compute Brier scores.")


if __name__ == "__main__":
    main()
