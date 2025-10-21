#!/usr/bin/env python3
"""
Calculate Brier scores from Delphi results using actual resolution data.
"""

import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
from dataset.dataloader import ForecastDataLoader


def calculate_brier_scores_for_directory(
    results_dir: Path, resolution_date: str = "2025-07-21"
):
    """Calculate Brier scores for all JSON files in a results directory."""

    # Initialize data loader
    loader = ForecastDataLoader()

    # Find all JSON files
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return None

    print(f"Found {len(json_files)} JSON files in {results_dir}")
    print("=" * 80)

    all_results = []
    round_stats = defaultdict(list)  # Statistics by round

    resolved_count = 0
    unresolved_count = 0

    for json_path in sorted(json_files):
        print(f"\nProcessing: {json_path.name}")

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            question_id = data.get("question_id", "unknown")
            question_text = data.get("question_text", "No question text")[:60] + "..."

            print(f"Question ID: {question_id}")
            print(f"Question: {question_text}")

            # Get resolution
            resolution = loader.get_resolution(question_id, resolution_date)

            if not resolution:
                print(f"  ❌ No resolution found for question {question_id}")
                unresolved_count += 1
                continue

            if not resolution.resolved:
                print(f"  ❌ Question {question_id} not resolved")
                unresolved_count += 1
                continue

            actual_outcome = resolution.resolved_to
            print(f"  ✅ Actual outcome: {actual_outcome}")
            resolved_count += 1

            if not data.get("rounds"):
                print("  ❌ No rounds found")
                continue

            # Process each round
            question_results = {
                "question_id": question_id,
                "question_text": question_text,
                "actual_outcome": actual_outcome,
                "rounds": {},
            }

            for round_data in data["rounds"]:
                round_num = round_data["round"]
                experts = round_data["experts"]

                if not experts:
                    continue

                # Calculate predictions and Brier scores for this round
                predictions = [expert["prob"] for expert in experts.values()]
                brier_scores = [(pred - actual_outcome) ** 2 for pred in predictions]

                median_pred = np.median(predictions)
                mean_pred = np.mean(predictions)

                median_brier = (median_pred - actual_outcome) ** 2
                mean_brier = np.mean(brier_scores)

                round_results = {
                    "median_prediction": median_pred,
                    "mean_prediction": mean_pred,
                    "median_brier": median_brier,
                    "mean_brier": mean_brier,
                    "individual_briers": brier_scores,
                    "n_experts": len(experts),
                }

                question_results["rounds"][round_num] = round_results
                round_stats[round_num].append(
                    {
                        "median_brier": median_brier,
                        "mean_brier": mean_brier,
                        "question_id": question_id,
                    }
                )

                print(
                    f"  Round {round_num}: Median pred={median_pred:.3f}, Median Brier={median_brier:.3f}, Mean Brier={mean_brier:.3f}"
                )

            all_results.append(question_results)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    print("\n" + "=" * 80)
    print("BRIER SCORE ANALYSIS")
    print("=" * 80)

    if not all_results:
        print("No results with resolutions found!")
        return None

    print(f"Successfully processed: {resolved_count} resolved questions")
    print(f"Unresolved questions: {unresolved_count}")
    print(f"Total questions: {resolved_count + unresolved_count}")

    # Calculate statistics by round
    print("\nBrier Score Statistics by Round:")
    print("-" * 50)

    for round_num in sorted(round_stats.keys()):
        round_data = round_stats[round_num]
        median_briers = [r["median_brier"] for r in round_data]
        mean_briers = [r["mean_brier"] for r in round_data]

        overall_median_brier = np.median(median_briers)
        overall_mean_brier = np.mean(
            median_briers
        )  # This is what you want: AVERAGE of median Briers
        std_median_brier = np.std(median_briers)

        print(f"Round {round_num}: n={len(round_data)}")
        print(
            f"  Average of median LLM Briers: {overall_mean_brier:.4f} ± {std_median_brier:.4f}"
        )
        print(f"  Median of median Briers: {overall_median_brier:.4f}")
        print(f"  Range: [{min(median_briers):.4f}, {max(median_briers):.4f}]")

    # Show improvement across rounds (if multiple rounds)
    if len(round_stats) > 1:
        print("\nImprovement Analysis:")
        print("-" * 30)

        round_nums = sorted(round_stats.keys())
        initial_round = round_nums[0]
        final_round = round_nums[-1]

        initial_briers = [r["median_brier"] for r in round_stats[initial_round]]
        final_briers = [r["median_brier"] for r in round_stats[final_round]]

        # Match questions across rounds
        improvements = []
        for init_result in round_stats[initial_round]:
            qid = init_result["question_id"]
            # Find matching final result
            final_result = next(
                (r for r in round_stats[final_round] if r["question_id"] == qid), None
            )
            if final_result:
                improvement = init_result["median_brier"] - final_result["median_brier"]
                improvements.append(improvement)

        if improvements:
            mean_improvement = np.mean(improvements)
            improved_count = sum(1 for imp in improvements if imp > 0)
            print(
                f"Questions that improved: {improved_count}/{len(improvements)} ({100 * improved_count / len(improvements):.1f}%)"
            )
            print(f"Mean improvement: {mean_improvement:.4f}")
            print(f"Median improvement: {np.median(improvements):.4f}")

    # Best and worst performing questions
    print("\nBest Performing Questions (lowest final Brier):")
    print("-" * 50)

    final_round_num = max(round_stats.keys())
    final_results = sorted(
        round_stats[final_round_num], key=lambda x: x["median_brier"]
    )

    for i, result in enumerate(final_results[:5]):
        qid = result["question_id"]
        question_data = next(q for q in all_results if q["question_id"] == qid)
        print(f"{i + 1}. {question_data['question_text'][:60]}...")
        print(f"   Brier: {result['median_brier']:.4f}")

    print("\nWorst Performing Questions (highest final Brier):")
    print("-" * 50)

    for i, result in enumerate(final_results[-5:]):
        qid = result["question_id"]
        question_data = next(q for q in all_results if q["question_id"] == qid)
        print(f"{i + 1}. {question_data['question_text'][:60]}...")
        print(f"   Brier: {result['median_brier']:.4f}")

    # Return summary statistics
    final_briers = [r["median_brier"] for r in round_stats[final_round_num]]
    return {
        "total_questions": resolved_count,
        "unresolved_questions": unresolved_count,
        "final_round_median_brier": np.median(final_briers),
        "final_round_average_median_brier": np.mean(
            final_briers
        ),  # Average of median LLM Briers
        "final_round_std_brier": np.std(final_briers),
        "round_stats": dict(round_stats),
        "all_results": all_results,
    }


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python calculate_brier_scores.py <results_directory> [resolution_date]"
        )
        print(
            "Example: python calculate_brier_scores.py results/experts_comparison_gpt_oss_20b 2025-07-21"
        )
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    resolution_date = sys.argv[2] if len(sys.argv) > 2 else "2025-07-21"

    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    print(f"Calculating Brier scores for: {results_dir}")
    print(f"Resolution date: {resolution_date}")
    print("=" * 80)

    results = calculate_brier_scores_for_directory(results_dir, resolution_date)

    if results:
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   Questions analyzed: {results['total_questions']}")
        print(
            f"   Average of median LLM Briers (final round): {results['final_round_average_median_brier']:.4f} ± {results['final_round_std_brier']:.4f}"
        )
        print(
            f"   Median of median Briers (final round): {results['final_round_median_brier']:.4f}"
        )
    else:
        print("\n❌ No valid results found")


if __name__ == "__main__":
    main()
