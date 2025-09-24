#!/usr/bin/env python3
"""
Launcher script for genetic prompt evolution experiments.

This script provides an easy way to run genetic evolution experiments
and includes utilities for comparing results.
"""

import asyncio
import argparse
from pathlib import Path
from utils.utils import load_experiment_config, setup_environment
from dotenv import load_dotenv

load_dotenv()


async def run_evolution_experiment(config_path: str):
    """Run a single evolution experiment."""
    from genetic_evolution.genetic_prompt_evolution import GeneticEvolutionPipeline

    config = load_experiment_config(config_path)
    setup_environment(config)
    print(f"\n{'=' * 80}")
    print(f"Running evolution experiment: {config_path}")
    print(f"{'=' * 80}")

    pipeline = GeneticEvolutionPipeline(config_path)
    await pipeline.run_evolution()

    print(f"\n✅ Experiment completed: {config_path}")
    return True


def compare_results(results_dirs: list):
    """Compare results from multiple evolution runs."""
    import json

    print(f"\n{'=' * 80}")
    print("RESULTS COMPARISON")
    print(f"{'=' * 80}")

    results = []

    for results_dir in results_dirs:
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"❌ Results directory not found: {results_dir}")
            continue

        # Find results JSON file
        json_files = list(results_path.glob("genetic_evolution_results_*.json"))
        if not json_files:
            print(f"❌ No results file found in: {results_dir}")
            continue

        # Load most recent results file
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)

        try:
            with open(latest_file, "r") as f:
                result_data = json.load(f)

            config_name = result_data["config"]["experiment"]["name"]
            best_fitness = result_data["evolution_results"]["best_fitness"]
            generations = result_data["evolution_results"]["total_generations"]
            final_validation = result_data["evolution_results"].get(
                "final_validation", {}
            )

            results.append(
                {
                    "name": config_name,
                    "fitness": best_fitness,
                    "generations": generations,
                    "mae": final_validation.get("mean_absolute_error", "N/A"),
                    "brier": final_validation.get("brier_score", "N/A"),
                    "questions": final_validation.get("questions_evaluated", "N/A"),
                }
            )

        except Exception as e:
            print(f"❌ Error loading results from {latest_file}: {e}")
            continue

    if not results:
        print("No results to compare!")
        return

    # Sort by fitness (descending)
    results.sort(key=lambda x: x["fitness"], reverse=True)

    print(
        f"\n{'Experiment':<25} {'Fitness':<8} {'Gens':<5} {'MAE':<8} {'Brier':<8} {'Questions':<10}"
    )
    print("-" * 75)

    for result in results:
        mae_str = (
            f"{result['mae']:.3f}"
            if isinstance(result["mae"], float)
            else str(result["mae"])
        )
        brier_str = (
            f"{result['brier']:.3f}"
            if isinstance(result["brier"], float)
            else str(result["brier"])
        )

        print(
            f"{result['name']:<25} {result['fitness']:<8.3f} {result['generations']:<5} "
            f"{mae_str:<8} {brier_str:<8} {result['questions']:<10}"
        )

    print(f"\nBest performing experiment: {results[0]['name']}")
    print(f"  Fitness: {results[0]['fitness']:.3f}")
    print(f"  Generations: {results[0]['generations']}")
    if isinstance(results[0]["mae"], float):
        print(f"  Mean Absolute Error: {results[0]['mae']:.3f}")
    if isinstance(results[0]["brier"], float):
        print(f"  Brier Score: {results[0]['brier']:.3f}")


async def main():
    """Main entry point with command-line interface."""

    parser = argparse.ArgumentParser(description="Genetic Prompt Evolution")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="DIR",
        help="Compare results from multiple directories instead of running evolution",
    )

    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare)
    elif args.config:
        await run_evolution_experiment(args.config)
    else:
        parser.error("Either --config or --compare is required")


if __name__ == "__main__":
    asyncio.run(main())
