import argparse
import asyncio
import math
from datetime import datetime
from pathlib import Path

from agents.btf_expert import BTFExpert
from agents.btf_utils import load_questions_from_csv


def _build_default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results/btf_runs") / f"btf_run_{timestamp}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the BTF expert over CSV questions."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset/btf_data/questions.csv"),
        help="Path to CSV containing BTF-formatted questions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Process at most this many questions.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Randomly sample questions instead of taking the first N (requires --limit).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSONL file for forecasts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for each question.",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    questions = load_questions_from_csv(
        str(args.csv),
        limit=args.limit,
        sample=args.sample,
    )
    if not questions:
        print("No questions loaded; check --csv path or sampling settings.")
        return

    expert = BTFExpert()
    results = []

    total = len(questions)
    for idx, question in enumerate(questions, start=1):
        result = await expert.forecast_btf_question(question)
        results.append(result)
        if args.verbose:
            print(f"[{idx}/{total}] {question.id}: p_yes={result.probability:.3f}")

    output_path = args.output or _build_default_output()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(result.model_dump_json())
            f.write("\n")

    print(f"Wrote {len(results)} forecasts to {output_path}")

    brier_values = [
        r.brier_score
        for r in results
        if isinstance(r.brier_score, float) and not math.isnan(r.brier_score)
    ]
    if brier_values:
        avg_brier = sum(brier_values) / len(brier_values)
        print(f"Mean Brier score (resolved questions): {avg_brier:.4f}")


def main() -> None:
    args = parse_args()

    if args.sample and not args.limit:
        raise SystemExit("--sample requires --limit to be set")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
