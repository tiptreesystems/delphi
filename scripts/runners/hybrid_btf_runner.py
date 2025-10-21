import argparse
import asyncio
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from agents.hybrid_btf_expert import HybridBTFExpert
from agents.btf_utils import load_questions_from_csv
from utils.models import LLMFactory, LLMModel

import debugpy

print("Waiting for debugger attach...")
debugpy.listen(5679)
debugpy.wait_for_client()
print("Debugger attached.")


def _build_default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results/hybrid_btf_runs") / f"hybrid_btf_run_{timestamp}.jsonl"


def _resolve_model(name: Optional[str]) -> Optional[Union[LLMModel, str]]:
    if not name:
        return None
    lowered = name.strip().lower()
    for candidate in LLMModel:
        if candidate.value.lower() == lowered:
            return candidate
    return name


def _build_llm(args: argparse.Namespace):
    model = _resolve_model(args.model)
    provider = (args.provider or "openai").strip().lower()
    if provider not in {"openai", "claude", "anthropic", "groq"}:
        raise SystemExit(
            f"Unsupported provider '{args.provider}'. Choose from openai, claude/anthropic, or groq."
        )
    if provider == "anthropic":
        provider = "claude"
    return LLMFactory.create_llm(
        provider,
        model=model,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Hybrid BTF expert over CSV questions."
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
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="LLM provider to use (openai, claude, groq).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model identifier to use.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional custom system prompt for the LLM.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key override for the chosen provider.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the forecasting call.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=600,
        help="Max completion tokens for the forecasting call.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=3,
        help="Maximum number of search queries to request per question.",
    )
    parser.add_argument(
        "--query-temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for search query generation.",
    )
    parser.add_argument(
        "--query-max-tokens",
        type=int,
        default=512,
        help="Max completion tokens for search query generation.",
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

    llm = _build_llm(args)
    config = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    retrieval_config = {
        "max_queries": args.max_queries,
        "query_temperature": args.query_temperature,
        "query_max_tokens": args.query_max_tokens,
    }

    expert = HybridBTFExpert(
        llm,
        config=config,
        retrieval_config=retrieval_config,
        questions_csv=args.csv,
    )

    results = []
    total = len(questions)

    for idx, question in enumerate(questions, start=1):
        expert.conversation_manager.messages.clear()
        forecast = await expert.forecast_with_details(question)
        results.append(forecast)
        if args.verbose:
            print(
                f"[{idx}/{total}] {question.id}: "
                f"p_yes={forecast.probability:.3f} "
                f"queries={len(forecast.queries)} "
                f"evidence={len(forecast.evidence)}"
            )

    output_target = args.output or _build_default_output()
    # If a file path was provided (e.g. .json or .jsonl) use a directory named after that file's stem,
    # otherwise treat the provided path as the directory to write into.
    if output_target.suffix.lower() in {".json", ".jsonl"}:
        output_dir = output_target.parent / output_target.stem
    else:
        output_dir = output_target
    output_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize(name: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))

    for i, forecast in enumerate(results, start=1):
        data = forecast.as_dict()
        qid = data.get("question_id") or data.get("id") or f"forecast_{i}"
        filename = f"{_sanitize(qid)}.json"
        file_path = output_dir / filename
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=None)

    print(f"Wrote {len(results)} forecasts to {output_dir}")

    brier_values = [
        r.brier_score
        for r in results
        if isinstance(r.brier_score, float) and not math.isnan(r.brier_score)
    ]
    if brier_values:
        avg_brier = sum(brier_values) / len(brier_values)
        print(f"Mean Brier score (resolved questions): {avg_brier:.4f}")

    fallback_count = sum(
        1 for r in results if r.metadata.get("fallback_to_btf_pipeline")
    )
    if fallback_count:
        print(f"Fallback to baseline forecast used on {fallback_count} question(s).")


def main() -> None:
    args = parse_args()

    if args.sample and not args.limit:
        raise SystemExit("--sample requires --limit to be set")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
