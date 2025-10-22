import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from agents.btf_agent import BTFAgent
from agents.btf_utils import load_questions_from_csv
from utils.models import LLMFactory, LLMModel


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import debugpy

print("Waiting for debugger attach on port 5679...")
debugpy.listen(5679)
debugpy.wait_for_client()
print("Debugger attached, running tests...")


def _build_default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results/btf_agent_runs") / f"btf_agent_run_{timestamp}.jsonl"


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
        description="Run the BTF forecasting agent over CSV questions."
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
        default=5,
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
        "--not-verbose",
        action="store_true",
        help="Suppress per-question progress output.",
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
        "--max-tool-iterations",
        type=int,
        default=5,
        help="Maximum tool iterations allowed during a single forecast.",
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
    agent = BTFAgent(
        llm,
        probability_retry_count=2,
        max_tool_iterations=args.max_tool_iterations,
    )

    results = []
    total = len(questions)

    for idx, question in enumerate(questions, start=1):
        forecast = await agent.forecast(
            question,
            prior_forecast_info="",
            max_tool_iterations=args.max_tool_iterations,
        )
        results.append(forecast)

        if not args.not_verbose:
            print("=" * 80)
            print(
                f"[{idx}/{total}] {question.id}: "
                f"p_yes={forecast.probability:.3f} "
                f"search_results={len(forecast.search_results)} "
                f"facts={len(forecast.extracted_facts)}"
            )
            print("=" * 80)

    output_target = args.output or _build_default_output()
    if output_target.suffix.lower() in {".json", ".jsonl"}:
        output_dir = output_target.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_target
    else:
        output_target.mkdir(parents=True, exist_ok=True)
        dest = output_target / "forecasts.jsonl"

    with dest.open("w", encoding="utf-8") as stream:
        for forecast in results:
            stream.write(json.dumps(forecast.model_dump()) + "\n")

    print(f"Wrote {len(results)} forecasts to {dest}")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
