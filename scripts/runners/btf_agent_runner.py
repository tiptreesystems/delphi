import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from agents.btf_agent import BTFAgent
from agents.btf_utils import BTFForecastResult, load_questions_from_csv
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
    return Path("results/btf_agent_runs") / f"btf_agent_run_{timestamp}"


def _sanitize_identifier(value: str, *, default: str = "question") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or default


def _extract_index_from_filename(name: str) -> int:
    try:
        prefix = name.split("_", 1)[0]
        return int(prefix)
    except (ValueError, IndexError):
        return 0


def _collect_design_choices(
    agent: BTFAgent, args: argparse.Namespace
) -> dict[str, object]:
    choices: dict[str, object] = {
        "agent_prompt_version": agent.prompt_version,
        "forecast_temperature": args.temperature,
        "forecast_max_tokens": args.max_tokens,
    }
    system_prompt = getattr(agent.conversation_manager.llm, "system_prompt", None)
    if system_prompt is not None:
        choices["system_prompt"] = system_prompt
    return choices


def _build_manifest_entry(
    forecast: BTFForecastResult, filename: str
) -> dict[str, Any]:
    return {
        "question_id": forecast.question_id,
        "question": forecast.question,
        "file": filename,
        "probability": forecast.probability,
        "search_result_count": len(forecast.search_results),
        "extracted_fact_count": len(forecast.extracted_facts),
        "fetched_url_content_count": forecast.fetched_url_content_count,
    }


def _write_manifest_file(
    output_dir: Path,
    entries_map: dict[str, dict[str, Any]],
    parameters: dict[str, Any],
    design_choices: dict[str, Any],
    source_csv: Path,
    created_at: str,
) -> Path:
    manifest_entries = sorted(
        entries_map.values(),
        key=lambda entry: (
            _extract_index_from_filename(entry.get("file", "")),
            entry.get("question_id", ""),
        ),
    )

    manifest = {
        "run_id": output_dir.name,
        "created_at": created_at,
        "question_count": len(manifest_entries),
        "source_csv": str(source_csv),
        "parameters": parameters,
        "design_choices": design_choices,
        "forecasts": manifest_entries,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest_path


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
        default=None,
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
        help="Destination directory (or basename) for per-question forecast JSON files.",
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
        default=10,
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

    output_target = args.output or _build_default_output()
    if output_target.suffix:
        logger.info(
            "Interpreting output path %s as directory %s",
            output_target,
            output_target.parent / output_target.stem,
        )
        output_dir = output_target.parent / output_target.stem
    else:
        output_dir = output_target
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load any existing manifest to enable cache reuse.
    manifest_path = output_dir / "manifest.json"
    existing_manifest: dict[str, object] = {}
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                existing_manifest = json.load(stream)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse existing manifest at {manifest_path}"
            ) from exc

    design_choices = _collect_design_choices(agent, args)
    existing_design = existing_manifest.get("design_choices", {})
    for key, value in design_choices.items():
        if key in existing_design and existing_design[key] != value:
            raise RuntimeError(
                f"Design choice '{key}' mismatch between existing manifest "
                f"({existing_design[key]!r}) and current run ({value!r}). "
                "Use a fresh output directory or delete the manifest."
            )
    merged_design_choices = dict(existing_design)
    merged_design_choices.update(design_choices)

    existing_records: dict[str, dict[str, object]] = {}
    used_filenames: set[str] = set()
    manifest_entries_map: dict[str, dict[str, Any]] = {}
    max_written_index = 0
    manifest_created_at = existing_manifest.get("created_at") if existing_manifest else datetime.utcnow().isoformat()

    for entry in existing_manifest.get("forecasts", []):
        qid = entry.get("question_id")
        file_name = entry.get("file")
        if not qid or not file_name:
            continue
        manifest_entries_map[qid] = dict(entry)
        file_path = output_dir / str(file_name)
        if not file_path.exists():
            logger.warning(
                "Forecast file '%s' listed in manifest but missing on disk; "
                "question %s will be recomputed.",
                file_name,
                qid,
            )
            continue
        try:
            with file_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            forecast_obj = BTFForecastResult.model_validate(payload)
        except Exception as exc:
            logger.warning(
                "Unable to load cached forecast %s (%s); recomputing.",
                file_name,
                exc,
            )
            continue
        if forecast_obj.question_id != qid:
            logger.warning(
                "Question ID mismatch for cached forecast %s (manifest=%s, file=%s); "
                "recomputing.",
                file_name,
                qid,
                forecast_obj.question_id,
            )
            continue
        existing_records[qid] = {
            "forecast": forecast_obj,
            "filename": str(file_name),
            "is_new": False,
        }
        used_filenames.add(str(file_name))
        max_written_index = max(
            max_written_index,
            _extract_index_from_filename(str(file_name)),
        )

    # Also load any per-question files that exist on disk but were not
    # referenced by the prior manifest (e.g., interrupted runs).
    for forecast_file in sorted(output_dir.glob("*.json")):
        if forecast_file.name == "manifest.json":
            continue
        if forecast_file.name in used_filenames:
            continue
        try:
            with forecast_file.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            forecast_obj = BTFForecastResult.model_validate(payload)
        except Exception as exc:
            logger.warning(
                "Skipping cached forecast %s due to load error: %s",
                forecast_file.name,
                exc,
            )
            continue
        existing_records.setdefault(
            forecast_obj.question_id,
            {
                "forecast": forecast_obj,
                "filename": forecast_file.name,
                "is_new": False,
            },
        )
        used_filenames.add(forecast_file.name)
        max_written_index = max(
            max_written_index,
            _extract_index_from_filename(forecast_file.name),
        )
        manifest_entries_map.setdefault(
            forecast_obj.question_id,
            _build_manifest_entry(forecast_obj, forecast_file.name),
        )

    parameters = dict(existing_manifest.get("parameters", {}))
    parameters.update(
        {
            "limit": args.limit,
            "sample": args.sample,
            "model": args.model,
            "provider": args.provider,
            "max_tool_iterations": args.max_tool_iterations,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    )

    results: list[dict[str, object]] = []
    total = len(questions)

    for idx, question in enumerate(questions, start=1):
        cached = existing_records.pop(question.id, None)
        if cached:
            results.append(cached)
            if not args.not_verbose:
                print("=" * 80)
                print(
                    f"[{idx}/{total}] {question.id}: cached forecast reused "
                    f"(p_yes={cached['forecast'].probability:.3f})"
                )
                print("=" * 80)

            manifest_entries_map[
                cached["forecast"].question_id
            ] = _build_manifest_entry(cached["forecast"], cached["filename"])
            used_filenames.add(cached["filename"])
            max_written_index = max(
                max_written_index,
                _extract_index_from_filename(cached["filename"]),
            )
            manifest_path = _write_manifest_file(
                output_dir,
                manifest_entries_map,
                parameters,
                merged_design_choices,
                Path(args.csv).resolve(),
                manifest_created_at,
            )
            continue

        forecast = await agent.forecast(
            question,
            prior_forecast_info="",
            max_tool_iterations=args.max_tool_iterations,
        )

        slug = _sanitize_identifier(forecast.question_id or f"q{idx:03d}")
        candidate_index = max(max_written_index + 1, idx)
        candidate = f"{candidate_index:03d}_{slug}.json"
        while candidate in used_filenames:
            candidate_index += 1
            candidate = f"{candidate_index:03d}_{slug}.json"
        used_filenames.add(candidate)
        max_written_index = max(max_written_index, candidate_index)

        file_path = output_dir / candidate
        with file_path.open("w", encoding="utf-8") as stream:
            json.dump(
                forecast.model_dump(),
                stream,
                ensure_ascii=False,
                indent=2,
            )

        results.append({"forecast": forecast, "filename": candidate, "is_new": False})

        if not args.not_verbose:
            print("=" * 80)
            print(
                f"[{idx}/{total}] {question.id}: "
                f"p_yes={forecast.probability:.3f} "
                f"search_results={len(forecast.search_results)} "
                f"url_content_fetched={forecast.fetched_url_content_count} "
                f"facts={len(forecast.extracted_facts)}"
            )
            print("=" * 80)

        manifest_entries_map[forecast.question_id] = _build_manifest_entry(
            forecast, candidate
        )
        manifest_path = _write_manifest_file(
            output_dir,
            manifest_entries_map,
            parameters,
            merged_design_choices,
            Path(args.csv).resolve(),
            manifest_created_at,
        )

    manifest_path = _write_manifest_file(
        output_dir,
        manifest_entries_map,
        parameters,
        merged_design_choices,
        Path(args.csv).resolve(),
        manifest_created_at,
    )

    print(
        f"Wrote {len(results)} forecasts to {output_dir} "
        f"(manifest: {manifest_path.name})"
    )


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
