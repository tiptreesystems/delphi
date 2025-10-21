import argparse
import asyncio
import copy
import json
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
from dotenv import load_dotenv

from analyze.utils import analyze_forecast_results
from dataset.dataloader import Forecast, ForecastDataLoader, Question
from agents.expert import Expert
from utils.llm_config import get_llm_from_config
from utils.sampling import sample_questions_by_topic
from utils.config_types import load_typed_experiment_config

from utils.convert_pickles_to_json import convert_pkl_to_json

from utils.config_types import RootConfig

# import debugpy
# if not debugpy.is_client_connected():
#     debugpy.listen(("localhost", 5679))
#     print("Waiting for debugger attach...")
#     debugpy.wait_for_client()


from utils.sampling import (
    # TRAIN_QUESTION_IDS,
    EVALUATION_QUESTION_IDS,
    EVOLUTION_EVALUATION_QUESTION_IDS,
)

load_dotenv()


class SubjectType(str, Enum):
    SUPERFORECASTER = "superforecaster"
    BASELINE = "baseline"  # no-examples, no SF id
    AGGREGATED = "aggregated"  # one expert using all SF examples
    SHARED = "shared"  # multiple experts share same examples


@dataclass
class TaskSpec:
    question: "Question"
    subject_type: SubjectType
    subject_id: Optional[str]  # sf_id for SUPERFORECASTER; None for BASELINE
    examples: Optional[List[Tuple["Question", "Forecast"]]]  # None for BASELINE


def build_specs_with_examples(
    sampled_questions,
    *,
    loader,
    date: str,
    min_examples: int = 1,
    max_examples: int = 5,
) -> List[TaskSpec]:
    """One TaskSpec per (question, sf_id) with that SF’s example pairs."""
    specs: List[TaskSpec] = []
    all_examples = _collect_example_forecasts(
        sampled_questions,
        loader,
        date,
        min_examples=min_examples,
        max_examples=max_examples,
    )
    for q in sampled_questions:
        for sf_id, pairs in all_examples.get(q.id, {}).items():
            if len(pairs) < min_examples:
                continue
            specs.append(
                TaskSpec(
                    question=q,
                    subject_type=SubjectType.SUPERFORECASTER,
                    subject_id=sf_id,
                    examples=pairs[:max_examples],
                )
            )
    return specs


def build_specs_aggregated(
    sampled_questions,
    *,
    loader,
    date: str,
    min_examples: int = 1,
    max_examples: Optional[int] = None,
    config: Optional[RootConfig] = None,
) -> List[TaskSpec]:
    """One TaskSpec per question aggregating examples across SFs.

    Parity mode (default true): mimic a K×M setup by
    - selecting K SFs using the same seed/ordering as the multi-expert run
    - taking up to M examples per selected SF
    - aggregating their examples (deduped by (question_id, user_id))

    If parity=false: aggregate across all SFs, optionally capped by max_examples.
    """

    specs: List[TaskSpec] = []

    if config is None:
        config = RootConfig()
    init_cfg = config.initial_forecasts or {}
    delphi_cfg = config.delphi or {}
    exp_cfg = config.experiment or {}

    parity = bool(config.initial_forecasts.get("parity", True))
    parity_source = (
        (init_cfg.get("parity_source") or "from_initial_forecasts")
        if parity
        else "none"
    )
    # How many experts to mirror (K) and examples per SF (M)
    parity_n_experts = int(init_cfg.get("parity_n_experts", 3))
    parity_per_sf_examples = int(
        init_cfg.get("parity_per_sf_examples", init_cfg.get("max_examples", 3))
    )
    # Seed to match expert selection
    parity_seed = int(delphi_cfg.get("expert_selection_seed", exp_cfg.get("seed", 42)))

    # Build all example pairs per SF for each question
    all_examples = _collect_example_forecasts(
        sampled_questions, loader, date, min_examples=1, max_examples=9999
    )

    for q in sampled_questions:
        sf_map = all_examples.get(q.id, {})  # {sf_id: [(Q,F), ...]}
        if not sf_map:
            continue

        if parity:
            agg_pairs: List[Tuple["Question", "Forecast"]] = []
            selected_sf_ids: List[str] = []

            if parity_source == "from_delphi_logs":
                # Read exact selected SF IDs from a paired Delphi run's logs
                delphi_out_dir = init_cfg.get(
                    "parity_delphi_output_dir"
                ) or exp_cfg.get("output_dir")
                seed_val = exp_cfg.get("seed", None)
                if seed_val is not None:
                    delphi_out_dir = os.path.join(
                        delphi_out_dir, f"seed_{int(seed_val)}"
                    )
                if delphi_out_dir and os.path.isdir(delphi_out_dir):
                    fname = f"delphi_eval_{q.id}_{date}.json"
                    fpath = os.path.join(delphi_out_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            log = json.load(f)
                        rounds = log.get("rounds") or []
                        if rounds:
                            experts_map = rounds[0].get("experts") or {}
                            selected_sf_ids = list(experts_map.keys())
                    except Exception:
                        selected_sf_ids = []
                # Fallback to deterministic if logs missing
                if not selected_sf_ids:
                    candidate_sf_ids = [
                        sf_id
                        for sf_id, pairs in sf_map.items()
                        if len(pairs) >= min_examples
                    ]
                    rng = random.Random(parity_seed)
                    k = min(parity_n_experts, len(candidate_sf_ids))
                    selected_sf_ids = rng.sample(candidate_sf_ids, k)

            else:
                # from_initial_forecasts (deterministic selection mirroring Delphi)
                candidate_sf_ids = [
                    sf_id
                    for sf_id, pairs in sf_map.items()
                    if len(pairs) >= min_examples
                ]
                if not candidate_sf_ids:
                    continue
                rng = random.Random(parity_seed)
                k = min(parity_n_experts, len(candidate_sf_ids))
                selected_sf_ids = rng.sample(candidate_sf_ids, k)

            # Aggregate with per-SF cap M
            for sf_id in selected_sf_ids:
                pairs = sf_map.get(sf_id, [])[:parity_per_sf_examples]
                agg_pairs.extend(pairs)
        else:
            # Non-parity: union of all SF examples
            agg_pairs = []
            for pairs in sf_map.values():
                agg_pairs.extend(pairs)

        # Deduplicate by (question_id, user_id)
        seen = set()
        deduped: List[Tuple["Question", "Forecast"]] = []
        for ex_q, ex_f in agg_pairs:
            key = (getattr(ex_q, "id", None), getattr(ex_f, "user_id", None))
            if key not in seen:
                seen.add(key)
                deduped.append((ex_q, ex_f))

        if len(deduped) < min_examples:
            continue
        if max_examples is not None:
            deduped = deduped[:max_examples]

        specs.append(
            TaskSpec(
                question=q,
                subject_type=SubjectType.AGGREGATED,
                subject_id="agg",
                examples=deduped,
            )
        )
    return specs


def _collect_example_forecasts(
    sampled_questions, loader, selected_date, *, min_examples=1, max_examples=5
):
    """
    Return {question_id: {sf_id: example_pairs}}, where each example pair is
    (question_obj, forecast_obj) — the same shape expected by
    `throttled_forecast_entry_with_examples`.
    """

    example_forecasts_dict = {}
    for q in sampled_questions:
        sf_ids = [
            f.user_id
            for f in loader.get_super_forecasts(
                question_id=q.id, resolution_date=selected_date
            )
        ]
        if not sf_ids:
            print(f"No superforecasters found for question {q.id}.")
            continue

        example_forecasts_dict[q.id] = {}
        for sf_id in sf_ids:
            forecasts = loader.get_super_forecasts(
                user_id=sf_id,
                resolution_date=selected_date,
                topic=q.topic,
            )

            # Remove forecasts for the target question and any in EVALUATION_QUESTION_IDS or EVOLUTION_EVALUATION_QUESTION_IDS
            forecasts = [
                f
                for f in forecasts
                if getattr(f, "id", None) != q.id
                and getattr(f, "id", None) not in EVALUATION_QUESTION_IDS
                and getattr(f, "id", None) not in EVOLUTION_EVALUATION_QUESTION_IDS
            ]

            example_pairs = []
            for f in forecasts:
                if getattr(f, "id", None) == q.id:
                    continue  # skip the target question itself
                q_obj = loader.get_question(f.id)  # ← direct lookup
                example_pairs.append((q_obj, f))
                if len(example_pairs) >= max_examples:
                    break

            example_forecasts_dict[q.id][sf_id] = example_pairs

    return example_forecasts_dict


def build_specs_baseline(
    sampled_questions,
) -> List[TaskSpec]:
    """One TaskSpec per question with no examples (baseline)."""
    return [
        TaskSpec(
            question=q,
            subject_type=SubjectType.BASELINE,
            subject_id=None,
            examples=None,
        )
        for q in sampled_questions
    ]


async def _run_specs(
    specs: List[TaskSpec],
    *,
    selected_resolution_date: str,
    forecast_due_date: str = "2024-07-21",
    concurrency: int = 5,
    timeout_s: int = 300,
    retries: int = 5,
    base_backoff_s: int = 10,
    n_samples: int = 1,
    config: RootConfig = None,
) -> List[Dict[str, Any]]:
    """
    Execute TaskSpecs and return normalized records:

    {
      "question_id": <str>,
      "subject_type": "superforecaster" | "baseline",
      "subject_id": <sf_id or None>,
      "date": <selected_resolution_date>,
      "mode": "with_examples" | "no_examples",
      "forecasts": [float, ...],
      "full_conversation": [...],
      "examples_used": [(Question, Forecast), ...] or [],
    }
    """
    sem = asyncio.Semaphore(concurrency)

    async def _call_one(spec: TaskSpec) -> Dict[str, Any]:
        async with sem:
            # Use provided llm or create one from config
            expert_llm = get_llm_from_config(config, role="expert")
            # Get expert config for other settings (temperature, etc.)
            expert = Expert(expert_llm, user_profile=None, config=config.model.expert)
            q_instance = copy.copy(spec.question)
            q_instance.resolution_date = selected_resolution_date
            q_instance.question = q_instance.question.replace(
                "{resolution_date}", selected_resolution_date
            )
            q_instance.question = q_instance.question.replace(
                "{forecast_due_date}", forecast_due_date
            )

            async def _one_try():
                if spec.examples:  # with examples → per-SF
                    return await expert.forecast_with_examples_in_context(
                        q_instance, spec.examples
                    )
                else:  # no examples → baseline
                    return await expert.forecast(q_instance)

            async def _retrying():
                for attempt in range(1, retries + 1):
                    try:
                        return await _one_try()
                    except openai.RateLimitError:
                        if attempt < retries:
                            sleep_t = base_backoff_s * (2 ** (attempt - 1))
                            print(
                                f"[{spec.question.id}] Rate limit (attempt {attempt}) — sleeping {sleep_t}s"
                            )
                            await asyncio.sleep(sleep_t)
                        else:
                            raise

            async def _sample_once():
                return await asyncio.wait_for(_retrying(), timeout=timeout_s)

            forecasts = await asyncio.gather(
                *(_sample_once() for _ in range(n_samples))
            )
            print(f"forecasts: {forecasts}")
            # Include richer example references when available for downstream reconstruction
            examples_used_pairs = []
            if spec.examples:
                for q_obj, f_obj in spec.examples:
                    try:
                        examples_used_pairs.append(
                            {
                                "question_id": getattr(q_obj, "id", None),
                                "user_id": getattr(f_obj, "user_id", None),
                            }
                        )
                    except Exception:
                        pass
            return {
                "question_id": spec.question.id,
                "subject_type": spec.subject_type.value,
                "subject_id": spec.subject_id,
                "date": selected_resolution_date,
                "mode": "with_examples" if spec.examples else "no_examples",
                "forecasts": forecasts,
                "full_conversation": expert.conversation_manager.messages,
                "examples_used": [q.id for q, _ in spec.examples]
                if spec.examples
                else [],
                "examples_used_pairs": examples_used_pairs,
            }

    return await asyncio.gather(*(_call_one(s) for s in specs), return_exceptions=True)


async def run_all_forecasts_with_examples(
    sampled_questions,
    *,
    loader=None,
    selected_resolution_date: str = None,
    forecast_due_date: str = None,
    min_examples: int = 1,
    max_examples: int = 5,
    concurrency: int = 5,
    timeout_s: int = 300,
    retries: int = 5,
    base_backoff_s: int = 10,
    n_samples: int = 1,
    config: RootConfig,
    llm=None,
):
    """
    PRODUCES: one record per (question, superforecaster).
    """
    # Use provided values or fall back to config/defaults
    if loader is None:
        loader = ForecastDataLoader()
    if config is None:
        config = RootConfig()

    # Get dates from config if not provided
    if selected_resolution_date is None:
        selected_resolution_date = config.data.resolution_date or "2025-07-21"
    if forecast_due_date is None:
        forecast_due_date = config.data.forecast_due_date or "2024-07-21"

    specs = build_specs_with_examples(
        sampled_questions,
        loader=loader,
        date=selected_resolution_date,
        min_examples=min_examples,
        max_examples=max_examples,
    )
    return await _run_specs(
        specs,
        selected_resolution_date=selected_resolution_date,
        forecast_due_date=forecast_due_date,
        concurrency=concurrency,
        timeout_s=timeout_s,
        retries=retries,
        base_backoff_s=base_backoff_s,
        n_samples=n_samples,
        config=config,
    )


async def run_all_forecasts_aggregated_examples(
    sampled_questions,
    *,
    loader=None,
    selected_resolution_date: str = None,
    forecast_due_date: str = None,
    min_examples: int = 1,
    max_examples: int = None,
    concurrency: int = 5,
    timeout_s: int = 300,
    retries: int = 5,
    base_backoff_s: int = 10,
    n_samples: int = 1,
    config: RootConfig = None,
    llm=None,
):
    """PRODUCES: one record per question with aggregated examples across SFs."""
    if loader is None:
        loader = ForecastDataLoader()
    if config is None:
        config = RootConfig()

    data_config = config.data
    if selected_resolution_date is None:
        selected_resolution_date = data_config.resolution_date or "2025-07-21"
    if forecast_due_date is None:
        forecast_due_date = data_config.forecast_due_date or "2024-07-21"

    specs = build_specs_aggregated(
        sampled_questions,
        loader=loader,
        date=selected_resolution_date,
        min_examples=min_examples,
        max_examples=max_examples,
        config=config,
    )
    return await _run_specs(
        specs,
        selected_resolution_date=selected_resolution_date,
        forecast_due_date=forecast_due_date,
        concurrency=concurrency,
        timeout_s=timeout_s,
        retries=retries,
        base_backoff_s=base_backoff_s,
        n_samples=n_samples,
        config=config,
    )


async def run_all_forecasts_shared_examples(
    sampled_questions,
    *,
    loader=None,
    selected_resolution_date: str = None,
    forecast_due_date: str = None,
    min_examples: int = 1,
    max_examples: int = None,
    concurrency: int = 5,
    timeout_s: int = 300,
    retries: int = 5,
    base_backoff_s: int = 10,
    n_samples: int = 1,
    n_experts: Optional[int] = None,
    config: RootConfig = None,
    llm=None,
):
    """
    PRODUCES: multiple records per question; each record uses the same example set.
    n_experts controls how many independent experts share the same examples.
    """
    if loader is None:
        loader = ForecastDataLoader()
    if config is None:
        config = RootConfig()

    data_config = config.data
    if selected_resolution_date is None:
        selected_resolution_date = data_config.resolution_date or "2025-07-21"
    if forecast_due_date is None:
        forecast_due_date = data_config.forecast_due_date or "2024-07-21"

    # Determine number of experts from config if not provided
    if n_experts is None:
        try:
            n_experts = int((config.delphi or {}).get("n_experts", 1))
        except Exception:
            n_experts = 1

    # Build one aggregated spec per question (respects parity settings)
    agg_specs = build_specs_aggregated(
        sampled_questions,
        loader=loader,
        date=selected_resolution_date,
        min_examples=min_examples,
        max_examples=max_examples,
        config=config,
    )

    # Replicate examples across n_experts with synthetic subject ids
    specs: List[TaskSpec] = []
    for s in agg_specs:
        for i in range(max(1, n_experts)):
            specs.append(
                TaskSpec(
                    question=s.question,
                    subject_type=SubjectType.SHARED,
                    subject_id=f"shared_{i}",
                    examples=s.examples,
                )
            )

    return await _run_specs(
        specs,
        selected_resolution_date=selected_resolution_date,
        forecast_due_date=forecast_due_date,
        concurrency=concurrency,
        timeout_s=timeout_s,
        retries=retries,
        base_backoff_s=base_backoff_s,
        n_samples=n_samples,
        config=config,
    )


async def run_all_forecasts_baseline(
    sampled_questions,
    *,
    selected_resolution_date: str = None,
    forecast_due_date: str = None,
    concurrency: int = 5,
    timeout_s: int = 300,
    retries: int = 10,
    base_backoff_s: int = 10,
    n_samples: int = 5,
    config: dict = None,
):
    """
    PRODUCES: one record per question (no SF id).
    """
    # Use provided values or fall back to config/defaults
    if config is None:
        config = {}

    # Get dates from config if not provided
    data_config = config.get("data", {})
    if selected_resolution_date is None:
        selected_resolution_date = data_config.get("resolution_date", "2025-07-21")
    if forecast_due_date is None:
        forecast_due_date = data_config.get("forecast_due_date", "2024-07-21")

    specs = build_specs_baseline(sampled_questions)
    return await _run_specs(
        specs,
        selected_resolution_date=selected_resolution_date,
        forecast_due_date=forecast_due_date,
        concurrency=concurrency,
        timeout_s=timeout_s,
        retries=retries,
        base_backoff_s=base_backoff_s,
        n_samples=n_samples,
        config=config,
    )


def generate_forecasts_for_questions(
    sampled_questions,
    selected_resolution_date,
    initial_forecasts_path,
    forecast_due_date="2024-07-21",
    config=None,
    with_examples=True,
):
    """Generate forecasts (with or without examples) for a list of questions."""
    forecast_type = "with_examples" if with_examples else "no_examples"
    print(f"Collecting forecasts {forecast_type}...")
    print(f"{len(sampled_questions)} questions to collect forecasts for")
    print(sampled_questions)

    for q in sampled_questions:
        json_filename = (
            f"collected_fcasts_{forecast_type}_{selected_resolution_date}_{q.id}.json"
        )
        json_path = os.path.join(initial_forecasts_path, json_filename)
        if os.path.exists(json_path):
            print(
                f"JSON for question {q.id} ({forecast_type}) already exists, skipping."
            )
            continue

        pickle_filename = (
            f"collected_fcasts_{forecast_type}_{selected_resolution_date}_{q.id}.pkl"
        )
        pickle_path = os.path.join(initial_forecasts_path, pickle_filename)

        if os.path.exists(pickle_path):
            print(
                f"Pickle for question {q.id} ({forecast_type}) already exists, converting to json."
            )
            convert_pkl_to_json(pickle_path, json_path)

        print(f"Collecting {forecast_type} forecasts for question {q.id}...")

        if with_examples:
            results = asyncio.run(
                run_all_forecasts_with_examples(
                    [q],
                    selected_resolution_date=selected_resolution_date,
                    forecast_due_date=forecast_due_date,
                    config=config,
                )
            )
        else:
            results = asyncio.run(
                run_all_forecasts_baseline(
                    [q],
                    selected_resolution_date=selected_resolution_date,
                    forecast_due_date=forecast_due_date,
                    config=config,
                )
            )

        with open(json_path, "w") as f:
            json.dump(results, f)
        print(f"Collected {forecast_type} forecasts for question {q.id}.")


def run_initial_forecast_experiment(config_path=None):
    """Main runner function for initial forecast generation and analysis."""
    config = load_typed_experiment_config(config_path)
    print(f"Loaded configuration from: {config_path}")

    # Get configuration values
    data_config = config.get("data", {})
    experiment_config = config.get("experiment", {})

    # Extract dates and paths from config
    selected_resolution_date = data_config.get("resolution_date", "2025-07-21")
    forecast_due_date = data_config.get("forecast_due_date", "2024-07-21")
    initial_forecasts_path = experiment_config.get(
        "initial_forecasts_dir", "results_initial_forecasts"
    )

    # Set random seed
    seed = experiment_config.get("seed")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Initialize data loader
    loader = ForecastDataLoader()
    questions_with_topic = loader.get_questions_with_topics()

    # Get sampling configuration
    sampling_config = data_config.get("sampling", {})
    n_per_topic = sampling_config.get("n_per_topic", 3)

    # Sample questions
    sampled_questions = sample_questions_by_topic(
        questions_with_topic, n_per_topic=n_per_topic
    )
    print(f"Sampled {len(sampled_questions)} questions")

    # Filter questions with resolutions
    sampled_questions = [
        q
        for q in sampled_questions
        if loader.get_resolution(
            question_id=q.id, resolution_date=selected_resolution_date
        )
        is not None
    ]
    print(f"Filtered to {len(sampled_questions)} questions with resolutions")

    # Ensure output directory exists
    if not os.path.exists(initial_forecasts_path):
        os.makedirs(initial_forecasts_path)

    # Generate forecasts with examples
    generate_forecasts_for_questions(
        sampled_questions,
        selected_resolution_date,
        initial_forecasts_path,
        forecast_due_date=forecast_due_date,
        config=config,
        with_examples=True,
    )

    # Generate baseline forecasts (no examples)
    generate_forecasts_for_questions(
        sampled_questions,
        selected_resolution_date,
        initial_forecasts_path,
        forecast_due_date=forecast_due_date,
        config=config,
        with_examples=False,
    )

    # Load and analyze results
    from utils.forecast_loader import load_forecast_jsons

    loaded_with_examples, loaded_no_examples = load_forecast_jsons(
        initial_forecasts_path, selected_resolution_date, loader
    )

    sf_aggregate, q_aggregate, qid_to_label = analyze_forecast_results(
        sampled_questions,
        loaded_with_examples,
        loaded_no_examples,
        loader,
        selected_resolution_date,
        forecast_due_date,
    )

    print("\nAnalysis complete:")
    print(f"  - Analyzed {len(q_aggregate)} questions")
    print(f"  - Found data for {len(sf_aggregate)} superforecasters")

    return {
        "sampled_questions": sampled_questions,
        "loaded_with_examples": loaded_with_examples,
        "loaded_no_examples": loaded_no_examples,
        "sf_aggregate": sf_aggregate,
        "q_aggregate": q_aggregate,
        "qid_to_label": qid_to_label,
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run initial forecasts with ICL")
    parser.add_argument(
        "config_path",
        nargs="?",  # Optional argument
        default="",
        help="Path to experiment configuration YAML file",
    )
    args = parser.parse_args()
    results = run_initial_forecast_experiment(args.config_path)
    print("\nExperiment completed successfully!")
