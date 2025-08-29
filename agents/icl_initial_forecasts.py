import argparse
import asyncio
import copy
import json
import os
import pickle
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
from utils.utils import load_experiment_config

from utils.convert_pickles_to_json import convert_pkl_to_json, batch_convert_pickles

import debugpy

if not debugpy.is_client_connected():
    debugpy.listen(("localhost", 5679))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()


from utils.sampling import (TRAIN_QUESTION_IDS, EVALUATION_QUESTION_IDS, EVOLUTION_EVALUATION_QUESTION_IDS,)

load_dotenv()


class SubjectType(str, Enum):
    SUPERFORECASTER = "superforecaster"
    BASELINE = "baseline"   # no-examples, no SF id

@dataclass
class TaskSpec:
    question: "Question"
    subject_type: SubjectType
    subject_id: Optional[str]                       # sf_id for SUPERFORECASTER; None for BASELINE
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
    all_examples = _collect_example_forecasts(sampled_questions, loader, date, min_examples=min_examples,max_examples=max_examples)
    for q in sampled_questions:
        for sf_id, pairs in all_examples.get(q.id, {}).items():
            if len(pairs) < min_examples:
                continue
            specs.append(TaskSpec(
                question=q,
                subject_type=SubjectType.SUPERFORECASTER,
                subject_id=sf_id,
                examples=pairs[:max_examples],
            ))
    return specs

def _collect_example_forecasts(sampled_questions, loader, selected_date, *, min_examples=1, max_examples=5):
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
                f for f in forecasts
                if getattr(f, "id", None) != q.id
                and getattr(f, "id", None) not in EVALUATION_QUESTION_IDS
                and getattr(f, "id", None) not in EVOLUTION_EVALUATION_QUESTION_IDS
            ]

            example_pairs = []
            for f in forecasts:
                if getattr(f, "id", None) == q.id:
                    continue  # skip the target question itself
                q_obj = loader.get_question(f.id)          # ← direct lookup
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
    config: dict = None,
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
            expert_llm = get_llm_from_config(config, role='expert')
            # Get expert config for other settings (temperature, etc.)
            if config:
                expert_config = config.get("model", {}).get("expert", config.get("model", {}))
            else:
                expert_config = {}
            expert = Expert(expert_llm, user_profile=None, config=expert_config)
            q_instance = copy.copy(spec.question)
            q_instance.resolution_date = selected_resolution_date
            q_instance.question = q_instance.question.replace("{resolution_date}", selected_resolution_date)
            q_instance.question = q_instance.question.replace("{forecast_due_date}", forecast_due_date)

            async def _one_try():
                if spec.examples:  # with examples → per-SF
                    return await expert.forecast_with_examples_in_context(q_instance, spec.examples)
                else:              # no examples → baseline
                    return await expert.forecast(q_instance)

            async def _retrying():
                for attempt in range(1, retries + 1):
                    try:
                        return await _one_try()
                    except openai.RateLimitError as e:
                        if attempt < retries:
                            sleep_t = base_backoff_s * (2 ** (attempt - 1))
                            print(f"[{spec.question.id}] Rate limit (attempt {attempt}) — sleeping {sleep_t}s")
                            await asyncio.sleep(sleep_t)
                        else:
                            raise

            async def _sample_once():
                return await asyncio.wait_for(_retrying(), timeout=timeout_s)

            forecasts = await asyncio.gather(*(_sample_once() for _ in range(n_samples)))
            print(f"forecasts: {forecasts}")
            return {
                "question_id": spec.question.id,
                "subject_type": spec.subject_type.value,
                "subject_id": spec.subject_id,
                "date": selected_resolution_date,
                "mode": "with_examples" if spec.examples else "no_examples",
                "forecasts": forecasts,
                "full_conversation": expert.conversation_manager.messages,
                "examples_used": [q.id for q, _ in spec.examples] if spec.examples else [],
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
    config: dict = None,
    llm = None,
):
    """
    PRODUCES: one record per (question, superforecaster).
    """
    # Use provided values or fall back to config/defaults
    if loader is None:
        loader = ForecastDataLoader()
    if config is None:
        config = {}

    # Get dates from config if not provided
    data_config = config.get('data', {})
    if selected_resolution_date is None:
        selected_resolution_date = data_config.get('resolution_date', '2025-07-21')
    if forecast_due_date is None:
        forecast_due_date = data_config.get('forecast_due_date', '2024-07-21')

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
    data_config = config.get('data', {})
    if selected_resolution_date is None:
        selected_resolution_date = data_config.get('resolution_date', '2025-07-21')
    if forecast_due_date is None:
        forecast_due_date = data_config.get('forecast_due_date', '2024-07-21')

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

def generate_forecasts_for_questions(sampled_questions, selected_resolution_date, initial_forecasts_path,
                                    forecast_due_date="2024-07-21", config=None, with_examples=True):
    """Generate forecasts (with or without examples) for a list of questions."""
    forecast_type = "with_examples" if with_examples else "no_examples"
    print(f'Collecting forecasts {forecast_type}...')
    print(f'{len(sampled_questions)} questions to collect forecasts for')
    print(sampled_questions)

    for q in sampled_questions:

        json_filename = f'collected_fcasts_{forecast_type}_{selected_resolution_date}_{q.id}.json'
        json_path = os.path.join(initial_forecasts_path, json_filename)
        if os.path.exists(json_path):
            print(f"JSON for question {q.id} ({forecast_type}) already exists, skipping.")
            continue

        pickle_filename = f'collected_fcasts_{forecast_type}_{selected_resolution_date}_{q.id}.pkl'
        pickle_path = os.path.join(initial_forecasts_path, pickle_filename)

        if os.path.exists(pickle_path):
            print(f"Pickle for question {q.id} ({forecast_type}) already exists, converting to json.")
            convert_pkl_to_json(pickle_path, json_path)

        print(f"Collecting {forecast_type} forecasts for question {q.id}...")

        if with_examples:
            results = asyncio.run(run_all_forecasts_with_examples(
                [q],
                selected_resolution_date=selected_resolution_date,
                forecast_due_date=forecast_due_date,
                config=config
            ))
        else:
            results = asyncio.run(run_all_forecasts_baseline(
                [q],
                selected_resolution_date=selected_resolution_date,
                forecast_due_date=forecast_due_date,
                config=config
            ))

        with open(json_path, 'w') as f:
            json.dump(results, f)
        print(f"Collected {forecast_type} forecasts for question {q.id}.")


def run_initial_forecast_experiment(config_path=None):
    """Main runner function for initial forecast generation and analysis."""
    config = load_experiment_config(config_path)
    print(f"Loaded configuration from: {config_path}")

    # Get configuration values
    data_config = config.get('data', {})
    experiment_config = config.get('experiment', {})

    # Extract dates and paths from config
    selected_resolution_date = data_config.get('resolution_date', '2025-07-21')
    forecast_due_date = data_config.get('forecast_due_date', '2024-07-21')
    initial_forecasts_path = experiment_config.get('initial_forecasts_dir', 'results_initial_forecasts')

    # Set random seed
    seed = experiment_config.get('seed')
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Initialize data loader
    loader = ForecastDataLoader()
    questions_with_topic = loader.get_questions_with_topics()

    # Get sampling configuration
    sampling_config = data_config.get('sampling', {})
    n_per_topic = sampling_config.get('n_per_topic', 3)

    # Sample questions
    sampled_questions = sample_questions_by_topic(questions_with_topic, n_per_topic=n_per_topic)
    print(f"Sampled {len(sampled_questions)} questions")

    # Filter questions with resolutions
    sampled_questions = [
        q for q in sampled_questions
        if loader.get_resolution(question_id=q.id, resolution_date=selected_resolution_date) is not None
    ]
    print(f"Filtered to {len(sampled_questions)} questions with resolutions")

    # Ensure output directory exists
    if not os.path.exists(initial_forecasts_path):
        os.makedirs(initial_forecasts_path)

    # Generate forecasts with examples
    generate_forecasts_for_questions(
        sampled_questions, selected_resolution_date, initial_forecasts_path,
        forecast_due_date=forecast_due_date, config=config, with_examples=True
    )

    # Generate baseline forecasts (no examples)
    generate_forecasts_for_questions(
        sampled_questions, selected_resolution_date, initial_forecasts_path,
        forecast_due_date=forecast_due_date, config=config, with_examples=False
    )

    # Load and analyze results
    from utils.forecast_loader import load_forecast_jsons
    loaded_with_examples, loaded_no_examples = load_forecast_jsons(initial_forecasts_path, selected_resolution_date, loader)

    sf_aggregate, q_aggregate, qid_to_label = analyze_forecast_results(
        sampled_questions, loaded_with_examples, loaded_no_examples,
        loader, selected_resolution_date, forecast_due_date
    )

    print(f"\nAnalysis complete:")
    print(f"  - Analyzed {len(q_aggregate)} questions")
    print(f"  - Found data for {len(sf_aggregate)} superforecasters")

    return {
        'sampled_questions': sampled_questions,
        'loaded_with_examples': loaded_with_examples,
        'loaded_no_examples': loaded_no_examples,
        'sf_aggregate': sf_aggregate,
        'q_aggregate': q_aggregate,
        'qid_to_label': qid_to_label
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run initial forecasts with ICL")
    parser.add_argument(
        "config_path",
        nargs="?",  # Optional argument
        default='',
        help="Path to experiment configuration YAML file"
    )
    args = parser.parse_args()
    results = run_initial_forecast_experiment(args.config_path)
    print("\nExperiment completed successfully!")
