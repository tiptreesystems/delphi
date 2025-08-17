from delphi import Expert
from models import LLMFactory, LLMProvider, LLMModel
from dataset.dataloader import Question, Forecast, Resolution, ForecastDataLoader

# load_config import removed - using local load_experiment_config instead
import os
from collections import defaultdict

import random
import copy
import asyncio
import time
import pickle
import json
import numpy as np

from dotenv import load_dotenv
load_dotenv()

import argparse
import yaml

import debugpy
# print("Waiting for debugger attach...")
# debugpy.listen(5679)
# debugpy.wait_for_client()
# print("Debugger attached.")

# Set all random seeds for reproducibility

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


import openai
import textwrap
import matplotlib.pyplot as plt

def load_experiment_config(config_path: str = './configs/config_openai.yml') -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Default config path - will be overridden by command line argument
default_config_path = "./configs/config_openai.yml"
config = None  # Will be loaded in main

resolutions_path = "./dataset/datasets/resolution_sets/2024-07-21_resolution_set.json"

# These will be set from config in main
provider = None
model = None
personalized_system_prompt = None
llm = None

# Loader will be initialized after config is loaded
loader = None
questions_with_topic = None

forecast_due_date = "2024-07-21"  # Example date, adjust as needed
selected_resolution_date = "2025-07-21"
initial_forecasts_path = 'outputs_initial_forecasts'
if not os.path.exists(initial_forecasts_path):
    os.makedirs(initial_forecasts_path)

n_samples = 5

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import asyncio
import copy


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
    concurrency: int = 5,
    timeout_s: int = 300,
    retries: int = 5,
    base_backoff_s: int = 10,
    n_samples: int = 1,
    config: dict = None,
    llm = None,
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
            # Use expert config from the loaded configuration
            if config:
                expert_config = config.get("model", {}).get("expert", config.get("model", {}))
            else:
                expert_config = {}
            
            # Use provided llm or try to get from globals
            expert_llm = llm if llm is not None else globals().get('llm')
            if expert_llm is None:
                raise ValueError("LLM instance not provided and not found in globals")

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

            return {
                "question_id": spec.question.id,
                "subject_type": spec.subject_type.value,
                "subject_id": spec.subject_id,
                "date": selected_resolution_date,
                "mode": "with_examples" if spec.examples else "no_examples",
                "forecasts": forecasts,
                "full_conversation": expert.conversation_manager.messages,
                "examples_used": spec.examples or [],
            }

    return await asyncio.gather(*(_call_one(s) for s in specs), return_exceptions=True)


async def run_all_forecasts_with_examples(
    sampled_questions,
    *,
    loader=None,
    selected_resolution_date: str = None,
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
    # Use provided values or fall back to globals/defaults
    if loader is None:
        loader = globals().get('loader') or ForecastDataLoader()
    if selected_resolution_date is None:
        selected_resolution_date = globals().get('selected_resolution_date', '2025-07-21')
    if config is None:
        config = globals().get('config', {})
    if llm is None:
        llm = globals().get('llm')
        
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
        concurrency=concurrency,
        timeout_s=timeout_s,
        retries=retries,
        base_backoff_s=base_backoff_s,
        n_samples=n_samples,
        config=config,
        llm=llm,
    )

async def run_all_forecasts_baseline(
    sampled_questions,
    *,
    selected_resolution_date: str = None,
    concurrency: int = 5,
    timeout_s: int = 300,
    retries: int = 10,
    base_backoff_s: int = 10,
    n_samples: int = 5,
    config: dict = None,
    llm = None,
):
    """
    PRODUCES: one record per question (no SF id).
    """
    # Use provided values or fall back to globals/defaults
    if selected_resolution_date is None:
        selected_resolution_date = globals().get('selected_resolution_date', '2025-07-21')
    if config is None:
        config = globals().get('config', {})
    if llm is None:
        llm = globals().get('llm')
        
    specs = build_specs_baseline(sampled_questions)
    return await _run_specs(
        specs,
        selected_resolution_date=selected_resolution_date,
        concurrency=concurrency,
        timeout_s=timeout_s,
        retries=retries,
        base_backoff_s=base_backoff_s,
        n_samples=n_samples,
        config=config,
        llm=llm,
    )

def sample_questions_by_topic(questions, n_per_topic=None, seed=42):
    random.seed(seed)
    unique_topics = set(q.topic for q in questions)
    topic_to_questions = defaultdict(list)

    shuffled_questions = random.sample(questions, len(questions))
    for topic in unique_topics:
        topic_questions = [q for q in shuffled_questions if q.topic == topic]
        if n_per_topic is None:
            topic_to_questions[topic] = topic_questions
        else:
            if len(topic_questions) < n_per_topic:
                raise ValueError(
                    f"Topic '{topic}' only has {len(topic_questions)} questions, "
                    f"but {n_per_topic} are required."
                )
            topic_to_questions[topic] = topic_questions[:n_per_topic]

    sampled_questions = [q for qs in topic_to_questions.values() for q in qs]
    return sampled_questions

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run initial forecasts with ICL")
    parser.add_argument(
        "config_path", 
        nargs="?",  # Optional argument
        default=default_config_path,
        help="Path to experiment configuration YAML file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_experiment_config(args.config_path)
    print(f"Loaded configuration from: {args.config_path}")
    
    # Setup model based on configuration
    model_config = config.get('model', {})
    
    # Determine provider and model from config
    provider_str = model_config.get('provider', 'openai').lower()
    if provider_str == 'openai':
        provider = LLMProvider.OPENAI
        model = LLMModel.GPT_4O_2024_05_13
        api_key = os.getenv(config.get('api', {}).get('openai', {}).get('api_key_env', 'OPENAI_API_KEY'))
        os.environ["OPENAI_API_KEY"] = api_key
    elif provider_str == 'groq':
        provider = LLMProvider.GROQ
        # Map model name to enum
        model_name = model_config.get('name', 'deepseek-r1-distill-llama-70b')
        if 'deepseek-r1-distill-llama-70b' in model_name:
            model = LLMModel.GROQ_DEEPSEEK_R1_DISTILL_70B
        else:
            # Fallback to a default Groq model
            model = LLMModel.GROQ_DEEPSEEK_R1_DISTILL_70B
        api_key = os.getenv(config.get('api', {}).get('groq', {}).get('api_key_env', 'GROQ_API_KEY'))
        os.environ["GROQ_API_KEY"] = api_key
    else:
        raise ValueError(f"Unsupported provider: {provider_str}")
    
    personalized_system_prompt = model_config.get(
        'system_prompt',
        "You are a helpful assistant with expertise in forecasting and decision-making."
    )
    
    # Create LLM instance
    llm = LLMFactory.create_llm(provider, model, system_prompt=personalized_system_prompt)
    
    # Initialize data loader and get questions
    loader = ForecastDataLoader()
    questions_with_topic = loader.get_questions_with_topics()
    
    # Get sampling configuration
    data_config = config.get('data', {})
    sampling_config = data_config.get('sampling', {})
    n_per_topic = sampling_config.get('n_per_topic', 3)
    
    sampled_questions = sample_questions_by_topic(questions_with_topic, n_per_topic=n_per_topic)

    # Remove questions that do not have a resolution on the selected date
    sampled_questions = [
        q for q in sampled_questions
        if loader.get_resolution(question_id=q.id, resolution_date=selected_resolution_date) is not None
    ]


    for q in sampled_questions:
        if os.path.exists(f'{initial_forecasts_path}/collected_fcasts_with_examples_{selected_resolution_date}_{q.id}.pkl'):
            print(f"Pickle for question {q.id} already exists, skipping.")
            continue
        print(f"Collecting forecasts for question {q.id}...")
        results = asyncio.run(run_all_forecasts_with_examples([q]))
        with open(f'{initial_forecasts_path}/collected_fcasts_with_examples_{selected_resolution_date}_{q.id}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"Collected forecasts for question {q.id}.")

    print('Moving to no-example forecasts...')
    for q in sampled_questions:
        if os.path.exists(f'{initial_forecasts_path}/collected_fcasts_no_examples_{selected_resolution_date}_{q.id}.pkl'):
            print(f"Pickle for question {q.id} (no examples) already exists, skipping.")
            continue
        print(f"Collecting no-example forecasts for question {q.id}...")
        results = asyncio.run(run_all_forecasts_baseline([q]))
        with open(f'{initial_forecasts_path}/collected_fcasts_no_examples_{selected_resolution_date}_{q.id}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"Collected no-example forecasts for question {q.id}.")


    pkl_files = [
        f for f in os.listdir(f"{initial_forecasts_path}/")
        if f.startswith("collected_fcasts") and f.endswith(".pkl") and f"{selected_resolution_date}" in f
    ]

    # Split pkl files into with_examples and no_examples
    with_examples_files = [
        f for f in pkl_files
        if f.startswith("collected_fcasts_with_examples") and f"{selected_resolution_date}" in f
    ]
    no_examples_files = [
        f for f in pkl_files
        if f.startswith("collected_fcasts_no_examples") and f"{selected_resolution_date}" in f
    ]


    loaded_fcasts_with_examples = {}
    for fname in with_examples_files:
        # Extract question id between 'collected_fcasts_' and '.pkl'
        qid = fname[len(f"collected_fcasts_with_examples_{selected_resolution_date}_"): -len(".pkl")]
        with open(f"{initial_forecasts_path}/{fname}", "rb") as f:
            loaded_fcasts_with_examples[qid] = [q for q in pickle.load(f)]

    loaded_fcasts_no_examples = {}
    for fname in no_examples_files:
        # Extract question id between 'collected_fcasts_no_examples_' and '.pkl'
        qid = fname[len(f"collected_fcasts_no_examples_{selected_resolution_date}_"): -len(".pkl")]
        with open(f"{initial_forecasts_path}/{fname}", "rb") as f:
            loaded_fcasts_no_examples[qid] = [q for q in pickle.load(f)]

    rng = np.random.default_rng(42)
    sf_aggregate = defaultdict(list)
    # For each question, collect a list of (ae_sf, ae_base) across SFIDs
    q_aggregate = defaultdict(list)
    qid_to_label = {}  # optional: pretty label per qid

    for q in sampled_questions:
        qid = q.id

        sf_payloads = loaded_fcasts_with_examples.get(qid, [])
        base_payloads = loaded_fcasts_no_examples.get(qid, [])

        if not sf_payloads or not base_payloads:
            print(f"[skip] Missing data for qid={qid} (sf_payloads={len(sf_payloads)}, base_payloads={len(base_payloads)})")
            continue

        # --- tiny helpers (scoped to this loop to keep diff minimal) ---
        def _extract_forecasts_from_payload(payload):
            """
            Handles shapes like:
              {'forecast': {'forecasts': [...]}}  (with-examples payload)
              {'forecasts': [...]}                (possible baseline payload)
              [...]                               (already a list of floats)
            """
            if isinstance(payload, dict):
                if "forecast" in payload and isinstance(payload["forecast"], dict) and "forecasts" in payload["forecast"]:
                    return payload["forecast"]["forecasts"]
                if "forecasts" in payload and isinstance(payload["forecasts"], (list, tuple)):
                    return payload["forecasts"]
            if isinstance(payload, (list, tuple)):
                return payload
            raise ValueError(f"Unrecognized payload shape for forecasts: {type(payload)} keys={list(payload.keys()) if isinstance(payload, dict) else 'N/A'}")

        def _safe_super_gt(qid_, sfid_):
            try:
                gt = loader.get_super_forecasts(
                    question_id=qid_,
                    user_id=sfid_,
                    resolution_date=selected_resolution_date
                )
                if gt:
                    # handle object vs. raw
                    val = getattr(gt[0], "forecast", gt[0])
                    return getattr(val, "value", val)
            except Exception as e:
                print(f"[warn] ground truth missing for qid={qid_}, sfid={sfid_}: {e}")
            return None
        # ----------------------------------------------------------------

        # Collect violin data for each SF (with examples)
        plot_data = []
        labels = []
        gt_per_sfid = []

        for payload in sf_payloads:
            sfid = payload.get("superforecaster_id", "unknown_sfid")
            try:
                forecasts = _extract_forecasts_from_payload(payload)
            except Exception as e:
                print(f"[skip] qid={qid}, sfid={sfid} extract error: {e}")
                continue

            if not forecasts:
                continue

            plot_data.append(forecasts)
            labels.append(sfid)

            gt_val = _safe_super_gt(qid, sfid)
            gt_per_sfid.append(gt_val)

        if not plot_data:
            print(f"[skip] No SF plot data for qid={qid}")
            continue

        # Add the baseline (no-examples) as the final violin
        base_item = base_payloads[0]
        try:
            base_forecasts = _extract_forecasts_from_payload(base_item)
        except Exception as e:
            print(f"[skip] qid={qid} baseline extract error: {e}")
            continue

        plot_data.append(base_forecasts)
        labels.append("No-Examples Baseline")

        # --- Plot ---
        plt.figure(figsize=(12, 6))
        positions = np.arange(1, len(labels) + 1)
        parts = plt.violinplot(plot_data, positions=positions, showmeans=True, widths=0.8)

        # Color the baseline violin (last one) to distinguish it
        try:
            if hasattr(parts, "bodies") and len(parts.bodies) >= 1:
                parts.bodies[-1].set_facecolor("orange")
                parts.bodies[-1].set_edgecolor("orange")
                parts.bodies[-1].set_alpha(0.7)
        except Exception as e:
            print(f"[warn] Could not color baseline violin: {e}")

        # X ticks
        plt.xticks(positions, labels, rotation=45, ha="right")

        # Draw per-SF ground truth as short dashed lines centered under each SF violin (skip baseline)
        n = len(labels)
        for i, gt_val in enumerate(gt_per_sfid):
            if gt_val is None:
                continue
            # Restrict the line span to the i-th violin region (avoid spanning entire axis)
            # positions are 1-indexed; normalize to [0,1] using count n
            xmin = (i + 0.2) / n
            xmax = (i + 0.8) / n
            plt.axhline(gt_val, color="red", linestyle="--", linewidth=1, xmin=xmin, xmax=xmax)

        suptitle_question = q.question
        q.question = q.question.replace("{resolution_date}", selected_resolution_date)
        q.question = q.question.replace("{forecast_due_date}", forecast_due_date)
        wrapped_title = "\n".join(textwrap.wrap(q.question, width=80))
        plt.title(f"LLM w/ Examples per Superforecaster vs. No-Examples Baseline — {qid}")
        plt.suptitle(wrapped_title, fontsize=10, y=1.02)

        plt.ylabel("Forecast Value")
        # Dynamically set y-limits based on data and ground truth values
        all_values = [v for forecasts in plot_data for v in forecasts if forecasts]
        all_gt = [gt for gt in gt_per_sfid if gt is not None]
        all_for_ylim = all_values + all_gt
        if all_for_ylim:
            min_y = min(all_for_ylim)
            max_y = max(all_for_ylim)
            y_margin = 0.05 * (max_y - min_y) if max_y > min_y else 0.05
            plt.ylim(min_y - y_margin, max_y + y_margin)
        else:
            plt.ylim(-0.05, 1.05)
        plt.xlim(0.5, len(labels) + 0.5)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        os.makedirs("initial_icl_tests_plots", exist_ok=True)
        safe_qid = qid.replace("/", "_").replace("\\", "_")
        filename = f"initial_icl_tests_plots/{safe_qid}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[saved] {filename}")


        # --- accumulate per-(qid, sfid) absolute errors for cross-question aggregation ---
        for payload in sf_payloads:
            sfid = payload.get("superforecaster_id", "unknown_sfid")
            gt_val = _safe_super_gt(qid, sfid)
            if gt_val is None:
                continue
            try:
                sf_forecasts = _extract_forecasts_from_payload(payload)
            except Exception:
                continue
            if not sf_forecasts:
                continue

            ae_sf = np.abs(np.asarray(sf_forecasts, dtype=float) - float(gt_val))
            ae_base = np.abs(np.asarray(base_forecasts, dtype=float) - float(gt_val))

            if len(ae_sf) and len(ae_base):
                sf_aggregate[sfid].append((ae_sf, ae_base))

        if qid not in qid_to_label:
            pretty = q.question.replace("{resolution_date}", selected_resolution_date).replace("{forecast_due_date}", forecast_due_date)
            qid_to_label[qid] = pretty[:80] + ("…" if len(pretty) > 80 else "")

        for payload in sf_payloads:
            sfid = payload.get("superforecaster_id", "unknown_sfid")
            gt_val = _safe_super_gt(qid, sfid)
            if gt_val is None:
                continue
            try:
                sf_forecasts = _extract_forecasts_from_payload(payload)
            except Exception:
                continue
            if not sf_forecasts:
                continue

            ae_sf = np.abs(np.asarray(sf_forecasts, dtype=float) - float(gt_val))
            ae_base = np.abs(np.asarray(base_forecasts, dtype=float) - float(gt_val))
            if len(ae_sf) and len(ae_base):
                q_aggregate[qid].append((ae_sf, ae_base))


        # --- 2nd plot: per-SF improvement over baseline (bar chart with SD as thick vertical line) ---
        # improvement := mean(|baseline - gt|) - mean(|sf - gt|);  positive => SF closer to GT than baseline
        # rng = np.random.default_rng(42)

        # def _bootstrap_sd_improvement(ae_sf, ae_base, B=500):
        #     ae_sf = np.asarray(ae_sf, dtype=float)
        #     ae_base = np.asarray(ae_base, dtype=float)
        #     n_sf, n_b = len(ae_sf), len(ae_base)
        #     if n_sf == 0 or n_b == 0:
        #         return np.nan
        #     samples = []
        #     for _ in range(B):
        #         sf_s = ae_sf[rng.integers(0, n_sf, n_sf)]
        #         b_s  = ae_base[rng.integers(0, n_b, n_b)]
        #         samples.append(b_s.mean() - sf_s.mean())
        #     return float(np.std(samples, ddof=1))

        # # Precompute baseline absolute errors once per question
        # # (we already extracted base_forecasts above)
        # # Need a ground truth per SF, so we compute improvements per-SF separately.
        # improvements = []
        # sd_improvements = []
        # sf_labels_for_bars = []

        # # We'll compute a baseline AE vector against each SF's GT (since GT may differ per SF)
        # for payload in sf_payloads:
        #     sfid = payload.get("superforecaster_id", "unknown_sfid")
        #     gt_val = _safe_super_gt(qid, sfid)
        #     if gt_val is None:
        #         print(f"[skip] No GT for bar chart qid={qid}, sfid={sfid}")
        #         continue

        #     try:
        #         sf_forecasts = _extract_forecasts_from_payload(payload)
        #     except Exception as e:
        #         print(f"[skip] qid={qid}, sfid={sfid} extract error (bar): {e}")
        #         continue
        #     if not sf_forecasts:
        #         continue

        #     ae_sf = np.abs(np.asarray(sf_forecasts, dtype=float) - float(gt_val))
        #     ae_base = np.abs(np.asarray(base_forecasts, dtype=float) - float(gt_val))

        #     imp = float(ae_base.mean() - ae_sf.mean())
        #     sd_imp = _bootstrap_sd_improvement(ae_sf, ae_base, B=500)

        #     improvements.append(imp)
        #     sd_improvements.append(sd_imp)
        #     sf_labels_for_bars.append(sfid)

        # if not improvements:
        #     print(f"[skip] No improvements to plot for qid={qid}")
        # else:
        #     plt.figure(figsize=(12, 5))
        #     x = np.arange(len(sf_labels_for_bars))
        #     bars = plt.bar(x, improvements)

        #     # Thick vertical line = SD of improvement (not SE)
        #     plt.errorbar(
        #         x,
        #         improvements,
        #         yerr=sd_improvements,
        #         fmt="none",
        #         ecolor="black",
        #         elinewidth=3,
        #         capsize=6,
        #     )

        #     # Zero line for reference
        #     plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)

        #     plt.xticks(x, sf_labels_for_bars, rotation=45, ha="right")
        #     plt.ylabel("Improvement in mean |error| vs. baseline")
        #     plt.title(f"SF closeness to GT vs. No-Examples Baseline — {qid}")
        #     plt.grid(axis="y", linestyle="--", alpha=0.5)
        #     plt.tight_layout()

        #     os.makedirs("initial_icl_tests_plots", exist_ok=True)
        #     safe_qid = qid.replace("/", "_").replace("\\", "_")
        #     out_bar = f"initial_icl_tests_plots/{safe_qid}__improvement_bar.png"
        #     plt.savefig(out_bar, dpi=300, bbox_inches="tight")
        #     plt.close()
        #     print(f"[saved] {out_bar}")



# --- 2nd plot (GLOBAL): per-SF improvement aggregated across all questions ---
# improvement := mean(|baseline−GT|) − mean(|SF−GT|), positive => SF closer to GT than baseline

    def bootstrap_sd_for_sf(pairs, B=1000):
        """
        pairs: list of (ae_sf, ae_base) arrays across questions for one SF.
        Two-stage bootstrap:
        1) resample the question-pairs with replacement,
        2) for each sampled pair, resample within-array to account for forecast sampling variability,
        then take the mean improvement across the resampled set.
        Returns (mean_improvement, sd_improvement).
        """
        if not pairs:
            return np.nan, np.nan

        # Point estimate (no resampling): average over questions of mean(ae_base) - mean(ae_sf)
        point_imps = [np.mean(ae_b) - np.mean(ae_s) for (ae_s, ae_b) in pairs]
        point_est = float(np.mean(point_imps))

        n = len(pairs)
        draws = np.empty(B, dtype=float)
        for b in range(B):
            # resample question-pairs with replacement
            idxs = rng.integers(0, n, n)
            imps = []
            for i in idxs:
                ae_s, ae_b = pairs[i]
                # resample within-array to capture forecast-sample uncertainty
                s_s = ae_s[rng.integers(0, len(ae_s), len(ae_s))]
                s_b = ae_b[rng.integers(0, len(ae_b), len(ae_b))]
                imps.append(float(np.mean(s_b) - np.mean(s_s)))
            draws[b] = float(np.mean(imps)) if imps else np.nan

        sd = float(np.nanstd(draws, ddof=1))
        return point_est, sd

    sf_ids = []
    mean_improvements = []
    sd_improvements = []

    for sfid, pairs in sf_aggregate.items():
        m, s = bootstrap_sd_for_sf(pairs, B=1000)
        if not np.isnan(m):
            sf_ids.append(sfid)
            mean_improvements.append(m)
            sd_improvements.append(s)

    if not sf_ids:
        print("[skip] No aggregated SF improvements to plot.")
    else:
        # Sort bars by improvement (optional, nicer to read)
        order = np.argsort(mean_improvements)[::-1]
        sf_ids = [sf_ids[i] for i in order]
        mean_improvements = [mean_improvements[i] for i in order]
        sd_improvements = [sd_improvements[i] for i in order]

        plt.figure(figsize=(14, 6))
        x = np.arange(len(sf_ids))
        bars = plt.bar(x, mean_improvements)

        # SD as thick vertical line on each bar
        plt.errorbar(
            x, mean_improvements, yerr=sd_improvements,
            fmt="none", ecolor="black", elinewidth=3, capsize=6,
        )

        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.xticks(x, sf_ids, rotation=45, ha="right")
        plt.ylabel("Improvement in mean |error| vs. baseline (↑ better)")
        plt.title("Superforecasters: Closeness to GT vs No-Examples Baseline — Aggregated Across Questions")

        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        os.makedirs("initial_icl_tests_plots", exist_ok=True)
        out_bar = "initial_icl_tests_plots/ALL__sf_improvement_bar.png"
        plt.savefig(out_bar, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[saved] {out_bar}")

    # --- 3rd plot (GLOBAL): per-question improvement aggregated across SFIDs ---
    def bootstrap_sd_for_question(pairs, B=1000):
        """
        pairs: list of (ae_sf, ae_base) arrays across SFIDs for one question.
        Two-stage bootstrap:
        1) resample SFIDs with replacement,
        2) within each selected pair, resample forecast samples to capture sampling variability,
        then average the improvement across selected SFIDs.
        Returns (mean_improvement, sd_improvement).
        """
        if not pairs:
            return np.nan, np.nan

        # Point estimate (no resampling): average over SFIDs of mean(ae_base) - mean(ae_sf)
        point_imps = [np.mean(ae_b) - np.mean(ae_s) for (ae_s, ae_b) in pairs]
        point_est = float(np.mean(point_imps))

        m = len(pairs)
        draws = np.empty(B, dtype=float)
        for b in range(B):
            idxs = rng.integers(0, m, m)  # resample SFIDs
            imps = []
            for i in idxs:
                ae_s, ae_b = pairs[i]
                s_s = ae_s[rng.integers(0, len(ae_s), len(ae_s))]  # resample SF samples
                s_b = ae_b[rng.integers(0, len(ae_b), len(ae_b))]  # resample baseline samples
                imps.append(float(np.mean(s_b) - np.mean(s_s)))
            draws[b] = float(np.mean(imps)) if imps else np.nan

        sd = float(np.nanstd(draws, ddof=1))
        return point_est, sd

    qids = []
    q_labels = []
    mean_improvements = []
    sd_improvements = []

    for qid, pairs in q_aggregate.items():
        m, s = bootstrap_sd_for_question(pairs, B=1000)
        if not np.isnan(m):
            qids.append(qid)
            topic = loader.get_topic(qid)
            q_labels.append(f"{qid[:4]}: {topic}")
            mean_improvements.append(m)
            sd_improvements.append(s)

    if qids:
        # optional: sort by improvement
        order = np.argsort(mean_improvements)[::-1]
        qids = [qids[i] for i in order]
        q_labels = [q_labels[i] for i in order]
        mean_improvements = [mean_improvements[i] for i in order]
        sd_improvements = [sd_improvements[i] for i in order]

        plt.figure(figsize=(16, 6))
        x = np.arange(len(qids))
        plt.bar(x, mean_improvements)

        # SD as thick vertical line on each bar
        plt.errorbar(
            x, mean_improvements, yerr=sd_improvements,
            fmt="none", ecolor="black", elinewidth=3, capsize=6,
        )

        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.xticks(x, q_labels, rotation=45, ha="right")
        plt.ylabel("Improvement in mean |error| vs. baseline (↑ better)")
        plt.title("Per-Question: SF Emulation vs No-Examples Baseline — Aggregated over SFIDs")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        os.makedirs("initial_icl_tests_plots", exist_ok=True)
        out_bar = "initial_icl_tests_plots/ALL__question_improvement_bar.png"
        plt.savefig(out_bar, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[saved] {out_bar}")
    else:
        print("[skip] No per-question improvements to plot.")
