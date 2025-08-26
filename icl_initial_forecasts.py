from delphi import Expert
from models import LLMFactory, LLMProvider, LLMModel
from dataset.dataloader import Question, Forecast, Resolution, ForecastDataLoader

from eval import load_config
import os
from collections import defaultdict

import random
import copy
import asyncio
import time
import pickle
import json
import numpy as np
import re
from enum import Enum
from dataclasses import dataclass
import asyncio
import copy
import re

from dotenv import load_dotenv
load_dotenv()

import psutil

from typing import Callable, List, Tuple, Optional, Dict, Any



# def _is_debugpy_running(port=5679):
#     """Check if debugpy is already listening on the given port."""
#     for proc in psutil.process_iter(attrs=["cmdline"]):
#         try:
#             cmdline = proc.info["cmdline"]
#             if cmdline and any("debugpy" in arg for arg in cmdline) and str(port) in " ".join(cmdline):
#                 return True
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             continue
#     return False

# if not _is_debugpy_running():
#     import debugpy
#     print("Waiting for debugger attach...")
#     debugpy.listen(5679)
#     debugpy.wait_for_client()
#     print("Debugger attached.")

# Set all random seeds for reproducibility

# forecast_pattern = re.compile(r'final\s*probability\s*:\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)


_NUMBER_RE = re.compile(r"""
    final \s* probability
    \s*[:\-]?\s*
    (?:`|\*{1,2}|")?      # optional fence/formatting
    (                     # capture the value
        (?:0(?:\.\d+)?|1(?:\.0+)?)   # strict 0–1 decimal
        | \d{1,3}\s?%                # or a percentage (fallback)
    )
""", re.IGNORECASE | re.VERBOSE)



import openai
import textwrap
import matplotlib.pyplot as plt

config_path = "/home/williaar/projects/delphi/configs/test_configs/icl_delphi_test_set_o3.yml"
config = load_config(config_path)

resolutions_path = "/home/williaar/projects/delphi/dataset/datasets/resolution_sets/2024-07-21_resolution_set.json"


SEED = config.get("seed", 42)
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

n_samples = config['n_samples']

provider = config['model']['provider']
model = config['model']['name']
personalized_system_prompt = config['model']['system_prompt']

openai_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

llm = LLMFactory.create_llm(provider, model, system_prompt=personalized_system_prompt)

# get questions that have a topic
loader = ForecastDataLoader()
questions_with_topic = loader.get_questions_with_topics()

forecast_due_date = config.get("forecast_due_date", "2024-07-21")
selected_resolution_date = config.get("selected_resolution_date", "2025-07-21")
initial_forecasts_path = config.get("initial_forecasts_path")
output_plots_dir = config.get("output_plots_dir", 'initial_icl_tests_plots_flexible_retry')

if not os.path.exists(initial_forecasts_path):
    os.makedirs(initial_forecasts_path)

if not os.path.exists(output_plots_dir):
    os.makedirs(output_plots_dir)




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
    """One TaskSpec per (question, sf_id) with that SF's example pairs."""
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
    Return {question_id: {sf_id: examples_used}}, where each example pair is
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

            examples_used = []
            for f in forecasts:
                if getattr(f, "id", None) == q.id:
                    continue  # skip the target question itself
                q_obj = loader.get_question(f.id)          # ← direct lookup
                examples_used.append((q_obj, f))
                if len(examples_used) >= max_examples:
                    break

            example_forecasts_dict[q.id][sf_id] = examples_used

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

def _messages_match_pattern(messages, pattern: re.Pattern) -> bool:
    try:
        for m in messages[1:]:
            # tolerate both dict-like and object-like messages
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content", "")
            if content and isinstance(content, str) and pattern.search(content):
                return True
    except Exception:
        # ultra-defensive: if message shape surprises us, stringify
        text = "\n".join(str(m) for m in messages)
        return bool(pattern.search(text))
    return False

async def _run_specs(
    specs: List[TaskSpec],
    *,
    selected_resolution_date: str,
    concurrency: int = 5,
    timeout_s: int = 300,
    retries: int = 5,
    base_backoff_s: int = 10,
    n_samples: int = 5,
    max_retries: int = 20,
    batch_size: int = 1,
    pattern: re.Pattern = _NUMBER_RE
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
            expert = Expert(llm, user_profile=None, config=config.get("model", {}))

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
                # NOTE: each call mutates expert.conversation_manager.messages to the latest conversation.
                # We snapshot messages right after to evaluate the pattern against the *per-sample* transcript.
                result = await asyncio.wait_for(_retrying(), timeout=timeout_s)
                # snapshot messages for this sample
                messages_snapshot = copy.deepcopy(expert.conversation_manager.messages)
                return result, messages_snapshot

            # ---- NEW adaptive collection logic ----
            all_forecasts: List[Any] = []
            accepted_indices: List[int] = []
            messages_per_sample: List[Any] = []

            attempts = 0
            while len(accepted_indices) < n_samples and attempts < max_retries:
                to_launch = min(batch_size, max_retries - attempts)
                batch = await asyncio.gather(*(_sample_once() for _ in range(to_launch)))
                for (res, msgs) in batch:
                    idx = len(all_forecasts)
                    all_forecasts.append(res)
                    messages_per_sample.append(msgs)
                    if _messages_match_pattern(msgs, pattern):
                        accepted_indices.append(idx)
                attempts += to_launch

            # Report whether we reached the max attempts or collected all required samples
            if len(accepted_indices) < n_samples:
                print(f"[{spec.question.id}] Reached max attempts ({attempts}) but only collected {len(accepted_indices)}/{n_samples} accepted samples.")
            else:
                print(f"[{spec.question.id}] Collected required {n_samples} accepted samples in {attempts} attempts (total calls={len(all_forecasts)}).")

            return_dict = {
                "question_id": spec.question.id,
                "subject_type": spec.subject_type.value,
                "subject_id": spec.subject_id,
                "date": selected_resolution_date,
                "mode": "with_examples" if spec.examples else "no_examples",
                "forecasts": [f for f in all_forecasts if f is not None],
                "full_conversation": messages_per_sample,
                "examples_used": [q.id for q, _ in spec.examples] if spec.examples else []
            }

            return return_dict

    return await asyncio.gather(*(_call_one(s) for s in specs), return_exceptions=True)

async def run_all_forecasts_with_examples(
    sampled_questions,
    *,
    loader=loader,
    selected_resolution_date: str = selected_resolution_date,
    min_examples: int = 1,
    max_examples: int = 5,
    concurrency: int = 5,
    timeout_s: int = 300,
    retries: int = 5,
    base_backoff_s: int = 10,
    n_samples: int = n_samples,
):
    """
    PRODUCES: one record per (question, superforecaster).
    """
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
    )

async def run_all_forecasts_baseline(
    sampled_questions,
    *,
    selected_resolution_date: str = selected_resolution_date,
    concurrency: int = 5,
    timeout_s: int = 300,
    retries: int = 10,
    base_backoff_s: int = 10,
    n_samples: int = n_samples,
):
    """
    PRODUCES: one record per question (no SF id).
    """
    specs = build_specs_baseline(sampled_questions)
    return await _run_specs(
        specs,
        selected_resolution_date=selected_resolution_date,
        concurrency=concurrency,
        timeout_s=timeout_s,
        retries=retries,
        base_backoff_s=base_backoff_s,
        n_samples=n_samples,
    )


def build_specs_with_per_question_examples(
    sampled_questions: List[Question],
    qid_to_examples: Dict[str, List[Tuple[Question, Forecast]]],
    *,
    subject_id: str = "provided_single_forecaster",
    max_examples: Optional[int] = None,
    require_nonempty: bool = True,
) -> List[TaskSpec]:
    """
    Each question gets its OWN example list from `qid_to_examples[question.id]`.
    `qid_to_examples` must map question_id -> List[(Question, Forecast)].
    """
    specs: List[TaskSpec] = []
    for q in sampled_questions:
        if q.id not in qid_to_examples:
            raise KeyError(f"No examples provided for question_id={q.id}")
        ex = qid_to_examples[q.id]
        if require_nonempty and not ex:
            raise ValueError(f"Empty examples for question_id={q.id}")
        if max_examples is not None:
            ex = ex[:max_examples]
        specs.append(
            TaskSpec(
                question=q,
                subject_type=SubjectType.SUPERFORECASTER,
                subject_id=subject_id,
                examples=ex,
            )
        )
    return specs


def build_specs_with_selector(
    sampled_questions: List[Question],
    example_selector: Callable[[Question], List[Tuple[Question, Forecast]]],
    *,
    subject_id: str = "provided_single_forecaster",
    max_examples: Optional[int] = None,
    require_nonempty: bool = True,
) -> List[TaskSpec]:
    """
    Alternative: pass a selector function to compute examples per question.
    """
    specs: List[TaskSpec] = []
    for q in sampled_questions:
        ex = example_selector(q)
        if require_nonempty and not ex:
            raise ValueError(f"Selector returned no examples for question_id={q.id}")
        if max_examples is not None:
            ex = ex[:max_examples]
        specs.append(
            TaskSpec(
                question=q,
                subject_type=SubjectType.SUPERFORECASTER,
                subject_id=subject_id,
                examples=ex,
            )
        )
    return specs


async def run_all_forecasts_single_forecaster_with_per_question_examples(
    sampled_questions: List[Question],
    qid_to_examples: Dict[str, List[Tuple[Question, Forecast]]],
    *,
    selected_resolution_date: str,
    subject_id: str = "provided_single_forecaster",
    max_examples: Optional[int] = None,
    concurrency: int = 5,
    timeout_s: int = 300,
    retries: int = 5,
    base_backoff_s: int = 10,
    n_samples: int = n_samples,
    batch_size: int = 1,
    pattern: Optional[re.Pattern] = None,
) -> List[Dict[str, Any]]:
    """
    PRODUCES: one record per question, each using its OWN example set (and the SAME forecaster id tag).
    Multi-sample acceptance/batching is delegated to _run_specs.
    """
    specs = build_specs_with_per_question_examples(
        sampled_questions,
        qid_to_examples,
        subject_id=subject_id,
        max_examples=max_examples,
        require_nonempty=True,
    )
    return await _run_specs(
        specs,
        selected_resolution_date=selected_resolution_date,
        concurrency=concurrency,
        timeout_s=timeout_s,
        retries=retries,
        base_backoff_s=base_backoff_s,
        n_samples=n_samples,
        batch_size=batch_size,
        pattern=pattern or _NUMBER_RE,
    )


__all__ = [
    "build_specs_with_per_question_examples",
    "build_specs_with_selector",
    "run_all_forecasts_single_forecaster_with_per_question_examples",
]

exclude_ids = {
        '1b12215032357c20078f36029eca8e2c67788d7834cba572d712b7d769a288ee',
        '1f7b3ef2436775c7530d7858dc141a4cf7a5692745e9f476e1bbd534d183f3f2',
        '4204aec5ff81b3d331f27141b072979d838ed95bcd0de36e887ca9a70523060a',
        '45db5d06a001a6fa62eb9b23236adab43c56970d70a833ca206fa42a57f4b7e6',
        '5713f8a61c04fa270a3a9e1791e4d9b5fa8e0d1cc1c1c232aef84d80c5b89c09',
        '61c0fb3703e68cee2439afd5c2d71522bc6649a1fa154491f58981456fa8ab68',
        '9043472375a02690dfb338bd3d11605105562e5cae9672a989961b0c5bef9b51',
        '9fa89a7d296950fe794a71be32c65e5d50930a8dbe0f9a8c780f27eec1529e60',
        'REAINTRATREARAT10Y',
        'SNA',
        'T10YIE',
        'a6df218adf1cf6a40983148234c46052bc83c7d2b8e31157dfdd6e58a9f83f5d',
        'c37effa43385e2f5a9a91bc99a278ac376fb8f10f1e11ea39fb621bfbf6e2c2f',
        'ccda7990a2565cabd7c375a036751bd3b953b8bed45d859010919cd3a84d7e78',
        'd61d058797047fb9793684b123dcf88a66f843695d9e65e9bc6df0f49ec9d936',
        'e53dabd31f71786f3b044bd12e498deee5a732a43de2d9be7468ebaced466977',
        'eda8e5b43e0db651905667586e1e72a7d5679cbb5b3ef4dd6faa6444759e2dee',
        'fa23cf1ab8ae4be34faeccb0c0453b19974158a7a1cb10657339b11a869ce089',
        'meteofrance_TEMPERATURE_celsius.07110.D',
        'meteofrance_TEMPERATURE_celsius.07190.D',
        'meteofrance_TEMPERATURE_celsius.07761.D'
    }


def sample_questions_by_topic(questions, n_per_topic=None, seed=42, exclude_ids=None):
    random.seed(seed)
    unique_topics = set(q.topic for q in questions)
    topic_to_questions = defaultdict(list)

    shuffled_questions = random.sample(questions, len(questions))

    already_run_jsons = os.listdir('/home/williaar/projects/delphi/outputs_initial_forecasts_flexible_retry')
    already_run_ids = {
        m.group(1)
        for fname in already_run_jsons
        if (m := re.search(
            r'^collected_fcasts_with_examples_2025-07-21_(.+)\.pkl$',
            fname
        ))
    }
    if already_run_ids is not None:
        shuffled_questions = [
            q for q in shuffled_questions
            if q.id not in already_run_ids
        ]

    if exclude_ids is not None:
        shuffled_questions = [q for q in shuffled_questions if q.id not in exclude_ids]

    # Remove questions that do not have a resolution on the selected date
    shuffled_questions = [
        q for q in shuffled_questions
        if loader.get_resolution(question_id=q.id, resolution_date=selected_resolution_date) is not None
    ]

    leftover_n_per_topic = [
        len([q for q in shuffled_questions if q.topic == topic])
        for topic in unique_topics
    ]

    min_leftover = min(leftover_n_per_topic)


    for topic in unique_topics:
        topic_questions = [q for q in shuffled_questions if q.topic == topic]
        if len(topic_questions) < n_per_topic:
            raise ValueError(
                f"Topic '{topic}' only has {len(topic_questions)} questions, "
                f"but {n_per_topic} are required."
            )
        topic_to_questions[topic] = topic_questions[:min_leftover]

    sampled_questions = [q for qs in topic_to_questions.values() for q in qs]
    return sampled_questions

new_sampled_questions = sample_questions_by_topic(
    questions_with_topic, n_per_topic=3, seed=42, exclude_ids=exclude_ids
)

print([q.id for q in new_sampled_questions])


if __name__ == "__main__":

    question_ids = config.get("question_ids", None)

    sampled_questions = [
        loader.get_question(qid) for qid in question_ids
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

    time.sleep(2)
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
            sfid = payload.get("subject_id", "unknown_sfid")
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
        os.makedirs(output_plots_dir, exist_ok=True)
        safe_qid = qid.replace("/", "_").replace("\\", "_")
        filename = f"{output_plots_dir}/{safe_qid}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[saved] {filename}")


        # --- accumulate per-(qid, sfid) absolute errors for cross-question aggregation ---
        for payload in sf_payloads:
            sfid = payload.get("subject_id", "unknown_sfid")
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
            sfid = payload.get("subject_id", "unknown_sfid")
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
        os.makedirs(output_plots_dir, exist_ok=True)
        out_bar = f"{output_plots_dir}/ALL__sf_improvement_bar.png"
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

        os.makedirs(output_plots_dir, exist_ok=True)
        out_bar = f"{output_plots_dir}/ALL__question_improvement_bar.png"
        plt.savefig(out_bar, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[saved] {out_bar}")
    else:
        print("[skip] No per-question improvements to plot.")
