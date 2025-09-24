import asyncio
import copy
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

from dataset.dataloader import ForecastDataLoader

from utils.llm_config import get_llm_from_config
from utils.sampling import sample_questions
import json

from runners.icl_initial_forecasts import (
    run_all_forecasts_with_examples,
    run_all_forecasts_baseline,
    run_all_forecasts_aggregated_examples,
    run_all_forecasts_shared_examples,
)

from utils.config_types import RootConfig

from utils.convert_pickles_to_json import convert_pkl_to_json


def generate_initial_forecasts_for_questions(
    questions,
    initial_forecasts_path,
    config: RootConfig,
    selected_resolution_date,
    with_examples=True,
):
    """Generate initial forecasts for the given questions and save them to the specified path.

    Respects `with_examples` to generate either ICL (with examples) or baseline (no examples) forecasts.
    """
    os.makedirs(initial_forecasts_path, exist_ok=True)
    forecast_type = "with_examples" if with_examples else "no_examples"
    initial_config = copy.deepcopy(config)
    initial_config.experiment.seed = 42

    print(
        f"🔧 Generating initial forecasts for {len(questions)} questions using fixed seed 42 ({forecast_type})..."
    )

    loader = ForecastDataLoader()
    llm = get_llm_from_config(initial_config, role="expert")

    for question in questions:
        print(
            f"  📊 Generating initial {forecast_type} forecasts for question {question.id[:8]}..."
        )

        try:
            json_filename = f"collected_fcasts_{forecast_type}_{selected_resolution_date}_{question.id}.json"
            json_path = os.path.join(initial_forecasts_path, json_filename)

            pickle_filename = f"collected_fcasts_{forecast_type}_{selected_resolution_date}_{question.id}.pkl"
            pickle_path = os.path.join(initial_forecasts_path, pickle_filename)

            if os.path.exists(pickle_path):
                print(
                    f"Pickle for question {question.id} ({forecast_type}) already exists, converting to json."
                )
                convert_pkl_to_json(pickle_path, json_path)
                continue

            if os.path.exists(json_path):
                print(
                    f"JSON for question {question.id} ({forecast_type}) already exists, skipping."
                )
                continue

            if with_examples:
                results = asyncio.run(
                    run_all_forecasts_with_examples(
                        [question],
                        loader=loader,
                        selected_resolution_date=selected_resolution_date,
                        config=initial_config,
                        llm=llm,
                    )
                )
            else:
                results = asyncio.run(
                    run_all_forecasts_baseline(
                        [question],
                        selected_resolution_date=selected_resolution_date,
                        config=initial_config,
                    )
                )

            with open(json_path, "w") as f:
                json.dump(results, f)
            print(
                f"  ✅ Saved initial forecasts for question {question.id[:8]} to {json_filename}"
            )

        except Exception as e:
            print(
                f"  ❌ Failed to generate initial forecasts for question {question.id[:8]}: {e}"
            )
            continue

    print(f"✅ Initial forecasts generation completed in {initial_forecasts_path}")


def load_pickled_forecasts(
    initial_forecasts_path: str, selected_resolution_date: str, loader
) -> Tuple[List, Dict, Dict]:
    """Load pickled forecasts from the specified directory."""
    pkl_files = [
        f
        for f in os.listdir(f"{initial_forecasts_path}/")
        if f.startswith("collected_fcasts_with_examples")
        and f.endswith(".pkl")
        and f"{selected_resolution_date}" in f
    ]

    loaded_llmcasts = {}
    for fname in pkl_files:
        qid = fname[
            len(f"collected_fcasts_with_examples_{selected_resolution_date}_") : -len(
                ".pkl"
            )
        ]
        with open(f"{initial_forecasts_path}/{fname}", "rb") as f:
            loaded_llmcasts[qid] = [q for q in pickle.load(f)]

    questions = []
    for qid, payloads in loaded_llmcasts.items():
        if payloads:
            question_obj = loader.get_question(qid)
            if question_obj:
                questions.append(question_obj)
            else:
                print(f"Warning: Could not find Question object for ID {qid}")

    llmcasts_by_qid_sfid = defaultdict(lambda: defaultdict(list))
    for qid, payloads in loaded_llmcasts.items():
        for i, p in enumerate(payloads):
            if isinstance(p, dict):
                sfid = p.get("subject_id")
                if sfid is not None:
                    llmcasts_by_qid_sfid[qid][sfid].append(
                        {
                            "forecast": p.get("forecasts", []),
                            "full_conversation": p.get("full_conversation", []),
                            "examples_used": p.get("examples_used", []),
                        }
                    )

    print(f"Loaded forecasts for {len(llmcasts_by_qid_sfid)} questions with experts")

    example_pairs_by_qid_sfid = defaultdict(lambda: defaultdict(list))
    for qid, payloads in loaded_llmcasts.items():
        for p in payloads:
            sfid = p.get("subject_id")
            if sfid is not None:
                example_pairs = p.get("examples_used", [])
                example_pairs_by_qid_sfid[qid][sfid].append(example_pairs)

    llmcasts_by_qid_sfid = {
        qid: dict(sfid_map) for qid, sfid_map in llmcasts_by_qid_sfid.items()
    }
    example_pairs_by_qid_sfid = {
        qid: dict(sfid_map) for qid, sfid_map in example_pairs_by_qid_sfid.items()
    }

    return questions, llmcasts_by_qid_sfid, example_pairs_by_qid_sfid


def load_forecast_jsons(
    initial_forecasts_path: str,
    selected_resolution_date: str,
    loader: ForecastDataLoader,
) -> Tuple[Dict, Dict]:
    """Load JSON forecast files from the specified directory."""

    json_files = [
        f
        for f in os.listdir(initial_forecasts_path)
        if f.startswith("collected_fcasts")
        and f.endswith(".json")
        and f"{selected_resolution_date}" in f
    ]

    # Split json files into with_examples and no_examples
    aggregated_files = [
        f
        for f in json_files
        if f.startswith("collected_fcasts_with_examples_aggregated_")
        and f"{selected_resolution_date}" in f
    ]
    shared_files = [
        f
        for f in json_files
        if f.startswith("collected_fcasts_with_examples_shared_")
        and f"{selected_resolution_date}" in f
    ]
    with_examples_per_sf_files = [
        f
        for f in json_files
        if f.startswith("collected_fcasts_with_examples_")
        and not f.startswith("collected_fcasts_with_examples_aggregated_")
        and f"{selected_resolution_date}" in f
    ]
    no_examples_files = [
        f
        for f in json_files
        if f.startswith("collected_fcasts_no_examples")
        and f"{selected_resolution_date}" in f
    ]

    # Load with_examples forecasts
    loaded_fcasts_with_examples = {}
    for fname in with_examples_per_sf_files:
        qid = fname[
            len(f"collected_fcasts_with_examples_{selected_resolution_date}_") : -len(
                ".json"
            )
        ]
        with open(os.path.join(initial_forecasts_path, fname), "r") as f:
            loaded_fcasts_with_examples.setdefault(qid, [])
            loaded_fcasts_with_examples[qid].extend([q for q in json.load(f)])
    for fname in aggregated_files:
        qid = fname[
            len(
                f"collected_fcasts_with_examples_aggregated_{selected_resolution_date}_"
            ) : -len(".json")
        ]
        with open(os.path.join(initial_forecasts_path, fname), "r") as f:
            loaded_fcasts_with_examples.setdefault(qid, [])
            loaded_fcasts_with_examples[qid].extend([q for q in json.load(f)])
    for fname in shared_files:
        qid = fname[
            len(
                f"collected_fcasts_with_examples_shared_{selected_resolution_date}_"
            ) : -len(".json")
        ]
        with open(os.path.join(initial_forecasts_path, fname), "r") as f:
            loaded_fcasts_with_examples.setdefault(qid, [])
            loaded_fcasts_with_examples[qid].extend([q for q in json.load(f)])

    # replace the examples_used, which current contains question ids, with a list of (Question, Forecast) tuples
    # by looking up the Question object from the loader
    for qid, forecasts in loaded_fcasts_with_examples.items():
        for forecast_entry in forecasts:
            # Prefer reconstructing exact (question, forecast) using stored pairs
            if (
                "examples_used_pairs" in forecast_entry
                and forecast_entry["examples_used_pairs"]
            ):
                pairs = []
                for pair in forecast_entry["examples_used_pairs"]:
                    ex_qid = pair.get("question_id")
                    ex_uid = pair.get("user_id")
                    if ex_qid is None or ex_uid is None:
                        continue
                    q_obj = loader.get_question(ex_qid)
                    # Find the exact forecast object for that question+user
                    forecasts_found = loader.get_super_forecasts(
                        question_id=ex_qid,
                        user_id=ex_uid,
                        resolution_date=selected_resolution_date,
                    )
                    f_obj = forecasts_found[0] if forecasts_found else None
                    if q_obj and f_obj:
                        pairs.append((q_obj, f_obj))
                if pairs:
                    forecast_entry["examples_used"] = pairs
                    continue

            # Backward-compatible path: attach the same SF forecast to all example questions
            superforecaster_id = forecast_entry.get("subject_id")
            try:
                sf_list = loader.get_super_forecasts(
                    question_id=qid,
                    user_id=superforecaster_id,
                    resolution_date=selected_resolution_date,
                )
                superforecast = sf_list[0] if sf_list else None
            except Exception:
                superforecast = None
            if superforecast is None:
                # If we cannot resolve a matching superforecast (e.g. aggregated), keep IDs
                continue
            if "examples_used" in forecast_entry:
                example_ids = forecast_entry["examples_used"]
                example_tuples = []
                for ex_id in example_ids:
                    question_obj = loader.get_question(ex_id)
                    if question_obj:
                        example_tuples.append((question_obj, superforecast))
                    else:
                        print(
                            f"Warning: Could not find Question object for ID {ex_id} used in examples for forecast {qid} by {superforecaster_id}"
                        )
                forecast_entry["examples_used"] = example_tuples

    # Load no_examples forecasts
    loaded_fcasts_no_examples = {}
    for fname in no_examples_files:
        qid = fname[
            len(f"collected_fcasts_no_examples_{selected_resolution_date}_") : -len(
                ".json"
            )
        ]
        with open(os.path.join(initial_forecasts_path, fname), "r") as f:
            loaded_fcasts_no_examples[qid] = [q for q in json.load(f)]

    return loaded_fcasts_with_examples, loaded_fcasts_no_examples


async def load_forecasts(config: RootConfig, loader: ForecastDataLoader, llm=None):
    """Load initial forecasts based on configuration."""

    selected_resolution_date = config.data.resolution_date
    base_initial_dir = config.experiment.initial_forecasts_dir
    seed = config.experiment.seed or None
    seeded_initial_dir = (
        os.path.join(base_initial_dir, f"seed_{seed}")
        if seed is not None
        else base_initial_dir
    )
    initial_forecasts_path = seeded_initial_dir
    os.makedirs(initial_forecasts_path, exist_ok=True)

    # Determine whether to use with_examples or no_examples forecasts
    rif = config.experiment.reuse_initial_forecasts
    with_examples_flag = (
        rif.with_examples
        if rif.with_examples is not None
        else (config.initial_forecasts or {}).get("with_examples")
    )
    if with_examples_flag is None:
        with_examples_flag = True  # default to with_examples for backward compatibility
    # Examples mode (only relevant when with_examples=True)
    examples_mode = (config.initial_forecasts or {}).get("examples_mode", "per_sf") if with_examples_flag else "none"
    if examples_mode not in ["per_sf", "aggregated", "shared", "none"]:
        examples_mode = "per_sf"
    forecast_type = "with_examples" if with_examples_flag else "no_examples"
    # Filename prefix (distinct for aggregated)
    if with_examples_flag and examples_mode == "aggregated":
        fname_prefix = "collected_fcasts_with_examples_aggregated"
    elif with_examples_flag and examples_mode == "shared":
        fname_prefix = "collected_fcasts_with_examples_shared"
    elif with_examples_flag:
        fname_prefix = "collected_fcasts_with_examples"
    else:
        fname_prefix = "collected_fcasts_no_examples"

    # Get and sample questions
    questions_with_topic = loader.get_questions_with_topics()
    print(f"Total questions available: {len(questions_with_topic)}")
    sampled_questions = sample_questions(config, questions_with_topic, loader)

    # Generate initial forecasts if not reusing
    if config.experiment.reuse_initial_forecasts.enabled:
        print(
            f"📁 Reusing initial forecasts [{forecast_type}] (seeded dir: {initial_forecasts_path})"
        )
        os.makedirs(initial_forecasts_path, exist_ok=True)
        for q in sampled_questions:
            filename = f"{fname_prefix}_{selected_resolution_date}_{q.id}.json"
            json_path_seeded = os.path.join(initial_forecasts_path, filename)
            if os.path.exists(json_path_seeded):
                continue
            # If not in seeded dir, try base dir and copy over
            json_path_base = os.path.join(base_initial_dir, filename)
            if os.path.exists(json_path_base):
                try:
                    with open(json_path_base, "r") as rf:
                        data = json.load(rf)
                    with open(json_path_seeded, "w") as wf:
                        json.dump(data, wf)
                    print(
                        f"🔁 Copied initial forecasts for {q.id} from base to seed dir"
                    )
                    continue
                except Exception as e:
                    print(
                        f"⚠️  Failed to copy from base dir for {q.id}: {e}; regenerating..."
                    )
            # Generate missing
            print(
                f"Generating missing initial forecasts [{forecast_type}/{examples_mode}] for question {q.id} at {json_path_seeded}..."
            )
            init_cfg = config.initial_forecasts or {}
            if with_examples_flag and examples_mode == "aggregated":
                results = await run_all_forecasts_aggregated_examples(
                    [q],
                    loader=loader,
                    selected_resolution_date=selected_resolution_date,
                    config=config,
                    min_examples=init_cfg.get("min_examples", 1),
                    max_examples=init_cfg.get("max_examples", None),
                    concurrency=init_cfg.get("concurrency", 5),
                    timeout_s=init_cfg.get("timeout_s", 300),
                    retries=init_cfg.get("retries", 5),
                    base_backoff_s=init_cfg.get("base_backoff_s", 10),
                    n_samples=init_cfg.get("n_samples", 1),
                )
            elif with_examples_flag and examples_mode == "shared":
                results = await run_all_forecasts_shared_examples(
                    [q],
                    loader=loader,
                    selected_resolution_date=selected_resolution_date,
                    config=config,
                    min_examples=init_cfg.get("min_examples", 1),
                    max_examples=init_cfg.get("max_examples", None),
                    concurrency=init_cfg.get("concurrency", 5),
                    timeout_s=init_cfg.get("timeout_s", 300),
                    retries=init_cfg.get("retries", 5),
                    base_backoff_s=init_cfg.get("base_backoff_s", 10),
                    n_samples=init_cfg.get("n_samples", 1),
                    n_experts=(config.delphi or {}).get("n_experts", 1),
                )
            elif with_examples_flag:
                results = await run_all_forecasts_with_examples(
                    [q],
                    loader=loader,
                    selected_resolution_date=selected_resolution_date,
                    config=config,
                    llm=llm,
                    min_examples=init_cfg.get("min_examples", 1),
                    max_examples=init_cfg.get("max_examples", 5),
                    concurrency=init_cfg.get("concurrency", 5),
                    timeout_s=init_cfg.get("timeout_s", 300),
                    retries=init_cfg.get("retries", 5),
                    base_backoff_s=init_cfg.get("base_backoff_s", 10),
                    n_samples=init_cfg.get("n_samples", 1),
                )
            else:
                results = await run_all_forecasts_baseline(
                    [q],
                    selected_resolution_date=selected_resolution_date,
                    concurrency=init_cfg.get("concurrency", 5),
                    timeout_s=init_cfg.get("timeout_s", 300),
                    retries=init_cfg.get("retries", 5),
                    base_backoff_s=init_cfg.get("base_backoff_s", 10),
                    n_samples=init_cfg.get("n_samples", 1),
                    config=config,
                )
            with open(json_path_seeded, "w") as f:
                json.dump(results, f)
                print(
                    f"Saved initial forecasts for question {q.id} at {json_path_seeded}"
                )
    else:
        print(
            f"📁 Generating initial forecasts [{forecast_type}/{examples_mode}] in: {initial_forecasts_path}"
        )
        print(" WARNING: This will overwrite any existing forecasts in this directory!")
        os.makedirs(initial_forecasts_path, exist_ok=True)
        init_cfg = config.initial_forecasts or {}
        for q in sampled_questions:
            json_path = f"{initial_forecasts_path}/{fname_prefix}_{selected_resolution_date}_{q.id}.json"
            print(
                f"Collecting {forecast_type}/{examples_mode} forecasts for question {q.id}..."
            )
            if with_examples_flag and examples_mode == "aggregated":
                results = await run_all_forecasts_aggregated_examples(
                    [q],
                    loader=loader,
                    selected_resolution_date=selected_resolution_date,
                    config=config,
                    min_examples=init_cfg.get("min_examples", 1),
                    max_examples=init_cfg.get("max_examples", None),
                    concurrency=init_cfg.get("concurrency", 5),
                    timeout_s=init_cfg.get("timeout_s", 300),
                    retries=init_cfg.get("retries", 5),
                    base_backoff_s=init_cfg.get("base_backoff_s", 10),
                    n_samples=init_cfg.get("n_samples", 1),
                )
            elif with_examples_flag and examples_mode == "shared":
                results = await run_all_forecasts_shared_examples(
                    [q],
                    loader=loader,
                    selected_resolution_date=selected_resolution_date,
                    config=config,
                    min_examples=init_cfg.get("min_examples", 1),
                    max_examples=init_cfg.get("max_examples", None),
                    concurrency=init_cfg.get("concurrency", 5),
                    timeout_s=init_cfg.get("timeout_s", 300),
                    retries=init_cfg.get("retries", 5),
                    base_backoff_s=init_cfg.get("base_backoff_s", 10),
                    n_samples=init_cfg.get("n_samples", 1),
                    n_experts=(config.delphi or {}).get("n_experts", 1),
                )
            elif with_examples_flag:
                results = await run_all_forecasts_with_examples(
                    [q],
                    loader=loader,
                    selected_resolution_date=selected_resolution_date,
                    config=config,
                    llm=llm,
                    min_examples=init_cfg.get("min_examples", 1),
                    max_examples=init_cfg.get("max_examples", 5),
                    concurrency=init_cfg.get("concurrency", 5),
                    timeout_s=init_cfg.get("timeout_s", 300),
                    retries=init_cfg.get("retries", 5),
                    base_backoff_s=init_cfg.get("base_backoff_s", 10),
                    n_samples=init_cfg.get("n_samples", 1),
                )
            else:
                results = await run_all_forecasts_baseline(
                    [q],
                    selected_resolution_date=selected_resolution_date,
                    concurrency=init_cfg.get("concurrency", 5),
                    timeout_s=init_cfg.get("timeout_s", 300),
                    retries=init_cfg.get("retries", 5),
                    base_backoff_s=init_cfg.get("base_backoff_s", 10),
                    n_samples=init_cfg.get("n_samples", 1),
                    config=config,
                )
            with open(json_path, "w") as f:
                json.dump(results, f)

    # Load and return the forecasts
    loaded_fcasts_with_examples, loaded_fcasts_no_examples = load_forecast_jsons(
        initial_forecasts_path, selected_resolution_date, loader
    )
    # Fallback: if none loaded from seed dir, try base dir (backward compatibility)
    if with_examples_flag and not loaded_fcasts_with_examples:
        try:
            print(
                f"ℹ️  No forecasts found in seeded dir; trying base dir {base_initial_dir}"
            )
            loaded_fcasts_with_examples, loaded_fcasts_no_examples = (
                load_forecast_jsons(base_initial_dir, selected_resolution_date, loader)
            )
        except Exception:
            pass
    if (not with_examples_flag) and (not loaded_fcasts_no_examples):
        try:
            print(
                f"ℹ️  No forecasts found in seeded dir; trying base dir {base_initial_dir}"
            )
            loaded_fcasts_with_examples, loaded_fcasts_no_examples = (
                load_forecast_jsons(base_initial_dir, selected_resolution_date, loader)
            )
        except Exception:
            pass

    # Build nested mapping by qid and sfid depending on forecast type
    llmcasts_by_qid_sfid = defaultdict(lambda: defaultdict(list))
    if with_examples_flag:
        # Filter payloads by examples_mode
        filtered_with_examples = {}
        for qid, payloads in loaded_fcasts_with_examples.items():
            if examples_mode == "aggregated":
                filtered = [
                    p
                    for p in payloads
                    if (
                        p.get("subject_type") == "aggregated"
                        or p.get("subject_id") in ("agg", "aggregated")
                    )
                ]
            else:
                filtered = [
                    p
                    for p in payloads
                    if p.get("subject_type") in (None, "superforecaster")
                    and p.get("subject_id") not in ("agg", "aggregated")
                ]
            if filtered:
                filtered_with_examples[qid] = filtered

        for qid, payloads in filtered_with_examples.items():
            for p in payloads:
                sfid = p.get("subject_id")
                if sfid is not None:
                    llmcasts_by_qid_sfid[qid][sfid].append(
                        {
                            "forecast": p.get("forecasts", []),
                            "full_conversation": p.get("full_conversation", []),
                            "examples_used": p.get("examples_used", []),
                        }
                    )
        questions = [q for q in sampled_questions if q.id in filtered_with_examples]

        example_pairs_by_qid_sfid = defaultdict(lambda: defaultdict(list))
        for qid, payloads in filtered_with_examples.items():
            for p in payloads:
                sfid = p.get("subject_id")
                if sfid is not None:
                    example_pairs = p.get("examples_used", [])
                    example_pairs_by_qid_sfid[qid][sfid].append(example_pairs)
    else:
        # Baseline: create multiple synthetic experts per question from independent samples
        desired_n_experts = (config.delphi or {}).get("n_experts", 1)
        for qid, payloads in loaded_fcasts_no_examples.items():
            for p in payloads:
                full_conv = p.get("full_conversation", []) or []
                initial_message = (
                    full_conv[0]
                    if full_conv
                    else {
                        "role": "user",
                        "content": "Provide a probability. FINAL PROBABILITY: 0.5",
                    }
                )

                # Collect assistant messages and align with forecasts list
                assistant_msgs = [
                    m
                    for m in full_conv
                    if isinstance(m, dict) and m.get("role") == "assistant"
                ]
                forecast_list = p.get("forecasts", []) or []
                available = max(len(assistant_msgs), len(forecast_list))
                if available == 0:
                    # Nothing usable; skip
                    continue
                k = min(desired_n_experts, available)

                for i in range(k):
                    # Build a minimal conversation: initial + one assistant sample
                    if i < len(assistant_msgs):
                        sample_msg = assistant_msgs[i]
                    else:
                        prob_val = forecast_list[i] if i < len(forecast_list) else 0.5
                        sample_msg = {
                            "role": "assistant",
                            "content": f"FINAL PROBABILITY: {prob_val}",
                        }

                    conv_i = [initial_message, sample_msg]
                    prob_i = [forecast_list[i]] if i < len(forecast_list) else []

                    llmcasts_by_qid_sfid[qid][f"baseline_{i}"].append(
                        {
                            "forecast": prob_i,
                            "full_conversation": conv_i,
                            "examples_used": [],
                        }
                    )

        questions = [q for q in sampled_questions if q.id in loaded_fcasts_no_examples]
        example_pairs_by_qid_sfid = defaultdict(lambda: defaultdict(list))
        for qid, sf_map in llmcasts_by_qid_sfid.items():
            for sfid in sf_map.keys():
                example_pairs_by_qid_sfid[qid][sfid].append([])

    return questions, llmcasts_by_qid_sfid, example_pairs_by_qid_sfid
