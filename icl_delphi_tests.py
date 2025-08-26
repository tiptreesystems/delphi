from delphi import Expert, Mediator
from models import LLMFactory, LLMProvider, LLMModel
from dataset.dataloader import Question, Forecast, Resolution, ForecastDataLoader

from eval import load_config
import os
import shutil
from collections import defaultdict

import random
import copy
import asyncio
import re

from collections import defaultdict

import openai

import time
import pickle
import json

from dotenv import load_dotenv
load_dotenv()

from icl_initial_forecasts import run_all_forecasts_with_examples, sample_questions_by_topic

import numpy as np
import psutil
import sys
import argparse
import debugpy
from pathlib import Path

def _is_debugpy_running(port=5679):
    """Check if debugpy is already listening on the given port."""
    for proc in psutil.process_iter(attrs=["cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if cmdline and any("debugpy" in arg for arg in cmdline) and str(port) in " ".join(cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

if not _is_debugpy_running():

    print("Waiting for debugger attach...")
    debugpy.listen(5679)
    debugpy.wait_for_client()
    print("Debugger attached.")
# Set all random seeds for reproducibility




_NUMBER_RE = re.compile(r"""
    final \s* probability
    \s*[:\-]?\s*
    (?:`|\*{1,2}|")?      # optional fence/formatting
    (                     # capture the value
        (?:0(?:\.\d+)?|1(?:\.0+)?)   # strict 0–1 decimal
        | \d{1,3}\s?%                # or a percentage (fallback)
    )
""", re.IGNORECASE | re.VERBOSE)

def _extract_prob(text: str) -> float:
    """
    Extract the last probability mentioned in the text, preferring explicit
    'FINAL PROBABILITY:' markers but falling back to any number match.
    """
    if not text:
        return 0.5

    # Find all explicit 'FINAL PROBABILITY:' occurrences and take the last one
    matches = _NUMBER_RE.findall(text)
    if matches:
        try:
            p = float(matches[-1])
            return max(0.0, min(1.0, p))
        except ValueError:
            pass

    # Fallback: find all bare numeric probabilities and take the last one
    nums = re.findall(r'0?\.\d+|1\.0|0|1', text)
    if nums:
        try:
            p = float(nums[-1])
            return max(0.0, min(1.0, p))
        except ValueError:
            pass

    return 0.5


def load_forecasts_no_examples():
    question_ids = config.get("question_ids", None)

    sampled_questions = [
        loader.get_question(qid) for qid in question_ids
    ]

    loaded_llmcasts = load_llm_forecasts_from_pickle(sampled_questions, with_examples=False)

    # each key is a question id, each value is a list of tuples (qid, sfid, payload)
    # each payload is a dict with keys: 'question', 'forecast', 'subject_id', 'examples_used'
    # Extract the list of questions, and return these as a list
    # Also, extract the payloads into a nested dict qid, sfid
    # (1) Extract list of questions (one per qid; taken from the first payload)
    questions = []
    for qid, payloads in loaded_llmcasts.items():
        if payloads:
            qid = payloads[0].get("question_id", "")
            question = loader.get_question(qid)
            questions.append(question)

    # (2) Nest payloads by qid -> sfid -> [payloads]
    llmcasts_by_qid = defaultdict(lambda: defaultdict(list))
    for qid, payloads in loaded_llmcasts.items():
        for p in payloads:
            llmcasts_by_qid[qid].append(p)

    # Convert nested default dicts to plain dicts and return
    llmcasts_by_qid = {qid: dict(sfid_map) for qid, sfid_map in llmcasts_by_qid.items()}
    return questions, llmcasts_by_qid

def load_forecasts_with_examples():

    question_ids = config.get("question_ids", None)

    sampled_questions = [
        loader.get_question(qid) for qid in question_ids
    ]

    loaded_llmcasts = load_llm_forecasts_from_pickle(sampled_questions, with_examples=True)

    # each key is a question id, each value is a list of tuples (qid, sfid, payload)
    # each payload is a dict with keys: 'question', 'forecast', 'subject_id', 'examples_used'
    # Extract the list of questions, and return these as a list
    # Also, extract the payloads into a nested dict qid, sfid
    # (1) Extract list of questions (one per qid; taken from the first payload)
    questions = []
    for qid, payloads in loaded_llmcasts.items():
        if payloads:
            qid = payloads[0].get("question_id", "")
            question = loader.get_question(qid)
            questions.append(question)

    # (2) Nest payloads by qid -> sfid -> [payloads]
    llmcasts_by_qid_sfid = defaultdict(lambda: defaultdict(list))
    for qid, payloads in loaded_llmcasts.items():
        for p in payloads:
            sfid = p.get("subject_id")
            if sfid is not None:
                llmcasts_by_qid_sfid[qid][sfid].append(p)

    examples_used_by_qid_sfid = defaultdict(lambda: defaultdict(list))
    for qid, payloads in loaded_llmcasts.items():
        for p in payloads:
            sfid = p.get("subject_id")
            if sfid is not None:
                examples_used = p["examples_used"]
                examples_used_by_qid_sfid[qid][sfid].append(examples_used)


    # Convert nested default dicts to plain dicts and return
    llmcasts_by_qid_sfid = {qid: dict(sfid_map) for qid, sfid_map in llmcasts_by_qid_sfid.items()}
    examples_used_by_qid_sfid = {qid: dict(sfid_map) for qid, sfid_map in examples_used_by_qid_sfid.items()}
    return questions, llmcasts_by_qid_sfid, examples_used_by_qid_sfid

def load_llm_forecasts_from_pickle(sampled_questions, with_examples: bool = True):
    if with_examples:
        prefix = "collected_fcasts_with_examples"
    else:
        prefix = "collected_fcasts_no_examples"
    for q in sampled_questions:
        if os.path.exists(f'{initial_forecasts_path}/{prefix}_{selected_resolution_date}_{q.id}.pkl'):
            print(f"Pickle for question {q.id} already exists, skipping.")
            continue
        print(f"Collecting forecasts for question {q.id}...")
        results = asyncio.run(run_all_forecasts_with_examples([q]))
        with open(f'{initial_forecasts_path}/{prefix}_{selected_resolution_date}_{q.id}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"Collected forecasts for question {q.id}.")

    pkl_files = [
        f for f in os.listdir(f"{initial_forecasts_path}/")
        if f.startswith(prefix) and f.endswith(".pkl") and f"{selected_resolution_date}" in f
    ]


    loaded_llmcasts = {}
    for fname in pkl_files:
        # Extract question id between 'collected_fcasts_' and '.pkl'
        qid = fname[len(f"{prefix}_{selected_resolution_date}_"): -len(".pkl")]
        with open(f"{initial_forecasts_path}/{fname}", "rb") as f:
            loaded_llmcasts[qid] = [q for q in pickle.load(f)]
    return loaded_llmcasts

def convert_pkl_to_json(pkl_path: str, json_path: str) -> None:
    """
    Convert a pickle file containing a list of dicts into a JSON file.
    For the 'examples_used' key, only keep the Question.id values.

    Parameters
    ----------
    pkl_path : str
        Path to the pickle file.
    json_path : str
        Path to save the JSON file.
    """
    # Load pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Transform
    converted = []
    for entry in data:
        new_entry = entry.copy()
        if "examples_used" in new_entry:
            # Replace list of (Question, Forecast) tuples with list of Question.id
            new_entry["examples_used"] = [
                q.id for (q, _forecast) in new_entry["examples_used"]
            ]
        converted.append(new_entry)

    # Save as JSON
    with open(json_path, "w") as f:
        json.dump(converted, f, indent=2)



def batch_convert_pickles(input_dir: str, output_dir: str) -> None:
    """
    Convert all pickle files in a directory to JSON using convert_pkl_to_json.
    Saves outputs into a parallel directory.

    Parameters
    ----------
    input_dir : str
        Directory containing .pkl files.
    output_dir : str
        Directory to save converted .json files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for pkl_file in input_path.glob("*.pkl"):
        json_file = output_path / (pkl_file.stem + ".json")
        convert_pkl_to_json(str(pkl_file), str(json_file))
        print(f"Converted {pkl_file.name} -> {json_file.name}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Delphi ICL tests.")
    parser.add_argument("config", type=str, help="Path to the config YAML file.")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    provider = config['model']['provider']
    model = config['model']['name']
    personalized_system_prompt = config['model']['system_prompt']
    llm = LLMFactory.create_llm(provider, model, system_prompt=personalized_system_prompt)

    openai_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_key

    # get questions that have a topic
    loader = ForecastDataLoader()

    forecast_due_date = config.get("forecast_due_date", "2024-07-21")
    selected_resolution_date = config.get("selected_resolution_date", "2025-07-21")

    n_rounds = config.get("n_rounds", 3)
    n_experts = config.get("n_experts", 5)

    SEED = config.get("seed")
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)


    initial_forecasts_path = config.get("initial_forecasts_path", "outputs_initial_forecasts_flexible_retry")
    initial_forecast_path = os.path.join(initial_forecasts_path, f"{SEED}")

    output_dir = config.get("output_dir", "outputs_initial_delphi_flexible_retry")
    output_dir = os.path.join(output_dir, f"seed_{SEED}")

    # Load initial forecasts from files
    questions, llmcasts_by_qid_sfid, examples_used_by_qid_sfid = load_forecasts_with_examples()

    os.makedirs(output_dir, exist_ok=True)

    for question in questions:

        output_file = os.path.join(output_dir, f"delphi_log_with_examples_{question.id}_{selected_resolution_date}.json")
        if os.path.exists(output_file):
            print(f"Skipping {output_file} (already exists)")
            continue

        llmcasts_by_sfid = llmcasts_by_qid_sfid[question.id]
        examples_used = examples_used_by_qid_sfid.get(question.id, {})

        # Each superforecaster gets their own expert instance
        experts = {sfid: Expert(llm, config=config.get('model', {})) for sfid in llmcasts_by_sfid.keys()}

        # Populate the experts with their initial forecasts
        # We take the median of the sample forecasts for each superforecaster
        for sfid, payload in llmcasts_by_sfid.items():
            # Reajusting code for new data structures
            # pairs are just zipped (prob, full_conversation) for "forecasts" and "full_conversation", both of which are lists
            inner_payload = payload[0]
            forecasts = inner_payload['forecasts']
            full_conversations = inner_payload['full_conversation']

            forecast_convo_pairs = list(zip(forecasts, full_conversations))

            forecast_convo_pairs.sort(key=lambda x: x[0])
            median_llmcast, median_convo = forecast_convo_pairs[len(forecast_convo_pairs) // 2]

            experts[sfid].conversation_manager.add_messages(median_convo)

        # Take n_experts random superforecasters if more than n_experts
        if len(experts) > n_experts:
            selected_sfs = random.Random(SEED).sample(list(experts.keys()), n_experts)
            experts = {sfid: experts[sfid] for sfid in selected_sfs}

        print(f"Running Delphi for question {question.id} with {len(experts)} out of {len(list(llmcasts_by_sfid.keys()))} experts")


        # Instantiate the Delphi mediator
        mediator = Mediator(llm, config=config.get('model', {}))

        # Structured log across all rounds
        delphi_log = {
            "question": question.id,
            "rounds": [],     # list of { round, mediator_feedback, experts: {id: {prob, response}} }
            "histories": None # filled at end with full convo histories
        }

        # Round 0: capture initial expert responses (text + parsed prob)
        initial_expert_entries = {}
        for sfid, expert in experts.items():
            resp = expert.get_last_response()
            initial_expert_entries[sfid] = {
                "prob": _extract_prob(resp['content']),
                "response": resp,
            }

        delphi_log["rounds"].append({
            "round": 0,
            "mediator_feedback": "Initial forecasts collected.",
            "experts": initial_expert_entries,
        })

        for round_idx in range(n_rounds):
            print(f"Round {round_idx + 1} for question {question.id}")

            # mediator context + intake
            mediator.start_round(round_idx=round_idx, question=question)
            expert_messages = {sfid: {"role": "assistant", "content": entry["response"]}
                            for sfid, entry in delphi_log["rounds"][-1]["experts"].items()}
            mediator.receive_messages(expert_messages)

            # craft mediator feedback (preserves convo history)
            feedback_message = asyncio.run(mediator.generate_feedback(round_idx=round_idx))

            update_instruction = (
                "After considering the other experts' perspectives, think through your reasoning and "
                "provide your final probability estimate.\n"
                "You may reason through the problem, but you MUST end your response with:\n"
                "FINAL PROBABILITY: [your decimal number between 0 and 1]"
            )
            broadcast_msg = f"{feedback_message}\n\n{update_instruction}"

            # experts update; store numeric prob + full response
            round_expert_entries = {}
            for sfid, expert in experts.items():
                prob, response = asyncio.run(expert.get_forecast_update(broadcast_msg))
                # prob comes from your Expert; still store parsed prob defensively from the text
                round_expert_entries[sfid] = {
                    "prob": max(0.0, min(1.0, float(prob))) if isinstance(prob, (int, float)) else _extract_prob(response),
                    "response": response,
                }

            delphi_log["rounds"].append({
                "round": round_idx + 1,
                "mediator_feedback": broadcast_msg,
                "experts": round_expert_entries,
            })


        # After all rounds: capture full conversation histories
        delphi_log["histories"] = {
            "mediator": list(mediator.conversation_manager.messages),
            "experts": {sfid: list(expert.conversation_manager.messages) for sfid, expert in experts.items()},
        }

        with open(output_file, "w") as f:
            json.dump(delphi_log, f, indent=2)

        print(f"Delphi log saved to {output_file}")
