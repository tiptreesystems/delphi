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

from icl_initial_forecasts import run_all_forecasts_with_examples, sample_questions_by_topic, run_all_forecasts_single_forecaster_with_per_question_examples

import numpy as np
import psutil

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

# if not _is_debugpy_running():
#     import debugpy
#     print("Waiting for debugger attach...")
#     debugpy.listen(5679)
#     debugpy.wait_for_client()
#     print("Debugger attached.")
# # Set all random seeds for reproducibility

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

config_path = '/home/williaar/projects/delphi/configs/test_configs/icl_delphi_test_set.yml'
config = load_config(config_path)

provider = LLMProvider.OPENAI
model = LLMModel.GPT_4O_2024_05_13
personalized_system_prompt = "You are a helpful assistant with expertise in forecasting and decision-making."

openai_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

llm = LLMFactory.create_llm(provider, model, system_prompt=personalized_system_prompt)

# get questions that have a topic
loader = ForecastDataLoader()
questions_with_topic = loader.get_questions_with_topics()

forecast_due_date = "2024-07-21"
selected_resolution_date = '2025-07-21'

n_rounds = 3

previous_forecasts_path = "outputs_initial_forecasts_flexible_retry_test_set"

output_dir = 'outputs_superagent_delphi_forecasts_flexible_retry_test_set'


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
        if os.path.exists(f'{previous_forecasts_path}/{prefix}_{selected_resolution_date}_{q.id}.pkl'):
            print(f"Pickle for question {q.id} already exists, skipping.")
            continue
        print(f"Collecting forecasts for question {q.id}...")
        results = asyncio.run(run_all_forecasts_with_examples([q]))
        with open(f'{previous_forecasts_path}/{prefix}_{selected_resolution_date}_{q.id}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"Collected forecasts for question {q.id}.")

    pkl_files = [
        f for f in os.listdir(f"{previous_forecasts_path}/")
        if f.startswith(prefix) and f.endswith(".pkl") and f"{selected_resolution_date}" in f
    ]


    loaded_llmcasts = {}
    for fname in pkl_files:
        # Extract question id between 'collected_fcasts_' and '.pkl'
        qid = fname[len(f"{prefix}_{selected_resolution_date}_"): -len(".pkl")]
        with open(f"{previous_forecasts_path}/{fname}", "rb") as f:
            loaded_llmcasts[qid] = [q for q in pickle.load(f)]
    return loaded_llmcasts


def get_initial_super_forecasts(selected_resolution_date, questions, examples_flattened_per_question):

    # Try to load forecasts from file if they exist, otherwise run and save them
    forecasts_file = os.path.join(output_dir, f"superagent_initial_forecasts_{selected_resolution_date}.pkl")
    if os.path.exists(forecasts_file):
        print(f"Loading initial super forecasts from {forecasts_file}")
        with open(forecasts_file, "rb") as f:
            forecasts = pickle.load(f)
        return forecasts

    print("Running initial super forecasts...")


    forecasts = asyncio.run(run_all_forecasts_single_forecaster_with_per_question_examples(
        sampled_questions=questions,
        qid_to_examples=examples_flattened_per_question,
        selected_resolution_date=selected_resolution_date,
    ))

    with open(forecasts_file, "wb") as f:
        pickle.dump(forecasts, f)
    print(f"Saved initial super forecasts to {forecasts_file}")

    return forecasts

if __name__ == "__main__":

    # Load initial forecasts from files
    questions, llmcasts_by_qid_sfid, examples_used_by_qid_sfid = load_forecasts_with_examples()

    os.makedirs(output_dir, exist_ok=True)


    examples_flattened_per_question = {}
    for question in questions:


        llmcasts_by_sfid = llmcasts_by_qid_sfid[question.id]
        examples_used = examples_used_by_qid_sfid.get(question.id, {})

        # Each superforecaster gets their own expert instance
        experts = {sfid: Expert(llm, config=config.get('model', {})) for sfid in llmcasts_by_sfid.keys()}

        # Populate the experts with their initial forecasts
        # We take the median of the sample forecasts for each superforecaster

        # Take 5 random superforecasters if more than 5
        if len(experts) > 5:
            selected_sfs = random.Random(SEED).sample(list(experts.keys()), 5)
            experts = {sfid: experts[sfid] for sfid in selected_sfs}
            examples_used = {sfid: examples_used.get(sfid, []) for sfid in selected_sfs}

        print(f"Running Delphi for question {question.id} with {len(experts)} experts")

        # Flatten all examples used across all superforecasters for this question, dedupe by question_id
        all_examples = []
        for sf_examples in examples_used.values():
            for ex_list in sf_examples:
                all_examples.extend(ex_list)
        # Deduplicate by question_id
        seen_qids = set()
        examples_used_flattened = []
        for ex in all_examples:
            qid = ex[0].id
            if qid and qid not in seen_qids:
                seen_qids.add(qid)
            examples_used_flattened.append(ex)

        examples_flattened_per_question[question.id] = examples_used_flattened

    del experts

    forecasts = get_initial_super_forecasts(selected_resolution_date, questions, examples_flattened_per_question)

    for question in questions:

        output_file = os.path.join(output_dir, f"delphi_log_with_examples_{question.id}_{selected_resolution_date}.json")
        if os.path.exists(output_file):
            print(f"Skipping {output_file} (already exists)")
            continue

        super_expert = Expert(llm, config=config.get('model', {}))

        # Find the forecast for this question from the forecasts list
        super_expert_forecasts = next(
            (f for f in forecasts if f.get("question_id") == question.id), None
        )

        forecast_values = super_expert_forecasts['forecasts']
        full_convos = super_expert_forecasts['full_conversation']

        median_index = np.argsort(forecast_values)[len(forecast_values) // 2]
        median_forecast = forecast_values[median_index]
        median_full_convo = full_convos[median_index]

        super_expert.conversation_manager.add_messages(median_full_convo)

        # Instantiate the Delphi mediator
        mediator = Mediator(llm, config=config.get('model', {}))

        # Structured log across all rounds
        delphi_log = {
            "question": question.id,
            "rounds": [],     # list of { round, mediator_feedback, experts: {id: {prob, response}} }
            "histories": None # filled at end with full convo histories
        }

        # Round 0: capture initial expert responses (text + parsed prob)
        initial_expert_entry = {
            0: {
                "prob": median_forecast,
                "response": super_expert.get_last_response()
            }
        }


        delphi_log["rounds"].append({
            "round": 0,
            "mediator_feedback": "Initial forecasts collected.",
            "experts": initial_expert_entry,
        })

        for round_idx in range(n_rounds):
            print(f"Round {round_idx + 1} for question {question.id}")

            # mediator context + intake
            mediator.start_round(round_idx=round_idx, question=question)
            expert_messages = {super_id: {"role": "assistant", "content": entry["response"]}
                            for super_id, entry in delphi_log["rounds"][-1]["experts"].items()}
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
            round_expert_entry = {}
            prob, response = asyncio.run(super_expert.get_forecast_update(broadcast_msg))
            # prob comes from your Expert; still store parsed prob defensively from the text
            round_expert_entry = {
                0: {
                    "prob": max(0.0, min(1.0, float(prob))) if isinstance(prob, (int, float)) else _extract_prob(response),
                    "response": response,
            }
            }

            delphi_log["rounds"].append({
                "round": round_idx + 1,
                "mediator_feedback": broadcast_msg,
                "experts": round_expert_entry,
            })


        # After all rounds: capture full conversation histories
        delphi_log["histories"] = {
            "mediator": list(mediator.conversation_manager.messages),
            "experts": {0: list(super_expert.conversation_manager.messages)}
        }

        with open(output_file, "w") as f:
            json.dump(delphi_log, f, indent=2)


        print(f"Delphi log saved to {output_file}")
