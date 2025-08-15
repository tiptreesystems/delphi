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

from collections import defaultdict

import openai

import time
import pickle
import json

from dotenv import load_dotenv
load_dotenv()

from initial_icl_tests import run_all_forecasts_with_examples, sample_questions_by_topic

import numpy as np
import debugpy
print("Waiting for debugger attach...")
debugpy.listen(5679)
debugpy.wait_for_client()
print("Debugger attached.")

# Set all random seeds for reproducibility

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

config_path = '/home/williaar/projects/delphi/configs/config_openai.yml'
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

initial_forecasts_path = "outputs_initial_forecasts"

output_dir = 'outputs_initial_delphi'


def load_forecasts():
    sampled_questions = sample_questions_by_topic(questions_with_topic, n_per_topic=3)

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

    pkl_files = [
        f for f in os.listdir(f"{initial_forecasts_path}/")
        if f.startswith("collected_fcasts_with_examples") and f.endswith(".pkl") and f"{selected_resolution_date}" in f
    ]


    loaded_llmcasts = {}
    for fname in pkl_files:
        # Extract question id between 'collected_fcasts_' and '.pkl'
        qid = fname[len(f"collected_fcasts_with_examples_{selected_resolution_date}_"): -len(".pkl")]
        with open(f"{initial_forecasts_path}/{fname}", "rb") as f:
            loaded_llmcasts[qid] = [q for q in pickle.load(f)]

    # each key is a question id, each value is a list of tuples (qid, sfid, payload)
    # each payload is a dict with keys: 'question', 'forecast', 'superforecaster_id', 'example_pairs'
    # Extract the list of questions, and return these as a list
    # Also, extract the payloads into a nested dict qid, sfid
    # (1) Extract list of questions (one per qid; taken from the first payload)
    questions = []
    for qid, payloads in loaded_llmcasts.items():
        if payloads:
            qtext = payloads[0].get("question", "")
            questions.append(qtext)

    # (2) Nest payloads by qid -> sfid -> [payloads]
    llmcasts_by_qid_sfid = defaultdict(lambda: defaultdict(list))
    for qid, payloads in loaded_llmcasts.items():
        for p in payloads:
            sfid = p.get("superforecaster_id")
            if sfid is not None:
                llmcasts_by_qid_sfid[qid][sfid].append(p['forecast'])

    example_pairs_by_qid_sfid = defaultdict(lambda: defaultdict(list))
    for qid, payloads in loaded_llmcasts.items():
        for p in payloads:
            sfid = p.get("superforecaster_id")
            if sfid is not None:
                example_pairs = p["example_pairs"]
                example_pairs_by_qid_sfid[qid][sfid].append(example_pairs)


    # Convert nested default dicts to plain dicts and return
    llmcasts_by_qid_sfid = {qid: dict(sfid_map) for qid, sfid_map in llmcasts_by_qid_sfid.items()}
    example_pairs_by_qid_sfid = {qid: dict(sfid_map) for qid, sfid_map in example_pairs_by_qid_sfid.items()}
    return questions, llmcasts_by_qid_sfid, example_pairs_by_qid_sfid


if __name__ == "__main__":

    # Load initial forecasts from files
    questions, llmcasts_by_qid_sfid, example_pairs_by_qid_sfid = load_forecasts()

    os.makedirs(output_dir, exist_ok=True)

    for question in questions:

        output_file = os.path.join(output_dir, f"delphi_log_{question.id}_{selected_resolution_date}.json")
        if os.path.exists(output_file):
            print(f"Skipping {output_file} (already exists)")
            continue

        llmcasts_by_sfid = llmcasts_by_qid_sfid[question.id]
        example_pairs = example_pairs_by_qid_sfid.get(question.id, {})

        # Each superforecaster gets their own expert instance
        experts = {sfid: Expert(llm, config=config.get('model', {})) for sfid in llmcasts_by_sfid.keys()}

        # Populate the experts with their initial forecasts
        # We take the median of the sample forecasts for each superforecaster
        for sfid, payload in llmcasts_by_sfid.items():
            full_conversation = payload[0]['full_conversation']
            initial_message = full_conversation[0]
            sampled_forecast_messages = full_conversation[1:]

            def _extract_final_prob(msg: str) -> float | None:
                import re
                # take the last occurrence if multiple; accept forms like 0.35, .35, 1, 0, 0.0
                m = list(re.finditer(r"FINAL\s+PROBABILITY:\s*([01](?:\.\d+)?|\.\d+)", msg, re.I))
                if not m:
                    return None
                p = float(m[-1].group(1))
                return max(0.0, min(1.0, p))  # clamp to [0,1]

            pairs = [(p, m) for m in sampled_forecast_messages if (p := _extract_final_prob(m['content'])) is not None]
            if not pairs:
                continue

            pairs.sort(key=lambda x: x[0])
            median_llmcast, median_message = pairs[len(pairs) // 2]

            conversation = [initial_message, median_message]

            experts[sfid].conversation_manager.add_messages(conversation)

        # Drop any experts that have no initial forecast
        experts = {sfid: expert for sfid, expert in experts.items() if expert.conversation_manager.messages}


        # Take 5 random superforecasters if more than 5
        if len(experts) > 5:
            selected_sfs = random.Random(SEED).sample(list(experts.keys()), 5)
            experts = {sfid: experts[sfid] for sfid in selected_sfs}

        print(f"Running Delphi for question {question.id} with {len(experts)} experts")

        import re

        _prob_pat = re.compile(r'FINAL PROBABILITY:\s*(0?\.\d+|1\.0|0|1)', re.IGNORECASE)

        def _extract_prob(text: str) -> float:
            """
            Extract the last probability mentioned in the text, preferring explicit
            'FINAL PROBABILITY:' markers but falling back to any number match.
            """
            if not text:
                return 0.5

            # Find all explicit 'FINAL PROBABILITY:' occurrences and take the last one
            matches = _prob_pat.findall(text)
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
            print(f"Mediator feedback: {feedback_message}")

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

        # Save the structured log to a file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"delphi_log_{question.id}_{selected_resolution_date}.json")
        with open(output_file, "w") as f:
            json.dump(delphi_log, f, indent=2)

        print(f"Delphi log saved to {output_file}")
