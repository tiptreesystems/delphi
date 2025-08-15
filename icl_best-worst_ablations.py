"""
Script for testing whether Large Language Models (LLMs) can update their forecasts based on feedback.

This script is designed to evaluate the adaptability of LLMs in response to new information or corrections.
It provides a framework for supplying feedback to the model and assessing changes in its subsequent predictions.

Usage:
    - Run the script to initiate a series of forecast prompts.
    - Provide feedback after each forecast.
    - Observe and record how the model's predictions evolve.

Note:
    This script is intended for experimental and research purposes to analyze LLM behavior in dynamic forecasting scenarios.
"""


from delphi import Expert
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

import numpy as np
import debugpy
# print("Waiting for debugger attach...")
# debugpy.listen(5679)
# debugpy.wait_for_client()
# print("Debugger attached.")

from icl_initial_forecasts import run_all_forecasts_with_examples, run_all_forecasts_baseline

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

initial_forecasts_path = "outputs_initial_forecasts"

ablation_forecasts_path = "outputs_ablation_forecasts"

# get questions that have a topic
loader = ForecastDataLoader()
questions_with_topic = loader.get_questions_with_topics()

n_samples = 5
forecast_due_date = "2024-07-21"
selected_resolution_date = '2025-07-21'


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

    sampled_questions = sample_questions_by_topic(questions_with_topic, n_per_topic=3)

    # Remove questions that do not have a resolution on the selected date
    sampled_questions = [
        q for q in sampled_questions
        if loader.get_resolution(question_id=q.id, resolution_date=selected_resolution_date) is not None
    ]

    question_manager = {
        question.id: {'question': question, 'superforecasts': [], 'resolution': None}
        for question in sampled_questions
    }

    # # Override for testing purposes
    # pkl_files = [
    #     f for f in os.listdir(".")
    #     if f.startswith("collected_fcasts") and f.endswith(".pkl")
    # ]

    # loaded_fcasts = {}
    # for fname in pkl_files:
    #     # Extract question id between 'collected_fcasts_' and '.pkl'
    #     qid = fname[len("collected_fcasts_"): -len(".pkl")]
    #     with open(fname, "rb") as f:
    #         loaded_fcasts[qid] = [q[2] for q in pickle.load(f)]

    # sampled_questions = [entries[0]['question'] for entries in loaded_fcasts.values()]

    # Step 1: Get all resolutions and superforecasts, per question
    for question in sampled_questions:

        question_id = question.id
        question_manager[question_id]['resolution'] = loader.get_resolution(question_id=question_id, resolution_date=selected_resolution_date)

        superforecasts = loader.get_super_forecasts(question_id=question.id, resolution_date=selected_resolution_date)
        question_manager[question.id]['superforecasts'] = superforecasts

    # Step 2: For each question, get the best superforecaster's forecast
    for question_id, data in question_manager.items():
        superforecasts = data['superforecasts']
        resolution = data['resolution']

        best_forecast = min(superforecasts, key=lambda f: abs(f.forecast - resolution.resolved_to))
        worst_forecast = max(superforecasts, key=lambda f: abs(f.forecast - resolution.resolved_to))
        question_manager[question_id]['best_superforecast'] = best_forecast
        question_manager[question_id]['worst_superforecast'] = worst_forecast

        formatted_question = data['question'].question
        formatted_question = formatted_question.replace("{resolution_date}", selected_resolution_date)
        formatted_question = formatted_question.replace("{forecast_due_date}", forecast_due_date)

        print(f"Question: {formatted_question}...")
        print(f"Resolution: {resolution.resolved_to if resolution else 'N/A'}")
        print(f"Best superforecast:")
        print(f"  Forecast: {best_forecast.forecast}")
        print(f"  Reasoning: {best_forecast.reasoning}")
        print(f"  Other superforecasts: {[f.forecast for f in superforecasts if f != best_forecast]}")
        print("=" * 80)


    # Step 3: Get the initial example-based LLM forecasts for each question

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

    print(f"With-examples files: {with_examples_files}")
    print(f"No-examples files: {no_examples_files}")

    loaded_fcasts_with_examples = {}
    for fname in with_examples_files:
        # Extract question id between 'collected_fcasts_' and '.pkl'
        qid = fname[len(f"collected_fcasts_with_examples_{selected_resolution_date}_"): -len(".pkl")]
        with open(f"{initial_forecasts_path}/{fname}", "rb") as f:
            loaded_fcasts_with_examples[qid] = [q for q in pickle.load(f)]

    # Step 3: For all superforecasters, get their LLM-based forecasts
    for question_id, data in question_manager.items():


        # Remove the best superforecast by matching superforecaster id
        best_superforecast = data.get('best_superforecast')
        worst_superforecast = data.get('worst_superforecast')
        middle_superforecasts = [
            entry for entry in data['superforecasts']
            if entry.user_id != best_superforecast.user_id and entry.user_id != worst_superforecast.user_id
        ]

        loaded_llmcasts = loaded_fcasts_with_examples[question_id]

        llmcast_by_sf_id = {
            cast['superforecaster_id']: cast['forecast']
            for cast in loaded_llmcasts
        }

        # loop through all superforecasts (except the best one)
        for superforecast_entry in middle_superforecasts[:5]:
            superforecaster_id = superforecast_entry.user_id
            llmcast = llmcast_by_sf_id.get(superforecaster_id, None)

            forecasts = llmcast['forecasts']
            question = llmcast['text']
            full_conversation = llmcast['full_conversation']
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

            # Create a conversation object (list of messages)
            conversation = [initial_message, median_message]

            # Craft feedback based on the best superforecaster's forecast
            result_path_single_best = os.path.join(ablation_forecasts_path, f"{question_id}_{selected_resolution_date}_{superforecast_entry.user_id}_single-best-feedback.json")
            if not os.path.exists(result_path_single_best):
                existing_files = [
                    fname for fname in os.listdir(ablation_forecasts_path)
                    if fname.startswith(f"{question_id}_{selected_resolution_date}_") and fname.endswith("_single-best-feedback.json")
                ]
                if len(existing_files) >= 5:
                    print(f"Already have {len(existing_files)} single-best-feedback results for question {question_id} on {selected_resolution_date}, skipping.")
                    continue
                feedback_message = ( "Another expert provided the following forecast: \n"
                f"REASONING: {best_superforecast.reasoning}\n"
                f"FINAL PROBABILITY: {best_superforecast.forecast:.2f}\n"
                f"After considering the other expert's perspective, think through your reasoning and provide your final probability estimate.\n"
                f"You may reason through the problem, but you MUST end your response with:\n"
                f"FINAL PROBABILITY: [your decimal number between 0 and 1]"
                )

                # Step 4: Get the LLM's updated forecast after feedback
                expert = Expert(llm, user_profile=None, config=config.get('model', {}))
                expert.conversation_manager.messages = conversation.copy()

                # Step 5: Store the updated forecast and reasoning
                time.sleep(10)
                updated_forecast, response = asyncio.run(expert.get_forecast_update(feedback_message))
                print(f"Original forecast: {median_llmcast}")
                print(f"Updated forecast: {updated_forecast}")
                print("-" * 80)
                print(f"Original reasoning: {median_message}")
                print("-" * 80)
                print(f"Feedback provided: {feedback_message}")
                print("-" * 80)
                # print(f"Best superforecaster's forecast: {best_superforecast.forecast}")
                # print(f"Best superforecaster's reasoning: {best_superforecast.reasoning}")
                print(f"LLM response: {response}")
                print("-" * 80)
                print(f"Resolution: {data['resolution'].resolved_to}")
                print(f"Is the best superforecaster closer to the resolution than the original forecast? {'Yes' if abs(best_superforecast.forecast - data['resolution'].resolved_to) < abs(median_llmcast - data['resolution'].resolved_to) else 'No'}")
                print(f"Is the updated forecast closer to the resolution? {'Yes' if abs(updated_forecast - data['resolution'].resolved_to) < abs(median_llmcast - data['resolution'].resolved_to) else 'No'}")
                print(f"Is the updated forecast closer to the best superforecaster's forecast? {'Yes' if abs(updated_forecast - best_superforecast.forecast) < abs(median_llmcast - best_superforecast.forecast) else 'No'}")
                print("=" * 80)
                # Calculate improvement (absolute error reduction)
                original_error = abs(median_llmcast - data['resolution'].resolved_to)
                updated_error = abs(updated_forecast - data['resolution'].resolved_to)
                improvement = original_error - updated_error

                # Save results to results directory

                os.makedirs(ablation_forecasts_path, exist_ok=True)
                result_data = {
                    "question_id": question_id,
                    "resolution_date": selected_resolution_date,
                    "forecast_due_date": forecast_due_date,
                    "superforecaster_id": superforecast_entry.user_id,
                    "question": formatted_question,
                    "resolution": data['resolution'].resolved_to if data['resolution'] else None,
                    "original_forecast": median_llmcast,
                    "updated_forecast": updated_forecast,
                    "original_error": original_error,
                    "updated_error": updated_error,
                    "improvement": improvement,
                    "original_reasoning": median_message,
                    "feedback_message": feedback_message,
                    "llm_response": response,
                    "best_superforecast": {
                        "id": best_superforecast.user_id,
                        "forecast": best_superforecast.forecast,
                        "reasoning": best_superforecast.reasoning,
                    },
                    "worst_superforecast": {
                        "id": worst_superforecast.user_id,
                        "forecast": worst_superforecast.forecast,
                        "reasoning": worst_superforecast.reasoning,
                    },
                    "conversation": expert.conversation_manager.messages
                }

                with open(result_path_single_best, "w") as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                print(f"Results saved to {result_path_single_best}")

            result_path_single_worst = os.path.join(ablation_forecasts_path, f"{question_id}_{selected_resolution_date}_{superforecast_entry.user_id}_single-worst-feedback.json")
            if not os.path.exists(result_path_single_worst):
                existing_files = [
                    fname for fname in os.listdir(ablation_forecasts_path)
                    if fname.startswith(f"{question_id}_{selected_resolution_date}_") and fname.endswith("_single-worst-feedback.json")
                ]
                if len(existing_files) >= 5:
                    print(f"Already have {len(existing_files)} single-worst-feedback results for question {question_id} on {selected_resolution_date}, skipping.")
                    continue

                feedback_message = ( "Another expert provided the following forecast: \n"
                f"REASONING: {worst_superforecast.reasoning}\n"
                f"FINAL PROBABILITY: {worst_superforecast.forecast:.2f}\n"
                f"After considering the other expert's perspective, think through your reasoning and provide your final probability estimate.\n"
                f"You may reason through the problem, but you MUST end your response with:\n"
                f"FINAL PROBABILITY: [your decimal number between 0 and 1]"
                )

                # Step 4: Get the LLM's updated forecast after feedback
                expert = Expert(llm, user_profile=None, config=config.get('model', {}))
                expert.conversation_manager.messages = conversation.copy()

                # Step 5: Store the updated forecast and reasoning
                time.sleep(10)
                updated_forecast, response = asyncio.run(expert.get_forecast_update(feedback_message))
                print(f"Original forecast: {median_llmcast}")
                print(f"Updated forecast: {updated_forecast}")
                print("-" * 80)
                print(f"Original reasoning: {median_message}")
                print("-" * 80)
                print(f"Feedback provided: {feedback_message}")
                print("-" * 80)
                print(f"LLM response: {response}")
                print("-" * 80)
                print(f"Resolution: {data['resolution'].resolved_to}")
                print(f"Is the worst superforecaster closer to the resolution than the original forecast? {'Yes' if abs(worst_superforecast.forecast - data['resolution'].resolved_to) < abs(median_llmcast - data['resolution'].resolved_to) else 'No'}")
                print(f"Is the updated forecast closer to the resolution? {'Yes' if abs(updated_forecast - data['resolution'].resolved_to) < abs(median_llmcast - data['resolution'].resolved_to) else 'No'}")
                print(f"Is the updated forecast closer to the worst superforecaster's forecast? {'Yes' if abs(updated_forecast - worst_superforecast.forecast) < abs(median_llmcast - worst_superforecast.forecast) else 'No'}")
                print("=" * 80)
                # Calculate improvement (absolute error reduction)
                original_error = abs(median_llmcast - data['resolution'].resolved_to)
                updated_error = abs(updated_forecast - data['resolution'].resolved_to)
                improvement = original_error - updated_error

                # Save results to results directory

                os.makedirs(ablation_forecasts_path, exist_ok=True)
                result_data = {
                    "question_id": question_id,
                    "resolution_date": selected_resolution_date,
                    "forecast_due_date": forecast_due_date,
                    "superforecaster_id": superforecast_entry.user_id,
                    "question": formatted_question,
                    "resolution": data['resolution'].resolved_to if data['resolution'] else None,
                    "original_forecast": median_llmcast,
                    "updated_forecast": updated_forecast,
                    "original_error": original_error,
                    "updated_error": updated_error,
                    "improvement": improvement,
                    "original_reasoning": median_message,
                    "feedback_message": feedback_message,
                    "llm_response": response,
                    "best_superforecast": {
                        "id": best_superforecast.user_id,
                        "forecast": best_superforecast.forecast,
                        "reasoning": best_superforecast.reasoning,
                    },
                    "worst_superforecast": {
                        "id": worst_superforecast.user_id,
                        "forecast": worst_superforecast.forecast,
                        "reasoning": worst_superforecast.reasoning,
                    },
                    "conversation": expert.conversation_manager.messages
                }

                with open(result_path_single_worst, "w") as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                print(f"Results saved to {result_path_single_worst}")

            result_path_double = os.path.join(ablation_forecasts_path, f"{question_id}_{selected_resolution_date}_{superforecast_entry.user_id}_both-best-worst-feedback.json")
            if not os.path.exists(result_path_double):

                existing_files = [
                    fname for fname in os.listdir(ablation_forecasts_path)
                    if fname.startswith(f"{question_id}_{selected_resolution_date}_") and fname.endswith("_both-best-worst-feedback.json")
                ]
                if len(existing_files) >= 5:
                    print(f"Already have {len(existing_files)} both-best-worst-feedback results for question {question_id} on {selected_resolution_date}, skipping.")
                    continue
                # Now we do both best and worst superforecaster in the feedback
                feedback_message = ( "Two experts provided the following forecasts: \n"
                f"Expert 1:\n"
                f"REASONING: {worst_superforecast.reasoning}\n"
                f"FINAL PROBABILITY: {worst_superforecast.forecast:.2f}\n"
                f"Expert 2:\n"
                f"REASONING: {best_superforecast.reasoning}\n"
                f"FINAL PROBABILITY: {best_superforecast.forecast:.2f}\n"
                f"After considering the other experts' perspectives, think through your reasoning and provide your final probability estimate.\n"
                f"You may reason through the problem, but you MUST end your response with:\n"
                f"PROBABILITY ADJUSTMENT: [your decimal number between 0 and 1]\n"
                f"ADJUSTMENT DIRECTION: [increase/decrease]\n"
                f"FINAL PROBABILITY: [your decimal number between 0 and 1]"
                )

                # Step 4: Get the LLM's updated forecast after feedback
                expert = Expert(llm, user_profile=None, config=config.get('model', {}))
                expert.conversation_manager.messages = conversation.copy()

                # Step 5: Store the updated forecast and reasoning
                time.sleep(10)
                updated_forecast, response = asyncio.run(expert.get_forecast_update(feedback_message))
                print(f"Original forecast: {median_llmcast}")
                print(f"Updated forecast: {updated_forecast}")
                print("-" * 80)
                print(f"Original reasoning: {median_message}")
                print("-" * 80)
                print(f"Feedback provided: {feedback_message}")
                print("-" * 80)
                # print(f"Best superforecaster's forecast: {best_superforecast.forecast}")
                # print(f"Best superforecaster's reasoning: {best_superforecast.reasoning}")
                print(f"LLM response: {response}")
                print("-" * 80)
                print(f"Resolution: {data['resolution'].resolved_to}")
                print(f"Is the best superforecaster closer to the resolution than the original forecast? {'Yes' if abs(best_superforecast.forecast - data['resolution'].resolved_to) < abs(median_llmcast - data['resolution'].resolved_to) else 'No'}")
                print(f"Is the updated forecast closer to the resolution? {'Yes' if abs(updated_forecast - data['resolution'].resolved_to) < abs(median_llmcast - data['resolution'].resolved_to) else 'No'}")
                print(f"Is the updated forecast closer to the best superforecaster's forecast? {'Yes' if abs(updated_forecast - best_superforecast.forecast) < abs(median_llmcast - best_superforecast.forecast) else 'No'}")
                print("=" * 80)

                # Calculate improvement (absolute error reduction)
                original_error = abs(median_llmcast - data['resolution'].resolved_to)
                updated_error = abs(updated_forecast - data['resolution'].resolved_to)
                improvement = original_error - updated_error

                # Save results to results directory
                result_data = {
                    "question_id": question_id,
                    "resolution_date": selected_resolution_date,
                    "forecast_due_date": forecast_due_date,
                    "superforecaster_id": superforecast_entry.user_id,
                    "question": formatted_question,
                    "resolution": data['resolution'].resolved_to if data['resolution'] else None,
                    "original_forecast": median_llmcast,
                    "updated_forecast": updated_forecast,
                    "improvement": improvement,
                    "original_error": original_error,
                    "updated_error": updated_error,
                    "original_reasoning": median_message,
                    "feedback_message": feedback_message,
                    "llm_response": response,
                    "best_superforecast": {
                        "id": best_superforecast.user_id,
                        "forecast": best_superforecast.forecast,
                        "reasoning": best_superforecast.reasoning,
                    },
                    "worst_superforecast": {
                        "id": worst_superforecast.user_id,
                        "forecast": worst_superforecast.forecast,
                        "reasoning": worst_superforecast.reasoning,
                    },
                    "conversation": expert.conversation_manager.messages
                }

                with open(result_path_double, "w") as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)

                print(f"Results saved to {result_path_double}")
                print("=" * 80)

    print("All forecasts completed.")
