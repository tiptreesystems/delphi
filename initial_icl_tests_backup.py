import yaml
from delphi import Expert
from models import LLMFactory, LLMProvider, LLMModel
from dataset.dataloader import ForecastDataLoader

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

config_path = "./configs/config_openai.yml"

def load_config(config_path: str = "configs/config.yml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config(config_path)

resolutions_path = "./dataset/datasets/resolution_sets/2024-07-21_resolution_set.json"


provider = LLMProvider.OPENAI
model = LLMModel.GPT_4O_2024_05_13
personalized_system_prompt = (
    "You are a helpful assistant with expertise in forecasting and decision-making."
)

openai_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

llm = LLMFactory.create_llm(provider, model, system_prompt=personalized_system_prompt)

# get questions that have a topic
loader = ForecastDataLoader()
questions_with_topic = loader.get_questions_with_topics()

forecast_due_date = "2024-07-21"  # Example date, adjust as needed
n_samples = 5
selected_dates = ["2034-07-19"]
# selected_dates = ['2025-07-21', '2027-07-21', '2029-07-20', '2034-07-19']


def find_example_questions(question, loader, selected_date, strategy="most_questions"):
    user_to_questions = find_topic_relevant_users(question, loader, selected_date)

    if not user_to_questions:
        print(f"No relevant users found for question {question.id}.")
        return [], None

    qa_pairs = []

    if strategy == "most_questions":
        # Find the user with the most questions
        max_user = max(user_to_questions.items(), key=lambda x: len(x[1]))[0]
        max_user_questions = user_to_questions[max_user]
        qa_pairs = get_super_forecasts_by_user(
            max_user, max_user_questions, loader, selected_date
        )

    elif strategy == "random":
        eligible_users = [
            user_id for user_id, qs in user_to_questions.items() if len(qs) >= 5
        ]
        if eligible_users:
            chosen_user = random.choice(eligible_users)
            chosen_user_questions = user_to_questions[chosen_user]
            qa_pairs = get_super_forecasts_by_user(
                chosen_user, chosen_user_questions, loader, selected_date
            )

    # qa_pairs now contains tuples of (question, forecast) for one superforecaster

    random.seed(42)
    random.shuffle(qa_pairs)

    return qa_pairs, max_user


def find_topic_relevant_users(question, loader, selected_date, cutoff=5):
    print(f"Finding topic-relevant users for {question.id}...")
    example_questions = loader.get_same_topic_questions(question.id)

    # Collect all forecasts for each user across all example questions
    user_to_questions = defaultdict(list)

    for q in example_questions:
        super_forecasts = loader.get_super_forecasts(
            question_id=q.id, selected_date=selected_date
        )
        for sf in super_forecasts:
            user_to_questions[sf.user_id].append(q)

    # Filter users with at least `cutoff` questions
    relevant_users = {
        user_id: qs for user_id, qs in user_to_questions.items() if len(qs) >= cutoff
    }
    # print(f"Found {len(relevant_users)} relevant users with at least {cutoff} questions.")

    return relevant_users


def get_super_forecasts_by_user(user_id, questions, loader, selected_date):
    user_forecasts = []
    for q in questions:
        forecasts = loader.get_super_forecasts(
            question_id=q.id, selected_date=selected_date
        )
        user_forecast = next((sf for sf in forecasts if sf.user_id == user_id), None)
        if user_forecast:
            user_forecasts.append((q, user_forecast))
    return user_forecasts


def sample_questions_by_topic(questions, n_per_topic=3, seed=42):
    random.seed(seed)
    # Collect all unique topics
    unique_topics = set(q.topic for q in questions)
    topic_to_questions = defaultdict(list)
    # Shuffle questions to randomize selection
    shuffled_questions = random.sample(questions, len(questions))
    for topic in unique_topics:
        topic_questions = [q for q in shuffled_questions if q.topic == topic]
        topic_to_questions[topic] = topic_questions[:n_per_topic]
    # Only include topics that have at least n_per_topic questions
    for topic in unique_topics:
        topic_questions = [q for q in shuffled_questions if q.topic == topic]
        if len(topic_questions) < n_per_topic:
            raise ValueError(
                f"Topic '{topic}' only has {len(topic_questions)} questions, but {n_per_topic} are required."
            )
        topic_to_questions[topic] = topic_questions[:n_per_topic]
    sampled_questions = [
        q for qs in topic_to_questions.values() if len(qs) == n_per_topic for q in qs
    ]
    return sampled_questions


def map_questions_to_forecaster_ids(
    loader, selected_dates, find_example_questions, sampled_questions
):
    # patch: map backwards from question id to forecaster id
    # This assumes that the question id is unique and corresponds to the forecaster id in the results
    # If the structure is different, this logic will need to be adjusted accordingly
    q_to_forecaster_ids = {}
    for q in sampled_questions:
        _, max_user = find_example_questions(
            q, loader, selected_dates[0], strategy="most_questions"
        )
        q_to_forecaster_ids[q.id] = max_user
    return q_to_forecaster_ids

async def run_all_forecasts(questions):

    semaphore = asyncio.Semaphore(5)

    async def forecast_question(q):
        tasks = [throttled_forecast_entry(q, date, semaphore) for date in selected_dates]
        entries = await asyncio.gather(*tasks)
        return q.id, {
            'question': q,
            'entries': entries
        }

    all_tasks = [forecast_question(q) for q in questions]
    results = await asyncio.gather(*all_tasks)

    return dict(results)

async def throttled_forecast_entry(q, date, semaphore, wait_time=30, retries=10):
    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                return await forecast_entry(q, date, n_samples=n_samples)
            except openai.RateLimitError as e:
                if attempt < retries:
                    sleep_time = wait_time * (2 ** (attempt - 1))
                    print(f"[{q.id}] Rate limit hit (attempt {attempt}) — sleeping {sleep_time}s")
                    await asyncio.sleep(sleep_time)
                else:
                    print(f"[{q.id}] Rate limit error after {retries} attempts. Giving up.")
                    return {
                    'date': date,
                    'text': q.question,
                    'forecasts': [],
                    'full_conversation': [],
                    'error': 'rate_limit_exceeded'
                    }

async def forecast_entry(q, date, n_samples=n_samples):
    expert = Expert(llm, user_profile=None, config=config.get('model', {}))
    q_instance = copy.copy(q)
    q_instance.resolution_date = date
    q_instance.question = q.question.replace("{resolution_date}", date)
    q_instance.question = q_instance.question.replace("{forecast_due_date}", forecast_due_date)

    # Parallel forecast generation
    forecast_tasks = [expert.forecast(q_instance, None) for _ in range(n_samples)]
    forecasts = await asyncio.gather(*forecast_tasks)

    return {
        'date': date,
        'text': q_instance.question,
        'forecasts': forecasts,
        'full_conversation': expert.conversation_manager.messages  # watch out for race conditions
    }


async def run_all_forecasts_for_all_superforecasters_with_examples(
    sampled_questions, *, max_examples=5
):
    all_examples = _collect_example_forecasts(sampled_questions, loader, selected_dates)

    for qid in all_examples:
        for sf_id in all_examples[qid]:
            all_examples[qid][sf_id] = all_examples[qid][sf_id][:max_examples]

    semaphore = asyncio.Semaphore(5)
    tasks = [
        asyncio.wait_for(task, timeout=300)  # 5 min per forecast
        for task in _build_tasks(sampled_questions, all_examples, semaphore)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

def _collect_example_forecasts(sampled_questions, loader, selected_dates, *, max_examples=5):
    """
    Return {question_id: {sf_id: example_pairs}}, where each example pair is
    (question_obj, forecast_obj) — the same shape expected by
    `throttled_forecast_entry_with_examples`.
    """
    # quick lookup for already-loaded Question objects
    questions_by_id = {q.id: q for q in sampled_questions}

    example_forecasts_dict = {}
    for q in sampled_questions:
        sf_ids = [
            f.user_id
            for f in loader.get_super_forecasts(
                question_id=q.id, resolution_date=selected_dates[0]
            )
        ]
        if not sf_ids:
            print(f"No superforecasters found for question {q.id}.")
            continue

        example_forecasts_dict[q.id] = {}
        for sf_id in sf_ids:
            forecasts = loader.get_super_forecasts(
                user_id=sf_id,
                resolution_date=selected_dates[0],
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

def _build_tasks(sampled_questions, all_examples, semaphore):
    for q in sampled_questions:
        for sf_id, pair in all_examples.get(q.id, {}).items():
            yield _forecast_one(q, sf_id, pair, semaphore)

async def _forecast_one(question, sf_id, example_forecasts, semaphore):
    return question.id, sf_id, {
        "question": question,
        "forecast": await throttled_forecast_entry_with_examples(
            question, selected_dates[0], example_forecasts, semaphore
        ),
        "superforecaster_id": sf_id,
        "example_pairs": example_forecasts,
    }

async def throttled_forecast_entry_with_examples(q, date, qa_pairs, semaphore, wait_time=30, retries=10):
    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                return await forecast_entry_with_examples(q, date, qa_pairs, n_samples=n_samples)
            except Exception as e:
                if attempt < retries:
                    sleep_time = wait_time * (2 ** (attempt - 1))
                    print(f"[{q.id}] Rate limit hit (attempt {attempt}) — sleeping {sleep_time}s")
                    await asyncio.sleep(sleep_time)
                else:
                    print(f"[{q.id}] Rate limit error after {retries} attempts. Giving up.")
                    return {
                        'date': date,
                        'text': q.question,
                        'forecasts': [],
                        'full_conversation': [],
                        'error': 'rate_limit_exceeded'
                    }

async def forecast_entry_with_examples(q, date, qa_pairs, n_samples=n_samples):
    expert = Expert(llm, user_profile=None, config=config.get('model', {}))
    q_instance = copy.copy(q)
    q_instance.resolution_date = date
    q_instance.question = q.question.replace("{resolution_date}", date)
    q_instance.question = q_instance.question.replace("{forecast_due_date}", forecast_due_date)

    retries = 5
    wait_time = 10

    async def call_with_retry():
            for attempt in range(1, retries+1):
                try:
                    return await expert.forecast_with_examples_in_context(q_instance, qa_pairs)
                except openai.RateLimitError as e:
                    if attempt < retries:
                        await asyncio.sleep(wait_time * (2 ** (attempt-1)))
                    else:
                        raise

    forecast_tasks = [call_with_retry() for _ in range(n_samples)]
    forecasts = await asyncio.gather(*forecast_tasks)
    return {
        'date': date,
        'text': q_instance.question,
        'forecasts': forecasts,
        'full_conversation': expert.conversation_manager.messages
    }




if __name__ == "__main__":

    sampled_questions = sample_questions_by_topic(questions_with_topic, n_per_topic=3)

    with open("forecast_results.json", "r") as f:
        forecast_results = json.load(f)

    with open("forecast_results_with_examples.json", "r") as f:
        forecast_results_with_examples = json.load(f)

    q_to_forecaster_ids = map_questions_to_forecaster_ids(
        loader, selected_dates, find_example_questions, sampled_questions
    )

    # 1st set of plots: Violin plots comparing forecasts with and without examples
    for q in sampled_questions:
        qid = q.id
        superforecaster_id = q_to_forecaster_ids.get(qid)

        plt.figure(figsize=(6, 6))

        # Each question has two sets of forecasts: base and with examples
        forecasts_base = forecast_results[qid]["entries"][0]["forecasts"]
        forecasts_examples = forecast_results_with_examples[qid]["entries"][0]["forecasts"]

        # Show each set of forecasts as a violin plot
        positions = [1, 2]
        parts = plt.violinplot(
            [forecasts_base, forecasts_examples],
            positions=positions,
            showmeans=True,
            widths=0.7,
        )

        # Show the superforecaster's forecast as a horizontal line
        sf_forecast = loader.get_super_forecasts(
            question_id=qid, user_id=superforecaster_id, selected_date=selected_dates[0]
        )[0]
        plt.axhline(
            sf_forecast.forecast,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Superforecaster",
        )

        wrapped_title = "\n".join(textwrap.wrap(q.question, width=60))
        plt.title(wrapped_title)
        plt.suptitle(qid, fontsize=10, y=1.03)
        plt.legend(loc="upper right")
        plt.xlabel("Forecast Method")
        plt.xticks(positions, ["Base", "With Examples"])
        plt.xlim(0.5, 2.5)
        plt.ylim(-0.05, 1.05)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.ylabel("Forecast Value")
        plt.tight_layout()
        plt.show()


    # Now we analyze the improvement in median forecast error
    plt.figure(figsize=(12, 6))

    question_ids = []
    improvement_values = []
    relative_improvement_values = []
    titles = []

    for q in sampled_questions:
        qid = q.id
        superforecaster_id = q_to_forecaster_ids.get(qid)
        # Get forecasts for both with and without examples
        forecasts_base = forecast_results[qid]["entries"][0]["forecasts"]
        forecasts_examples = forecast_results_with_examples[qid]["entries"][0]["forecasts"]

        # Only plot if superforecaster forecast exists
        sf_forecast = loader.get_super_forecasts(
            question_id=qid, user_id=superforecaster_id, selected_date=selected_dates[0]
        )[0]

        median_base = np.median(forecasts_base) if forecasts_base else np.nan
        median_examples = np.median(forecasts_examples) if forecasts_examples else np.nan
        sf_value = sf_forecast.forecast

        # Calculate improvement in fidelity to superforecaster
        base_error = abs(median_base - sf_value)
        examples_error = abs(median_examples - sf_value)
        improvement = base_error - examples_error
        relative_improvement = (
            (base_error - examples_error) / base_error if base_error != 0 else -1
        )

        question_ids.append(qid)
        improvement_values.append(improvement)
        relative_improvement_values.append(relative_improvement)
        titles.append("\n".join(q.id))

    # Plot bar chart
    bars = plt.bar(range(len(improvement_values)), improvement_values, color="orange")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Improvement in Median Forecast Error\n(Base - With Examples)")
    plt.title("Improvement in Median Forecast Error (Closer to Superforecaster)")
    plt.tight_layout()
    plt.show()

    mean_improvement = np.mean(improvement_values)
    std_improvement = np.std(improvement_values)
    median_improvement = np.median(improvement_values)

    print(f"Mean improvement: {mean_improvement:.3f}")
    print(f"Std improvement: {std_improvement:.3f}")
    print(f"Median improvement: {median_improvement:.3f}")

    mean_relative_improvement = np.mean(relative_improvement_values)
    std_relative_improvement = np.std(relative_improvement_values)
    median_relative_improvement = np.median(relative_improvement_values)


    print(f"Mean relative improvement: {mean_relative_improvement:.3f}")
    print(f"Std relative improvement: {std_relative_improvement:.3f}")
    print(f"Median relative improvement: {median_relative_improvement:.3f}")
