from dataset.dataloader import ForecastDataLoader

from pathlib import Path

try:
    from utils.sampling import (
        TRAIN_QUESTION_IDS,
        EVALUATION_QUESTION_IDS,
        EVOLUTION_EVALUATION_QUESTION_IDS,
    )
except Exception:
    import importlib.util as _ilu

    _root = Path(__file__).resolve().parents[1]
    _sampling_fp = _root / "utils" / "sampling.py"
    _spec = _ilu.spec_from_file_location("project_utils_sampling", str(_sampling_fp))
    _mod = _ilu.module_from_spec(_spec)
    assert _spec and _spec.loader, f"Cannot load sampling module at {_sampling_fp}"
    _spec.loader.exec_module(_mod)
    TRAIN_QUESTION_IDS = set(getattr(_mod, "TRAIN_QUESTION_IDS", []))
    EVALUATION_QUESTION_IDS = set(getattr(_mod, "EVALUATION_QUESTION_IDS", []))
    EVOLUTION_EVALUATION_QUESTION_IDS = set(
        getattr(_mod, "EVOLUTION_EVALUATION_QUESTION_IDS", [])
    )

import debugpy

if not debugpy.is_client_connected():
    debugpy.listen(5679)
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    print("Debugger attached")
# This script returns statistics on question topics


def main():
    loader = ForecastDataLoader()

    eval_question_ids = [q for q in loader.questions if q in EVALUATION_QUESTION_IDS]
    eval_questions = [loader.questions[q] for q in eval_question_ids]
    train_question_ids = [q for q in loader.questions if q in TRAIN_QUESTION_IDS]
    # train_questions = [loader.questions[q] for q in train_question_ids]
    evolution_eval_question_ids = [
        q for q in loader.questions if q in EVOLUTION_EVALUATION_QUESTION_IDS
    ]
    # evolution_eval_questions = [loader.questions[q] for q in evolution_eval_question_ids]

    all_question_ids = (
        eval_question_ids + train_question_ids + evolution_eval_question_ids
    )
    all_question_ids = list(set(all_question_ids))
    all_questions = [loader.questions[q] for q in all_question_ids]

    question_to_topic = {}
    for q in all_questions:
        topic = q.topic
        question_to_topic[q.id] = topic

    question_to_source = {}
    for q in all_questions:
        source = q.source
        question_to_source[q.id] = source

    unique_topics = set(question_to_topic.values())

    question_count_by_topic = {topic: 0 for topic in unique_topics}
    for q in all_questions:
        topic = question_to_topic[q.id]
        question_count_by_topic[topic] += 1

    sources_per_topic = {topic: set() for topic in unique_topics}
    for q in all_questions:
        topic = question_to_topic[q.id]
        source = question_to_source[q.id]
        sources_per_topic[topic].add(source)

    # print all questions in both Retail and Company operations
    retail_and_company_questions = {"retail": [], "company_operations": []}

    for q in all_questions:
        topic = question_to_topic[q.id]
        if "Retail" in topic:
            retail_and_company_questions["retail"].append(q.id)
        if "Company operations" in topic:
            retail_and_company_questions["company_operations"].append(q.id)

    print(f"Loaded {len(eval_questions)} evaluation questions")


if __name__ == "__main__":
    main()
