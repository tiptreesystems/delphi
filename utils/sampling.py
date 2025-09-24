import os
import random
from typing import List
from collections import defaultdict

import copy

from pytorch_lightning import seed_everything


# Training set question IDs - All resolved questions by 2025-07-21, excluding evaluation set (69 total)
TRAIN_QUESTION_IDS_WITH_EVOLVE_EVAL = {
    "1a0426c8b3d715290f3ad93dae8dfc9f84d6f6e38ea1fd4d20f11496049204b5",
    "1b12215032357c20078f36029eca8e2c67788d7834cba572d712b7d769a288ee",
    "1d9b2c247a0f16b0531b5fc49bfed62658a4e7c10dc8bd6fee98fb5ff57cf8c0",
    "1f7b3ef2436775c7530d7858dc141a4cf7a5692745e9f476e1bbd534d183f3f2",
    "27b3562010fedb45de26cc72b0f1dfcd422aaa195c5c1acdd369833f55a2ac51",
    "3527af895c3fba5850bad3c91193244d88ef272e0bc7251e828b9b2029236ad9",
    "38ffedf32d4ca14903ace5aea31ec5a49444b8f646db87a8d0a32746c5359abe",
    "4204aec5ff81b3d331f27141b072979d838ed95bcd0de36e887ca9a70523060a",
    "447c809c60421e327f266315b62b36749b6362b8b688c133985c93c9ad9be608",
    "45db5d06a001a6fa62eb9b23236adab43c56970d70a833ca206fa42a57f4b7e6",
    "4d0226bb7d28b3509344cecbac6e7003dbb7cb5ac9ffc68518218fb163925931",
    "5713f8a61c04fa270a3a9e1791e4d9b5fa8e0d1cc1c1c232aef84d80c5b89c09",
    "600e496f50daaa9743707de63ce115d974431d123cd136ff68b8f92bc74eb435",
    "61c0fb3703e68cee2439afd5c2d71522bc6649a1fa154491f58981456fa8ab68",
    "7c17d34e37d8cea481d3933f4e1c2c091bd523c3980043e539cde90fbc08f29a",
    "8cdd7e6836d4ad5c7964232f54746ec002813b42220648e34541e1757fd1d2be",
    "9043472375a02690dfb338bd3d11605105562e5cae9672a989961b0c5bef9b51",
    "9d683eff8747219752163f7d11d94ff96bbf3b3386a147b59c4829ddb5dac130",
    "9fa89a7d296950fe794a71be32c65e5d50930a8dbe0f9a8c780f27eec1529e60",
    "AXON",
    "BAMLCC0A0CMTRIV",
    "BAMLH0A2HYB",
    "CBRE",
    "CCL",
    "CLX",
    "DAAA",
    "DAY",
    "DCOILBRENTEU",
    "DFII10",
    "DPCREDIT",
    "DPSACBW027SBOG",
    "ECBDFR",
    "EXPINF30YR",
    "FFIV",
    "H41RESPPALDKNWW",
    "IHLIDXUS",
    "IUDSOIA",
    "KHC",
    "REAINTRATREARAT10Y",
    "RRPONTTLD",
    "SNA",
    "SOFRINDEX",
    "SOLV",
    "T",
    "T10YIE",
    "TPR",
    "TSCO",
    "WLCFLPCL",
    "a6df218adf1cf6a40983148234c46052bc83c7d2b8e31157dfdd6e58a9f83f5d",
    "ac7b7ca153f613fa81d9c0eff4f0aa25bb62bba04da7610ae4eec7e1586425bb",
    "c37effa43385e2f5a9a91bc99a278ac376fb8f10f1e11ea39fb621bfbf6e2c2f",
    "ccda7990a2565cabd7c375a036751bd3b953b8bed45d859010919cd3a84d7e78",
    "d61d058797047fb9793684b123dcf88a66f843695d9e65e9bc6df0f49ec9d936",
    "e2ebe8c99e10583715b46ee35c93275a9a6b5721f812e6ce07f996daa8159732",
    "e53dabd31f71786f3b044bd12e498deee5a732a43de2d9be7468ebaced466977",
    "eda8e5b43e0db651905667586e1e72a7d5679cbb5b3ef4dd6faa6444759e2dee",
    "f29228009a407cdf130062251d274b898fc8e925f21b5d9c16376e35fd5a9fbd",
    "fa23cf1ab8ae4be34faeccb0c0453b19974158a7a1cb10657339b11a869ce089",
    "meteofrance_TEMPERATURE_celsius.07020.D",
    "meteofrance_TEMPERATURE_celsius.07110.D",
    "meteofrance_TEMPERATURE_celsius.07117.D",
    "meteofrance_TEMPERATURE_celsius.07130.D",
    "meteofrance_TEMPERATURE_celsius.07190.D",
    "meteofrance_TEMPERATURE_celsius.07240.D",
    "meteofrance_TEMPERATURE_celsius.07335.D",
    "meteofrance_TEMPERATURE_celsius.07591.D",
    "meteofrance_TEMPERATURE_celsius.07621.D",
    "meteofrance_TEMPERATURE_celsius.07650.D",
    "meteofrance_TEMPERATURE_celsius.07761.D",
}

EVALUATION_QUESTION_IDS = [
    "LULU",
    "DECK",
    "CTAS",
    "CZR",
    "KMB",
    "NASDAQ100",
    "DTB1YR",
    "TMBACBW027SBOG",
    "DTB3",
    "DTB4WK",
    "3d8c7edc8a70211d39a3e4827f448e5908cd5e98dbbb1e3d7501c7761a877788",
    "7de1da056e3c1cf3e02778d4631cd6d10a716795d083dd2f64d3988d10b27848",
    "0c20162bae6257b7fcaa67c009a710e82f0faad0ce168d775b2bc3524121f83a",
    "d9c14625f757e5fb87a7bdfe5154170e6fd569a1bb4b215cf1fb50e704bdc739",
    "b48c0a80ed5a62b128815b699b93d3d167bf2a3265058028871320c0a5c85913",
    "27e948952d66a749e55f37f9cb63036955768951570c627c3f16ec378df057f9",
    "3c37fef353460bfd130fde0117638badaee913ee8c79b8cdf4c35e2c5710126a",
    "cfecaf75abdfe4be7627c5e61a5d7c88541a74fbf3f030dd0b3b81e3f456e655",
    "b70970a0440d1b7dedde9220fb60ffe3f2ed8b00ef12b45341772046caa12092",
    "4c98ccd0d64f8cad6896b50e8131b7173dcc95e40e1492413d69a7bf75d3328c",
    "meteofrance_TEMPERATURE_celsius.07015.D",
    "meteofrance_TEMPERATURE_celsius.07607.D",
    "meteofrance_TEMPERATURE_celsius.07577.D",
    "meteofrance_TEMPERATURE_celsius.07481.D",
    "meteofrance_TEMPERATURE_celsius.78925.D",
    "WYNN",
    "RJF",
    "GLW",
    "HUM",
    "FANG",
    "cf373450466c71a49a7f2e82e176b66198255960bb1cf71d1721cd258d0c2b81",
    "2cab1958b1bd726695b512f3fa260e0b2ad5dbeadc5c5e1bc5b35104733c57fc",
    "2f9337f5d2cc530629386a651bc047c7b76cb3d2fa3222a2fca72a8a7a20e7b2",
    "5f484517610914d8596bbc5302f164a81bb62bab27d481137fe178bc7a038ed8",
    "917240d03f7e16595c3decf76c8e905081ffcc59d16c50f0128332abd87516ee",
]

# New test set question IDs - Additional questions for testing new models/prompts
EVOLUTION_EVALUATION_QUESTION_IDS = [
    "eda8e5b43e0db651905667586e1e72a7d5679cbb5b3ef4dd6faa6444759e2dee",
    "meteofrance_TEMPERATURE_celsius.07190.D",
    "BAMLCC0A0CMTRIV",
    "meteofrance_TEMPERATURE_celsius.07117.D",
    "T",
    "RRPONTTLD",
    "45db5d06a001a6fa62eb9b23236adab43c56970d70a833ca206fa42a57f4b7e6",
    "EXPINF30YR",
    "meteofrance_TEMPERATURE_celsius.07130.D",
    "ECBDFR",
    "fa23cf1ab8ae4be34faeccb0c0453b19974158a7a1cb10657339b11a869ce089",
    "1b12215032357c20078f36029eca8e2c67788d7834cba572d712b7d769a288ee",
    "447c809c60421e327f266315b62b36749b6362b8b688c133985c93c9ad9be608",
    "H41RESPPALDKNWW",
    "9043472375a02690dfb338bd3d11605105562e5cae9672a989961b0c5bef9b51",
    "SOLV",
    "CLX",
    "4204aec5ff81b3d331f27141b072979d838ed95bcd0de36e887ca9a70523060a",
]

TRAIN_QUESTION_IDS = list(
    TRAIN_QUESTION_IDS_WITH_EVOLVE_EVAL
    - set(EVALUATION_QUESTION_IDS)
    - set(EVOLUTION_EVALUATION_QUESTION_IDS)
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


def sample_questions(
    base_config: dict, questions_with_topic: List, loader, method_override=None
) -> List:
    """Sample questions based on the configuration."""
    config = copy.deepcopy(base_config)
    sampling_config = config["data"]["sampling"]
    experiment_config = config["experiment"]
    selected_resolution_date = config["data"]["resolution_date"]
    base_initial_dir = experiment_config["initial_forecasts_dir"]
    reuse_config = experiment_config.get("reuse_initial_forecasts", {})
    seed = experiment_config["seed"]
    # Use seed-specific directory for initial forecasts
    initial_forecasts_path = os.path.join(base_initial_dir, f"seed_{seed}")
    seed_everything(seed)
    print(f"Set random seed to {seed} before question sampling")

    method = sampling_config["method"]
    if method_override is not None:
        method = method_override
        print(f"Overriding sampling method to: {method_override}")

    if method == "by_topic":
        n_per_topic = sampling_config["n_per_topic"]
        sampled_questions = sample_questions_by_topic(
            questions_with_topic, n_per_topic=n_per_topic, seed=seed
        )
        print(f"Sampling method: by_topic ({n_per_topic} per topic)")

    elif method == "random":
        n_questions = sampling_config.get("n_questions", 10)
        sampled_questions = random.sample(
            questions_with_topic, min(n_questions, len(questions_with_topic))
        )
        print(f"Sampling method: random ({n_questions} questions)")

    elif method == "train" or method == "tune":
        # 'tune' kept for backward compatibility
        sampled_questions = _handle_predefined_questions(
            TRAIN_QUESTION_IDS,
            questions_with_topic,
            reuse_config,
            initial_forecasts_path,
            selected_resolution_date,
            config,
            "train",
        )

    elif method == "train_small":
        n_per_topic = sampling_config.get("n_per_topic", 2)
        all_train_questions = [
            q for q in questions_with_topic if q.id in TRAIN_QUESTION_IDS
        ]
        sampled_questions = sample_questions_by_topic(
            all_train_questions, n_per_topic=n_per_topic, seed=seed
        )
        print(
            f"Sampling method: train_small ({n_per_topic} per topic from training set, total {len(sampled_questions)} questions)"
        )

    elif method == "evaluation":
        sampled_questions = _handle_predefined_questions(
            EVALUATION_QUESTION_IDS,
            questions_with_topic,
            reuse_config,
            initial_forecasts_path,
            selected_resolution_date,
            config,
            "evaluation",
        )

    elif method == "evolution_evaluation":
        sampled_questions = _handle_predefined_questions(
            EVOLUTION_EVALUATION_QUESTION_IDS,
            questions_with_topic,
            reuse_config,
            initial_forecasts_path,
            selected_resolution_date,
            config,
            "evolution_evaluation",
        )

    elif method == "single":
        question_id = sampling_config.get("question_id")
        if not question_id:
            raise ValueError(
                "For 'single' sampling method, 'question_id' must be specified in sampling config"
            )

        matching_questions = [q for q in questions_with_topic if q.id == question_id]
        if not matching_questions:
            raise ValueError(f"No question found with ID: {question_id}")

        sampled_questions = matching_questions
        print(f"Sampling method: single (question ID: {question_id})")

    elif method == "first":
        n_questions = sampling_config.get("n_questions", 10)
        sampled_questions = questions_with_topic[:n_questions]
        print(f"Sampling method: first ({n_questions} questions)")

    elif method == "from_initial_forecasts":
        sampled_questions = _handle_from_initial_forecasts(
            sampling_config,
            reuse_config,
            initial_forecasts_path,
            selected_resolution_date,
            questions_with_topic,
            config,
            seed,
        )

    else:
        n_questions = sampling_config.get("n_questions", 10)
        sampled_questions = questions_with_topic[:n_questions]
        print(f"Sampling method: default ({n_questions} questions)")

    print(f"Questions after initial sampling: {len(sampled_questions)}")

    if config["data"]["filters"]["require_resolution"]:
        before_filter = len(sampled_questions)
        sampled_questions = [
            q
            for q in sampled_questions
            if loader.get_resolution(
                question_id=q.id, resolution_date=selected_resolution_date
            )
            is not None
        ]
        print(
            f"Questions after resolution filter: {len(sampled_questions)} (filtered out {before_filter - len(sampled_questions)})"
        )

    return sampled_questions


def _handle_predefined_questions(
    include_ids,
    questions_with_topic,
    reuse_config,
    initial_forecasts_path,
    selected_resolution_date,
    config,
    method_name,
):
    """Handle tune and evaluation question sampling."""
    # if reuse_config.get('enabled', False):
    #     if os.path.exists(initial_forecasts_path):
    #         pkl_files = [
    #             f for f in os.listdir(initial_forecasts_path)
    #             if f.startswith("collected_fcasts_with_examples") and f.endswith(".json") and f"{selected_resolution_date}" in f
    #         ]
    #         existing_question_ids = [fname[len(f"collected_fcasts_with_examples_{selected_resolution_date}_"): -len(".json")] for fname in pkl_files]
    #         available_eval_ids = [id for id in include_ids if id in existing_question_ids]
    #     else:
    #         available_eval_ids = []

    #     if len(available_eval_ids) == len(include_ids):
    #         sampled_questions = [q for q in questions_with_topic if q.id in include_ids]
    #         print(f"Sampling method: {method_name} ({len(sampled_questions)} questions from existing initial forecasts)")
    #     else:
    #         missing_ids = [id for id in include_ids if id not in available_eval_ids]
    #         print(f"Missing initial forecasts for {len(missing_ids)} {method_name} questions, generating...")

    #         sampled_questions = [q for q in questions_with_topic if q.id in include_ids]

    #         if not sampled_questions:
    #             raise ValueError(f"No questions found matching the {method_name} include_ids")

    #         missing_questions = [q for q in sampled_questions if q.id in missing_ids]
    #         if missing_questions:
    #             from utils.forecast_loader import generate_initial_forecasts_for_questions
    #             generate_initial_forecasts_for_questions(missing_questions, initial_forecasts_path, config, selected_resolution_date)

    #         print(f"Sampling method: {method_name} ({len(sampled_questions)} questions, generated {len(missing_questions)} new initial forecasts)")
    # else:
    sampled_questions = [q for q in questions_with_topic if q.id in include_ids]

    if not sampled_questions:
        raise ValueError(f"No questions found matching the {method_name} include_ids")

    print(
        f"Sampling method: {method_name} ({len(sampled_questions)} questions from predefined list, no reuse)"
    )

    return sampled_questions


def _handle_from_initial_forecasts(
    sampling_config,
    reuse_config,
    initial_forecasts_path,
    selected_resolution_date,
    questions_with_topic,
    config,
    seed,
):
    """Handle from_initial_forecasts sampling method."""
    if not reuse_config.get("enabled", False):
        raise ValueError(
            "'from_initial_forecasts' sampling method requires reuse_initial_forecasts.enabled=true"
        )

    if os.path.exists(initial_forecasts_path):
        pkl_files = [
            f
            for f in os.listdir(initial_forecasts_path)
            if f.startswith("collected_fcasts_with_examples")
            and f.endswith(".json")
            and f"{selected_resolution_date}" in f
        ]
    else:
        pkl_files = []

    n_questions = sampling_config.get("n_questions", 10)
    n_per_topic = sampling_config.get("n_per_topic", None)

    if pkl_files:
        question_ids = [
            fname[
                len(
                    f"collected_fcasts_with_examples_{selected_resolution_date}_"
                ) : -len(".json")
            ]
            for fname in pkl_files
        ]
        if len(question_ids) >= n_questions:
            sorted_question_ids = sorted(question_ids)
            available_questions = [
                q for q in questions_with_topic if q.id in sorted_question_ids
            ]

            if n_per_topic is not None:
                sampled_questions = sample_questions_by_topic(
                    available_questions, n_per_topic=n_per_topic, seed=seed
                )
                print(
                    f"Sampling method: from_initial_forecasts ({len(sampled_questions)} questions from existing {initial_forecasts_path}, stratified by topic: {n_per_topic} per topic)"
                )
            else:
                selected_question_ids = sorted_question_ids[:n_questions]
                sampled_questions = [
                    q for q in questions_with_topic if q.id in selected_question_ids
                ]
                print(
                    f"Sampling method: from_initial_forecasts ({len(sampled_questions)} questions from existing {initial_forecasts_path}, deterministic selection)"
                )
        else:
            print(
                f"Not enough questions in existing initial forecasts ({len(question_ids)} < {n_questions}), generating new initial forecasts..."
            )
            if n_per_topic is not None:
                sampled_questions = sample_questions_by_topic(
                    questions_with_topic, n_per_topic=n_per_topic, seed=seed
                )
                print(
                    f"Generating new initial forecasts with stratified sampling: {n_per_topic} per topic"
                )
            else:
                sampled_questions = questions_with_topic[:n_questions]
            from utils.forecast_loader import generate_initial_forecasts_for_questions

            reuse_cfg = (config.get("experiment", {}) or {}).get(
                "reuse_initial_forecasts", {}
            )
            with_examples_flag = (
                reuse_cfg.get("with_examples")
                if isinstance(reuse_cfg, dict) and "with_examples" in reuse_cfg
                else (config.get("initial_forecasts", {}) or {}).get(
                    "with_examples", True
                )
            )
            generate_initial_forecasts_for_questions(
                sampled_questions,
                initial_forecasts_path,
                config,
                selected_resolution_date,
                with_examples=with_examples_flag,
            )
            print(
                f"Sampling method: from_initial_forecasts ({len(sampled_questions)} questions, generated new initial forecasts)"
            )
    else:
        print("No initial forecasts found, generating new initial forecasts...")
        if n_per_topic is not None:
            sampled_questions = sample_questions_by_topic(
                questions_with_topic, n_per_topic=n_per_topic, seed=seed
            )
            print(
                f"Generating new initial forecasts with stratified sampling: {n_per_topic} per topic, total {len(sampled_questions)} questions"
            )
        else:
            sampled_questions = questions_with_topic[:n_questions]
            print(f"Generating new initial forecasts for {n_questions} questions...")
        from utils.forecast_loader import generate_initial_forecasts_for_questions

        reuse_cfg = (config.get("experiment", {}) or {}).get(
            "reuse_initial_forecasts", {}
        )
        with_examples_flag = (
            reuse_cfg.get("with_examples")
            if isinstance(reuse_cfg, dict) and "with_examples" in reuse_cfg
            else (config.get("initial_forecasts", {}) or {}).get("with_examples", True)
        )
        generate_initial_forecasts_for_questions(
            sampled_questions,
            initial_forecasts_path,
            config,
            selected_resolution_date,
            with_examples=with_examples_flag,
        )
        print(
            f"Sampling method: from_initial_forecasts ({len(sampled_questions)} questions, generated new initial forecasts)"
        )

    return sampled_questions
