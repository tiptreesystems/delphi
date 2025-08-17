import argparse
from delphi import Expert, Mediator
from models import LLMFactory, LLMProvider, LLMModel
from dataset.dataloader import Question, Forecast, Resolution, ForecastDataLoader

import yaml
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

from icl_initial_forecasts import run_all_forecasts_with_examples, sample_questions_by_topic, load_experiment_config as load_initial_config

import numpy as np


def make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    try:
        # Test if already serializable
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass
    
    if obj is None:
        return None
    elif hasattr(obj, '__dict__'):
        # Convert objects with attributes
        if hasattr(obj, 'id') and hasattr(obj, 'question'):  # Question object
            return {
                'question_id': str(obj.id),
                'question_text': str(obj.question),
                'question_background': str(getattr(obj, 'background', '')),
                'type': 'question'
            }
        else:
            # Try to convert other objects to dict
            try:
                return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
            except:
                return str(obj)
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    else:
        # Convert to string as last resort
        return str(obj)


def load_experiment_config(config_path: str = './configs/delphi_experiment.yml') -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_environment(config: dict):
    """Setup environment based on configuration."""
    # Set random seeds
    seed = config['experiment']['seed']
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Setup debugging if enabled
    if config['debug']['enabled']:
        # check if "williaar" is in the file path for this file
        if "williaar" in __file__:
            import debugpy
            print(f"Waiting for debugger attach on port {config['debug']['debugpy_port']}...")
            debugpy.listen(config['debug']['debugpy_port'])
            debugpy.wait_for_client()
            print("Debugger attached.")
    
    if config['debug']['breakpoint_on_start']:
        breakpoint()
    
    # Setup API keys
    api_config = config.get('api', {})
    if 'openai' in api_config:
        openai_key = os.getenv(api_config['openai']['api_key_env'])
        os.environ["OPENAI_API_KEY"] = openai_key
    if 'groq' in api_config:
        groq_key = os.getenv(api_config['groq']['api_key_env'])
        os.environ["GROQ_API_KEY"] = groq_key


def get_llm_from_config(config: dict):
    """Create LLM instance from configuration."""
    model_config = config['model']
    
    # Map string provider to enum
    provider_map = {
        'openai': LLMProvider.OPENAI,
        'claude': LLMProvider.CLAUDE,
        'anthropic': LLMProvider.CLAUDE,
        'groq': LLMProvider.GROQ,
    }
    provider = provider_map.get(model_config['provider'].lower())
    
    # Map model name to enum (you may need to extend this)
    model_map = {
        'gpt-4o-2024-05-13': LLMModel.GPT_4O_2024_05_13,
        'gpt-4o': LLMModel.GPT_4O,
        'claude-3-sonnet': LLMModel.CLAUDE_3_5_SONNET,
        'openai/gpt-oss-20b': LLMModel.GROQ_GPT_OSS_20B,
        'openai/gpt-oss-120b': LLMModel.GROQ_GPT_OSS_120B,
        'llama-3.3-70b-versatile': LLMModel.GROQ_LLAMA_3_3_70B,
        'llama-3.1-70b-versatile': LLMModel.GROQ_LLAMA_3_1_70B,
        'mixtral-8x7b-32768': LLMModel.GROQ_MIXTRAL_8X7B,
        # New models for expert comparison
        'meta-llama/llama-4-maverick-17b-128e-instruct': LLMModel.GROQ_LLAMA_4_MAVERICK_17B,
        'qwen/qwen3-32b': LLMModel.GROQ_QWEN3_32B,
        'deepseek-r1-distill-llama-70b': LLMModel.GROQ_DEEPSEEK_R1_DISTILL_70B,
        'o3-2025-04-16': LLMModel.O3_2025_04_16,
        'gpt-5-2025-08-07': LLMModel.GPT_5_2025_08_07,
        'o1-2024-12-17': LLMModel.O1_2024_12_17,
        'claude-3-7-sonnet-20250219': LLMModel.CLAUDE_3_7_SONNET,
    }
    model = model_map.get(model_config['name'].lower(), LLMModel.GPT_4O_2024_05_13)
    
    system_prompt = model_config.get('system_prompt', '')
    
    return LLMFactory.create_llm(provider, model, system_prompt=system_prompt)


def load_forecasts(config: dict, loader: ForecastDataLoader, llm=None):
    """Load initial forecasts based on configuration."""
    data_config = config['data']
    sampling_config = data_config['sampling']
    experiment_config = config['experiment']
    
    selected_resolution_date = data_config['resolution_date']
    initial_forecasts_path = experiment_config['initial_forecasts_dir']
    
    # Get questions based on sampling method
    questions_with_topic = loader.get_questions_with_topics()
    print(f"Total questions available: {len(questions_with_topic)}")
    
    # CRITICAL: Set random seed right before sampling to ensure reproducibility
    seed = experiment_config['seed']
    random.seed(seed)
    np.random.seed(seed)
    print(f"Set random seed to {seed} before question sampling")
    if sampling_config['method'] == 'by_topic':
        n_per_topic = sampling_config['n_per_topic']
        sampled_questions = sample_questions_by_topic(
            questions_with_topic, 
            n_per_topic=n_per_topic,
            seed=seed
        )
        print(f"Sampling method: by_topic ({n_per_topic} per topic)")
    elif sampling_config['method'] == 'random':
        # Random sampling with specified number of questions
        n_questions = sampling_config.get('n_questions', 10)
        sampled_questions = random.sample(questions_with_topic, min(n_questions, len(questions_with_topic)))
        print(f"Sampling method: random ({n_questions} questions)")
    elif sampling_config['method'] == 'first':
        # Take first N questions (deterministic)
        n_questions = sampling_config.get('n_questions', 10)
        sampled_questions = questions_with_topic[:n_questions]
        print(f"Sampling method: first ({n_questions} questions)")
    else:
        # Default fallback
        n_questions = sampling_config.get('n_questions', 10)
        sampled_questions = questions_with_topic[:n_questions]
        print(f"Sampling method: default ({n_questions} questions)")
    
    print(f"Questions after initial sampling: {len(sampled_questions)}")
    
    # Apply filters
    if data_config['filters']['require_resolution']:
        before_filter = len(sampled_questions)
        sampled_questions = [
            q for q in sampled_questions
            if loader.get_resolution(question_id=q.id, resolution_date=selected_resolution_date) is not None
        ]
        print(f"Questions after resolution filter: {len(sampled_questions)} (filtered out {before_filter - len(sampled_questions)})")
    
    # Process questions if needed
    for q in sampled_questions:
        pickle_path = f'{initial_forecasts_path}/collected_fcasts_with_examples_{selected_resolution_date}_{q.id}.pkl'
        if os.path.exists(pickle_path):
            if config['processing']['skip_existing']:
                print(f"Pickle for question {q.id} already exists, skipping.")
                continue
        
        print(f"Collecting forecasts for question {q.id}...")
        # Pass config and other parameters to run_all_forecasts_with_examples
        results = asyncio.run(run_all_forecasts_with_examples(
            [q],
            loader=loader,
            selected_resolution_date=selected_resolution_date,
            config=config,
            llm=llm
        ))
        os.makedirs(initial_forecasts_path, exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Collected forecasts for question {q.id}.")
    
    # Load pickled forecasts
    pkl_files = [
        f for f in os.listdir(f"{initial_forecasts_path}/")
        if f.startswith("collected_fcasts_with_examples") and f.endswith(".pkl") and f"{selected_resolution_date}" in f
    ]
    
    loaded_llmcasts = {}
    for fname in pkl_files:
        qid = fname[len(f"collected_fcasts_with_examples_{selected_resolution_date}_"): -len(".pkl")]
        with open(f"{initial_forecasts_path}/{fname}", "rb") as f:
            loaded_llmcasts[qid] = [q for q in pickle.load(f)]
    
    # Process loaded data - get actual Question objects from loader
    questions = []
    for qid, payloads in loaded_llmcasts.items():
        if payloads:
            # Get the actual Question object from the loader using the question ID
            question_obj = loader.get_question(qid)
            if question_obj:
                questions.append(question_obj)
            else:
                print(f"Warning: Could not find Question object for ID {qid}")
    
    llmcasts_by_qid_sfid = defaultdict(lambda: defaultdict(list))
    for qid, payloads in loaded_llmcasts.items():
        for i, p in enumerate(payloads):
            if isinstance(p, dict):
                # Use subject_id instead of superforecaster_id
                sfid = p.get("subject_id")
                if sfid is not None:
                    # The forecast data and conversation are in the payload
                    llmcasts_by_qid_sfid[qid][sfid].append({
                        'forecast': p.get('forecasts', []),  # forecasts is likely a list
                        'full_conversation': p.get('full_conversation', []),
                        'examples_used': p.get('examples_used', [])
                    })
    
    print(f"Loaded forecasts for {len(llmcasts_by_qid_sfid)} questions with experts")
    
    example_pairs_by_qid_sfid = defaultdict(lambda: defaultdict(list))
    for qid, payloads in loaded_llmcasts.items():
        for p in payloads:
            sfid = p.get("subject_id")
            if sfid is not None:
                example_pairs = p.get("examples_used", [])
                example_pairs_by_qid_sfid[qid][sfid].append(example_pairs)
    
    llmcasts_by_qid_sfid = {qid: dict(sfid_map) for qid, sfid_map in llmcasts_by_qid_sfid.items()}
    example_pairs_by_qid_sfid = {qid: dict(sfid_map) for qid, sfid_map in example_pairs_by_qid_sfid.items()}
    
    return questions, llmcasts_by_qid_sfid, example_pairs_by_qid_sfid


def run_delphi_experiment(config: dict):
    """Run the Delphi experiment based on configuration."""
    # Setup environment
    setup_environment(config)
    
    # Initialize components
    llm = get_llm_from_config(config)
    loader = ForecastDataLoader()
    
    # Load forecasts
    questions, llmcasts_by_qid_sfid, example_pairs_by_qid_sfid = load_forecasts(config, loader, llm)
    
    # Get configuration values
    n_rounds = config['delphi']['n_rounds']
    max_experts = config['delphi']['n_experts']
    output_dir = config['experiment']['output_dir']
    selected_resolution_date = config['data']['resolution_date']
    seed = config['experiment']['seed']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each question
    for question in questions:
        output_pattern = config['output']['file_pattern']
        output_file = os.path.join(
            output_dir, 
            output_pattern.format(
                question_id=question.id,
                resolution_date=selected_resolution_date
            )
        )
        
        if os.path.exists(output_file) and config['processing']['skip_existing']:
            print(f"Skipping {output_file} (already exists)")
            continue
        
        llmcasts_by_sfid = llmcasts_by_qid_sfid.get(question.id, {})
        example_pairs = example_pairs_by_qid_sfid.get(question.id, {})
        
        # Debug info (can be commented out in production)
        # print(f"Debug: Looking for question.id = '{question.id}'")
        # print(f"Debug: Available question IDs in llmcasts_by_qid_sfid: {list(llmcasts_by_qid_sfid.keys())}")
        # print(f"Debug: llmcasts_by_sfid for this question: {len(llmcasts_by_sfid)} experts")
        
        if not llmcasts_by_sfid:
            print(f"No forecasts found for question {question.id}, skipping")
            continue
        
        # Create experts with configuration
        model_config = config['model'].get('expert', config['model'])
        experts = {
            sfid: Expert(llm, config=model_config) 
            for sfid in llmcasts_by_sfid.keys()
        }
        
        # Populate experts with initial forecasts
        for sfid, payload_list in llmcasts_by_sfid.items():
            if not payload_list:
                continue
                
            # Get the first payload (there should only be one per subject usually)
            payload = payload_list[0]
            full_conversation = payload.get('full_conversation', [])
            
            if not full_conversation:
                print(f"No conversation found for subject {sfid}")
                continue
                
            initial_message = full_conversation[0]
            sampled_forecast_messages = full_conversation[1:]
            
            def _extract_final_prob(msg: str) -> float | None:
                import re
                m = list(re.finditer(r"FINAL\s+PROBABILITY:\s*([01](?:\.\d+)?|\.\d+)", msg, re.I))
                if not m:
                    return None
                p = float(m[-1].group(1))
                return max(0.0, min(1.0, p))
            
            pairs = [(p, m) for m in sampled_forecast_messages 
                    if (p := _extract_final_prob(m['content'])) is not None]
            if not pairs:
                print(f"No valid forecasts found for subject {sfid}")
                continue
            
            pairs.sort(key=lambda x: x[0])
            median_llmcast, median_message = pairs[len(pairs) // 2]
            
            conversation = [initial_message, median_message]
            experts[sfid].conversation_manager.add_messages(conversation)
            print(f"Initialized expert {sfid} with conversation")
        
        # Filter experts with conversations
        experts = {sfid: expert for sfid, expert in experts.items() 
                  if expert.conversation_manager.messages}
        
        # Select experts based on configuration
        if len(experts) > max_experts:
            if config['delphi']['expert_selection'] == 'random':
                # Use the same seed for expert selection to ensure reproducibility
                expert_rng = random.Random(seed)
                selected_sfs = expert_rng.sample(list(experts.keys()), max_experts)
                experts = {sfid: experts[sfid] for sfid in selected_sfs}
                print(f"Randomly selected {max_experts} experts from {len(experts) + (max_experts - len(experts))} available")
        
        print(f"Running Delphi for question {question.id} with {len(experts)} experts")
        
        # Use centralized probability extraction
        from probability_parser import extract_final_probability as _extract_prob
        
        # Initialize mediator with configuration
        mediator_config = config['model'].get('mediator', config['model'])
        
        # Check if mediator uses a different model
        mediator_model = mediator_config.get('model')
        base_model = config['model']['name']
        
        if mediator_model and mediator_model != base_model:
            # Create separate LLM instance for mediator
            mediator_model_config = config['model'].copy()
            mediator_model_config['name'] = mediator_model
            mediator_llm = get_llm_from_config({'model': mediator_model_config})
            mediator = Mediator(mediator_llm, config=mediator_config)
        else:
            # Use same LLM instance
            mediator = Mediator(llm, config=mediator_config)
        
        # Initialize structured log
        delphi_log = {
            "question_id": question.id,
            "question_text": question.question,
            "question_background": question.background,
            "question_url": getattr(question, 'url', ''),
            "config": {
                "seed": seed,
                "n_rounds": n_rounds,
                "n_experts": max_experts,
                "resolution_date": selected_resolution_date,
                "model_provider": config['model']['provider'],
                "model_name": config['model']['name']
            },
            "rounds": [],
            "histories": None
        }
        
        # Round 0: capture initial responses
        initial_expert_entries = {}
        for sfid, expert in experts.items():
            resp = expert.get_last_response()
            try:
                initial_expert_entries[sfid] = {
                    "prob": _extract_prob(resp['content']),
                    "response": resp,
                }
            except Exception as e:
                # breakpoint()
                print("you gotta fix this --------------------------")
                pass
        delphi_log["rounds"].append({
            "round": 0,
            "mediator_feedback": "Initial forecasts collected.",
            "experts": initial_expert_entries,
        })
        
        # Run Delphi rounds
        for round_idx in range(n_rounds):
            print(f"Round {round_idx + 1} for question {question.id}")
            
            # Mediator processing
            mediator.start_round(round_idx=round_idx, question=question)
            expert_messages = {
                sfid: {"role": "assistant", "content": entry["response"]}
                for sfid, entry in delphi_log["rounds"][-1]["experts"].items()
            }
            mediator.receive_messages(expert_messages)
            
            # Generate feedback
            feedback_message = asyncio.run(mediator.generate_feedback(round_idx=round_idx))
            print(f"Mediator feedback: {feedback_message}")  # Print preview
            update_instruction = (
                "Consider the arguments of the other experts and reason over how they interact with your own reasoning, then "
                "provide your final probability estimate.\n"
                "You should reason through the problem, but you MUST end your response with:\n"
                "FINAL PROBABILITY: [your decimal number between 0 and 1]"
            )

            broadcast_msg = f"{feedback_message}\n\n{update_instruction}"
            broadcast_msg = feedback_message
            
            # Expert updates
            round_expert_entries = {}
            for sfid, expert in experts.items():
                prob, response = asyncio.run(expert.get_forecast_update(broadcast_msg))
                print(f"Expert {sfid} prob: {prob}")
                round_expert_entries[sfid] = {
                    "prob": prob,
                    "response": response,
                }
            delphi_log["rounds"].append({
                "round": round_idx + 1,
                "mediator_feedback": broadcast_msg,
                "experts": round_expert_entries,
            })
        
        # Save conversation histories if configured
        if config['output']['save']['conversation_histories']:
            delphi_log["histories"] = {
                "mediator": list(mediator.conversation_manager.messages),
                "experts": {sfid: list(expert.conversation_manager.messages) 
                          for sfid, expert in experts.items()},
            }
        
        # Save example pairs if configured
        if config['output']['save']['example_pairs']:
            try:
                delphi_log["example_pairs"] = make_json_serializable(example_pairs)
            except Exception as e:
                print(f"Warning: Could not serialize example_pairs: {e}")
                delphi_log["example_pairs"] = {"error": "Failed to serialize example pairs"}
        
        # Save the log
        try:
            with open(output_file, "w") as f:
                json.dump(delphi_log, f, indent=2)
        except TypeError as e:
            print(f"JSON serialization error: {e}")
            
            # Test each part of the log to identify the problematic object
            for key, value in delphi_log.items():
                try:
                    json.dumps(value)
                    print(f"  {key}: OK")
                except TypeError as sub_e:
                    print(f"  {key}: ERROR - {sub_e}")
                    if key == "rounds":
                        for i, round_data in enumerate(value):
                            try:
                                json.dumps(round_data)
                                print(f"    round {i}: OK")
                            except TypeError as round_e:
                                print(f"    round {i}: ERROR - {round_e}")
            raise e
        
        print(f"Delphi log saved to {output_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Delphi experiment")
    parser.add_argument("config_path", help="Path to experiment configuration YAML file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_experiment_config(args.config_path)
    
    # Run experiment
    run_delphi_experiment(config)