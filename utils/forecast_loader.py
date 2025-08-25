import asyncio
import copy
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import yaml

from dataset.dataloader import ForecastDataLoader

from utils.llm_config import get_llm_from_config
from utils.sampling import sample_questions


def generate_initial_forecasts_for_questions(questions, initial_forecasts_path, config, resolution_date):
    """Generate initial forecasts for the given questions and save them to the specified path."""
    os.makedirs(initial_forecasts_path, exist_ok=True)
    
    initial_config = copy.deepcopy(config)
    initial_config['experiment']['seed'] = 42
    
    print(f"🔧 Generating initial forecasts for {len(questions)} questions using fixed seed 42...")
    
    from dataset.dataloader import ForecastDataLoader
    from icl_initial_forecasts import run_all_forecasts_with_examples
    
    loader = ForecastDataLoader()
    llm = get_llm_from_config(initial_config)
    
    for question in questions:
        print(f"  📊 Generating initial forecasts for question {question.id[:8]}...")
        
        try:
            results = asyncio.run(run_all_forecasts_with_examples(
                [question],
                loader=loader,
                selected_resolution_date=resolution_date,
                config=initial_config,
                llm=llm
            ))
            
            pickle_filename = f'collected_fcasts_with_examples_{resolution_date}_{question.id}.pkl'
            pickle_path = os.path.join(initial_forecasts_path, pickle_filename)
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"  ✅ Saved initial forecasts for question {question.id[:8]} to {pickle_filename}")
            
        except Exception as e:
            print(f"  ❌ Failed to generate initial forecasts for question {question.id[:8]}: {e}")
            continue
    
    print(f"✅ Initial forecasts generation completed in {initial_forecasts_path}")


def find_matching_initial_forecasts_dir(current_config: dict) -> str:
    """
    Find an initial forecasts directory that matches the current configuration
    (ignoring seed differences). Returns the path to the initial forecasts directory.
    """
    results_path = Path("results")
    if not results_path.exists():
        return None
    
    target_model = current_config.get('model', {}).get('name', '')
    target_n_experts = current_config.get('delphi', {}).get('n_experts', 0)
    
    sweep_dirs = sorted([d for d in results_path.iterdir() if d.is_dir()],
                       key=lambda x: x.name, reverse=True)
    for sweep_dir in sweep_dirs:
        config_files = list(sweep_dir.glob("config_*.yml"))
        
        for config_file in config_files:
            with open(config_file, 'r') as f:
                other_config = yaml.safe_load(f)
            
            other_model = other_config.get('model', {}).get('name', '')
            other_n_experts = other_config.get('delphi', {}).get('n_experts', 0)
            
            if (other_model == target_model and 
                other_n_experts == target_n_experts):
                
                config_base = config_file.stem.replace('config_', '')
                possible_initial_dir = sweep_dir / f"results_{config_base}_initial"
                if possible_initial_dir.exists():
                    pkl_files = list(possible_initial_dir.glob("*.pkl"))
                    if pkl_files:
                        print(f"🔍 Found matching initial forecasts in: {possible_initial_dir}")
                        print(f"    (matched config: {config_file.name})")
                        return str(possible_initial_dir)
                    
    print("❌ No matching initial forecasts directory found")
    return None


def load_pickled_forecasts(initial_forecasts_path: str, selected_resolution_date: str, loader) -> Tuple[List, Dict, Dict]:
    """Load pickled forecasts from the specified directory."""
    pkl_files = [
        f for f in os.listdir(f"{initial_forecasts_path}/")
        if f.startswith("collected_fcasts_with_examples") and f.endswith(".pkl") and f"{selected_resolution_date}" in f
    ]
    
    loaded_llmcasts = {}
    for fname in pkl_files:
        qid = fname[len(f"collected_fcasts_with_examples_{selected_resolution_date}_"): -len(".pkl")]
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
                    llmcasts_by_qid_sfid[qid][sfid].append({
                        'forecast': p.get('forecasts', []),
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


def load_forecast_pickles(initial_forecasts_path, selected_resolution_date):
    """Load all forecast pickle files and return them organized by type."""
    pkl_files = [
        f for f in os.listdir(initial_forecasts_path)
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

    # Load with_examples forecasts
    loaded_fcasts_with_examples = {}
    for fname in with_examples_files:
        qid = fname[len(f"collected_fcasts_with_examples_{selected_resolution_date}_"): -len(".pkl")]
        with open(os.path.join(initial_forecasts_path, fname), "rb") as f:
            loaded_fcasts_with_examples[qid] = [q for q in pickle.load(f)]

    # Load no_examples forecasts
    loaded_fcasts_no_examples = {}
    for fname in no_examples_files:
        qid = fname[len(f"collected_fcasts_no_examples_{selected_resolution_date}_"): -len(".pkl")]
        with open(os.path.join(initial_forecasts_path, fname), "rb") as f:
            loaded_fcasts_no_examples[qid] = [q for q in pickle.load(f)]

    return loaded_fcasts_with_examples, loaded_fcasts_no_examples


async def load_forecasts(config: dict, loader: ForecastDataLoader, llm=None):
    """Load initial forecasts based on configuration."""
    data_config = config['data']
    experiment_config = config['experiment']

    selected_resolution_date = data_config['resolution_date']
    initial_forecasts_path = experiment_config['initial_forecasts_dir']

    # Determine if we're reusing existing forecasts
    reuse_config = experiment_config.get('reuse_initial_forecasts', {})
    if reuse_config.get('enabled', False):
        source_dir = reuse_config.get('source_dir', 'auto')
        if source_dir == 'auto':
            source_dir = find_matching_initial_forecasts_dir(config)

        if source_dir and os.path.exists(source_dir):
            print(f"🔄 Reusing initial forecasts from: {source_dir}")
            initial_forecasts_path = source_dir
        else:
            print(f"🔍 No matching forecasts found. Generating new ones in: {initial_forecasts_path}")
            reuse_config['enabled'] = False  # Disable reuse to trigger generation

    # Get and sample questions
    questions_with_topic = loader.get_questions_with_topics()
    print(f"Total questions available: {len(questions_with_topic)}")
    sampled_questions = sample_questions(config, questions_with_topic, loader)

    # Generate initial forecasts if not reusing
    if not reuse_config.get('enabled', False):
        os.makedirs(initial_forecasts_path, exist_ok=True)
        from icl_initial_forecasts import run_all_forecasts_with_examples
        for q in sampled_questions:
            pickle_path = f'{initial_forecasts_path}/collected_fcasts_with_examples_{selected_resolution_date}_{q.id}.pkl'
            if os.path.exists(pickle_path) and config['processing']['skip_existing']:
                print(f"Pickle for question {q.id} already exists, skipping.")
                continue

            print(f"Collecting forecasts for question {q.id}...")
            results = await run_all_forecasts_with_examples(
                [q], loader=loader, selected_resolution_date=selected_resolution_date,
                config=config, llm=llm
            )
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
    else:
        print(f"📁 Skipping initial forecast collection (reusing from {initial_forecasts_path})")

    # Load and return the forecasts
    return load_pickled_forecasts(initial_forecasts_path, selected_resolution_date, loader)

