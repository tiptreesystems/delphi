import json
import os
import random
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import yaml


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
    # if config.get('debug', {}).get('enabled', False):
    #     # check if "williaar" is in the file path for this file
    #     if "williaar" in __file__:
    #         import debugpy
    #         print(f"Waiting for debugger attach on port {config['debug']['debugpy_port']}...")
    #         debugpy.listen(config['debug']['debugpy_port'])
    #         debugpy.wait_for_client()
    #         print("Debugger attached.")

    if config.get('debug', {}).get('breakpoint_on_start', False):
        breakpoint()

    # Setup API keys
    api_config = config.get('api', {})
    if 'openai' in api_config:
        openai_key = os.getenv(api_config['openai']['api_key_env'])
        os.environ["OPENAI_API_KEY"] = openai_key
    if 'groq' in api_config:
        groq_key = os.getenv(api_config['groq']['api_key_env'])
        os.environ["GROQ_API_KEY"] = groq_key


def split_train_valid(questions, valid_ratio: float = 0.2, seed: int = 42) -> Tuple[List, List]:
    """Split questions into train and validation sets with stratification by topic."""
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Group questions by topic
    topic_questions = defaultdict(list)
    for q in questions:
        topic = q.topic if q.topic else 'unknown'
        topic_questions[topic].append(q)
    
    train_questions = []
    valid_questions = []
    
    # Split each topic proportionally
    for topic, topic_qs in topic_questions.items():
        n_valid = max(1, int(len(topic_qs) * valid_ratio))
        n_train = len(topic_qs) - n_valid
        
        # Shuffle within topic for random split
        shuffled = topic_qs.copy()
        random.shuffle(shuffled)
        
        train_questions.extend(shuffled[:n_train])
        valid_questions.extend(shuffled[n_train:])
    
    # Shuffle final lists
    random.shuffle(train_questions)
    random.shuffle(valid_questions)
    
    print(f"\nData split (stratified by topic):")
    print(f"  Training set: {len(train_questions)} questions")
    print(f"  Validation set: {len(valid_questions)} questions")
    
    return train_questions, valid_questions
