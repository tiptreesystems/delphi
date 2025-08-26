"""Utilities for expert initialization and management."""
from typing import Dict, List, Optional, Any
from utils.probability_parser import extract_final_probability
from utils.utils import make_json_serializable


def find_median_forecast(sampled_forecast_messages: List[Dict]) -> Optional[tuple[float, Dict]]:
    """Find the median forecast from a list of sampled messages.
    
    Args:
        sampled_forecast_messages: List of message dictionaries with 'content' key
        
    Returns:
        Tuple of (median_probability, median_message) or None if no valid forecasts
    """
    pairs = [(p, m) for m in sampled_forecast_messages 
            if (p := extract_final_probability(m['content'])) != -1]
    if not pairs:
        return None
    
    pairs.sort(key=lambda x: x[0])
    return pairs[len(pairs) // 2]


def create_delphi_log(question, config: dict, seed: int, selected_resolution_date: str) -> Dict:
    """Create the initial structure for a Delphi log.
    
    Args:
        question: Question object with id, question, background, url attributes
        config: Configuration dictionary
        seed: Random seed value
        selected_resolution_date: Resolution date string
        
    Returns:
        Dictionary with initialized Delphi log structure
    """
    return {
        "question_id": question.id,
        "question_text": question.question,
        "question_background": question.background,
        "question_url": getattr(question, 'url', ''),
        "config": {
            "seed": seed,
            "n_rounds": config['delphi']['n_rounds'],
            "n_experts": config['delphi']['n_experts'],
            "resolution_date": selected_resolution_date,
            "model_provider": config['model']['provider'],
            "model_name": config['model']['name'],
            "expert_model_name": config['model']['expert']['model'],
            "mediator_model_name": config['model']['mediator']['model']
        },
        "rounds": [],
        "histories": None
    }


def create_expert_entry(expert, sfid: str) -> Dict:
    """Create an expert entry for the Delphi log.
    
    Args:
        expert: Expert object with get_last_response method
        sfid: Expert identifier
        
    Returns:
        Dictionary with expert's probability and response
    
    Raises:
        ValueError: If expert has no valid response
    """
    resp = expert.get_last_response()
    
    if resp is None:
        raise ValueError(
            f"Expert {sfid} has no last response. This should never happen - "
            f"expert initialization must have failed. Check that initial forecasts "
            f"were properly loaded and conversation history was set."
        )
    
    if not isinstance(resp, dict):
        raise TypeError(
            f"Expert {sfid} response is not a dict (got {type(resp)}). "
            f"Expected format: {{'role': 'assistant', 'content': '...'}}"
        )
    
    if 'content' not in resp:
        raise KeyError(
            f"Expert {sfid} response missing 'content' field. "
            f"Got keys: {list(resp.keys())}"
        )
    
    prob = extract_final_probability(resp['content'])
    
    if prob == -1:
        raise ValueError(
            f"Expert {sfid} response does not contain a valid FINAL PROBABILITY. "
            f"Content: {resp['content'][:200]}..."
        )
    
    return {
        "prob": prob,
        "response": resp,
    }


def finalize_delphi_log(delphi_log: Dict, config: Dict, mediator, experts: Dict, 
                        example_pairs: Optional[Dict] = None) -> Dict:
    """Add optional conversation histories and example pairs to the Delphi log.
    
    Args:
        delphi_log: The main Delphi log dictionary
        config: Configuration dictionary
        mediator: Mediator object with conversation_manager
        experts: Dictionary of Expert objects
        example_pairs: Optional example pairs to serialize
        
    Returns:
        Updated Delphi log with optional additions
    """
    
    if config['output']['save']['conversation_histories']:
        delphi_log["histories"] = {
            "mediator": list(mediator.conversation_manager.messages),
            "experts": {sfid: list(expert.conversation_manager.messages) 
                      for sfid, expert in experts.items()},
        }
    
    if config['output']['save']['example_pairs'] and example_pairs:
        try:
            delphi_log["example_pairs"] = make_json_serializable(example_pairs)
        except Exception:
            delphi_log["example_pairs"] = {"error": "Failed to serialize example pairs"}
    
    return delphi_log


def add_round_to_log(delphi_log: Dict, round_idx: int, feedback_message: str, 
                     expert_entries: Dict) -> None:
    """Add a round's results to the Delphi log.
    
    Args:
        delphi_log: The main Delphi log dictionary (modified in place)
        round_idx: Round index (0-based, but displayed as 1-based)
        feedback_message: Mediator's feedback message
        expert_entries: Dictionary of expert responses
    """
    delphi_log["rounds"].append({
        "round": round_idx + 1,
        "mediator_feedback": feedback_message,
        "experts": expert_entries,
    })


def add_initial_round(delphi_log: Dict, experts: Dict) -> None:
    """Add the initial round (round 0) to the Delphi log.
    
    Args:
        delphi_log: The main Delphi log dictionary (modified in place)
        experts: Dictionary of Expert objects
    """
    initial_expert_entries = {
        sfid: create_expert_entry(expert, sfid) 
        for sfid, expert in experts.items()
    }
    delphi_log["rounds"].append({
        "round": 0,
        "mediator_feedback": "Initial forecasts collected.",
        "experts": initial_expert_entries,
    })
