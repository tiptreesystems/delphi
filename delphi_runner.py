import asyncio
import random
from typing import Dict, Any

from agents.mediator import Mediator
from agents.expert import Expert
from utils.llm_config import get_llm_from_config
from utils.expert_utils import (
    find_median_forecast,
    create_delphi_log,
    add_initial_round,
    add_round_to_log,
    finalize_delphi_log
)


def initialize_experts(llmcasts_by_sfid: Dict, config: dict, llm) -> Dict[str, Expert]:
    """Initialize experts with their initial forecasts."""
    model_config = config['model'].get('expert', config['model'])
    experts = {
        sfid: Expert(llm, config=model_config)
        for sfid in llmcasts_by_sfid.keys()
    }
    for sfid, payload_list in llmcasts_by_sfid.items():
        if not payload_list:
            raise ValueError(
                f"Expert {sfid} has empty payload list. This indicates initial forecasts "
                f"were not properly generated or loaded."
            )

        payload = payload_list[0]
        full_conversation = payload.get('full_conversation', [])

        if not full_conversation:
            raise ValueError(
                f"Expert {sfid} has no conversation history in payload. "
                f"Payload keys: {list(payload.keys())}. "
                f"This indicates a problem with initial forecast generation."
            )

        initial_message = full_conversation[0]
        sampled_forecast_messages = full_conversation[1:]

        if not sampled_forecast_messages:
            raise ValueError(
                f"Expert {sfid} has no forecast messages (only initial message). "
                f"Expected at least one forecast response."
            )

        median_result = find_median_forecast(sampled_forecast_messages)
        if not median_result:
            raise ValueError(
                f"Expert {sfid} has no valid median forecast. "
                f"Sampled messages: {len(sampled_forecast_messages)}. "
                f"This indicates none of the forecasts contain valid probabilities."
            )

        median_llmcast, median_message = median_result

        # Ensure median_message has correct role
        if not isinstance(median_message, dict):
            raise ValueError(
                f"Expert {sfid} median message is not a dict (got {type(median_message)}). "
                f"Expected message dictionary with 'role' and 'content' fields."
            )

        # Fix role if missing or incorrect
        if median_message.get('role') != 'assistant':
            median_message = {
                **median_message,
                'role': 'assistant'
            }

        conversation = [initial_message, median_message]
        experts[sfid].conversation_manager.add_messages(conversation)

    # Verify all experts are properly initialized
    for sfid, expert in experts.items():
        if not expert.conversation_manager.messages:
            raise ValueError(
                f"Expert {sfid} has no messages after initialization. "
                f"This should never happen if initialization succeeded."
            )

        # Verify the expert has at least one assistant message
        has_assistant_msg = any(
            msg.get('role') == 'assistant'
            for msg in expert.conversation_manager.messages
        )

        if not has_assistant_msg:
            raise ValueError(
                f"Expert {sfid} has no assistant messages after initialization. "
                f"Messages: {[msg.get('role') for msg in expert.conversation_manager.messages]}"
            )

        # Verify get_last_response works
        last_response = expert.get_last_response()
        if last_response is None:
            raise ValueError(
                f"Expert {sfid} get_last_response() returns None despite having messages. "
                f"This indicates a bug in the Expert.get_last_response() method."
            )

    return experts


def select_experts(experts: Dict[str, Expert], config: dict) -> Dict[str, Expert]:
    """Select experts based on configuration."""
    max_experts = config['delphi']['n_experts']

    if len(experts) <= max_experts:
        return experts

    if config['delphi']['expert_selection'] == 'random':
        seed = config['experiment']['seed']
        expert_selection_seed = config['delphi'].get('expert_selection_seed', seed)
        expert_rng = random.Random(expert_selection_seed)
        selected_sfs = expert_rng.sample(list(experts.keys()), max_experts)
        return {sfid: experts[sfid] for sfid in selected_sfs}

    return experts


async def _run_expert_updates(experts: Dict[str, Expert], feedback_message: str) -> Dict[str, Dict]:
    """Run expert updates in parallel."""
    tasks = {
        sfid: expert.get_forecast_update(feedback_message)
        for sfid, expert in experts.items()
    }
    results = await asyncio.gather(*tasks.values())

    return {
        sfid: {"prob": prob, "response": response}
        for sfid, (prob, response) in zip(tasks.keys(), results)
    }


async def run_delphi_rounds(question, experts: Dict[str, Expert], config: dict, example_pairs: Dict) -> Dict[str, Any]:
    """Run the Delphi rounds for a single question."""
    # Setup
    mediator_config = config['model'].get('mediator', config['model'])
    mediator_llm = get_llm_from_config(config, role='mediator')
    mediator = Mediator(mediator_llm, config=mediator_config)

    # Initialize log
    delphi_log = create_delphi_log(
        question, config,
        config['experiment']['seed'],
        config['data']['resolution_date']
    )
    add_initial_round(delphi_log, experts)

    # Core Delphi loop
    for round_idx in range(config['delphi']['n_rounds']):
        # Prepare mediator with previous round's responses
        mediator.start_round(round_idx=round_idx, question=question)
        expert_messages = {
            sfid: {"role": "assistant", "content": entry["response"]}
            for sfid, entry in delphi_log["rounds"][-1]["experts"].items()
        }
        mediator.receive_messages(expert_messages)

        # Generate feedback and get expert updates
        feedback_message = await mediator.generate_feedback()
        expert_updates = await _run_expert_updates(experts, feedback_message)

        # Record round results
        add_round_to_log(delphi_log, round_idx, feedback_message, expert_updates)

    # Finalize with optional data
    return finalize_delphi_log(delphi_log, config, mediator, experts, example_pairs)
