import os
from typing import List, Tuple, Optional, Dict
from models import LLMFactory, LLMProvider, LLMModel, BaseLLM, ConversationManager
from dataset.dataloader import ForecastDataLoader, Question, Forecast
import random
import json
import dotenv
import numpy as np
import asyncio
from prompt_loader import load_prompt, get_prompt_loader
from probability_parser import extract_final_probability, extract_final_probability_with_retry

dotenv.load_dotenv()

# Load prompts
prompt_loader = get_prompt_loader()

class Expert:
    def __init__(self, llm: BaseLLM, user_profile: Optional[dict] = None, config: Optional[dict] = None):
        self.llm = llm
        self.user_profile = user_profile
        self.config = config or {}
        self.conversation_manager = ConversationManager(llm)
        self.token_warnings = []  # Track token usage warnings
        self.retry_count = 0  # Track retries for token issues

    async def forecast(self, question: Question, conditioning_forecast: Optional[Forecast] = None, seed: Optional[int] = None) -> float:
        # Add the user's actual forecast for this question if available
        prior_forecast_info = ""
        if conditioning_forecast:
            prior_forecast_info = (
                f"Some of your notes on this question are: {conditioning_forecast.reasoning}\n"
            )
        # Get prompt version from config, default to 'v1'
        prompt_version = self.config.get('prompt_version', 'v1')
        
        prompt = load_prompt(
            'expert_forecast',
            prompt_version,
            question=question.question,
            background=question.background,
            resolution_criteria=question.resolution_criteria,
            url=question.url,
            freeze_datetime_value=question.freeze_datetime_value,
            prior_forecast_info=prior_forecast_info
        )
        temperature = self.config.get('temperature', 0.3)
        self.conversation_manager.messages.clear()
        
        # Pass seed if provided for deterministic outputs
        kwargs = {'max_tokens': 500, 'temperature': temperature}
        if seed is not None:
            kwargs['seed'] = seed
            
        response = await self.conversation_manager.generate_response(prompt, **kwargs)
        response = response.strip()
        
        prob = await extract_final_probability_with_retry(
            response, 
            self.conversation_manager, 
            max_retries=1
        )
        if prob != -1:
            self.retry_count = 0  # Reset retry count on success
            return prob

        # Fallback if no valid number found - this suggests a real issue
        print(f"❌ No valid probability found in response. Response: '{response[:200]}...'")
        self.token_warnings.append(f"No valid probability found in response of {len(response)} chars")
        return -1

    async def forecast_with_examples_in_context(self, question: Question, examples: List[Tuple[Question, Forecast]]) -> float:
        examples_text = ""

        for i, (ex_q, ex_f) in enumerate(examples):
            examples_text += (
                f"Example {i+1}:\n"
                f"Question: {ex_q.question}\n"
                f"Background: {ex_q.background}\n"
                f"Resolution criteria: {ex_q.resolution_criteria}\n"
                f"Forecast: {ex_f.forecast}\n"
                f"Rationale: {ex_f.reasoning}\n"
                f"FINAL PROBABILITY: {ex_f.forecast}\n\n"
            )

        examples_text += "--------------------------------\n\n"

        prompt = load_prompt(
            'expert_forecast',
            'v1_with_examples',
            examples_text=examples_text,
            question=question.question,
            background=question.background,
            resolution_criteria=question.resolution_criteria,
            url=question.url,
            freeze_datetime_value=question.freeze_datetime_value,
            freeze_datetime_value_explanation=question.freeze_datetime_value_explanation
        )

        temperature = self.config.get('temperature', 0.3)
        self.conversation_manager.messages.clear()
        response = await self.conversation_manager.generate_response(prompt, max_tokens=self.config.get('max_tokens', 500), temperature=temperature)
        response = response.strip()
        # Check for token usage issues

        prob = await extract_final_probability_with_retry(
            response, 
            self.conversation_manager, 
            max_retries=3
        )

        if prob != -1:
            return prob
            
        print(f"❌ No valid probability in examples response: '{response[:200]}...'")
        return -1

    async def get_forecast_update(self, input_message) -> float:
        """Get a response without clearing the conversation, used after feedback."""
        if not self.conversation_manager.messages:
            raise RuntimeError("No conversation history found. Cannot update forecast without prior context.")
        
        response = await self.conversation_manager.generate_response(input_message, max_tokens=self.config.get('max_tokens', 800), temperature=self.config.get('temperature', 0.3))
        response = response.strip()
        # Extract the final probability
        # prob = extract_final_probability(response)
        prob = await extract_final_probability_with_retry(
            response, 
            self.conversation_manager, 
            max_retries=1
        )

        if prob != -1:
            return prob, response
        return -1, response

    def get_last_response(self) -> Optional[str]:
        """
        Returns the content of the most recent assistant message in the conversation,
        or None if there is no assistant message.
        """
        if not self.conversation_manager.messages:
            return None
        for msg in reversed(self.conversation_manager.messages):
            if msg.get("role") == "assistant":
                return msg
        return None


class Mediator:
    """
    Minimal Delphi Mediator:
    - holds its own ConversationManager
    - receives & stores expert messages
    - crafts a single feedback memo
    """

    def __init__(self, llm: BaseLLM, config: Optional[dict] = None):
        self.llm = llm
        self.config = config or {}
        self.conversation_manager = ConversationManager(llm)
        default_mediator_prompt = load_prompt('mediator_system', 'v1')
        self.system_prompt = self.config.get(
            "system_prompt",
            default_mediator_prompt
        )
        self.max_tokens = self.config.get("feedback_max_tokens", 800)
        self.temperature = self.config.get("feedback_temperature", 0.2)

        # Local store of messages (expert_id -> text)
        self.expert_messages: Dict[str, str] = {}

    # ---- message intake ----

    def reset(self) -> None:
        """Clear mediator and conversation state."""
        self.expert_messages.clear()
        self.conversation_manager.messages.clear()

    def receive_message(self, *, expert_id: str, message: Dict[str, str]) -> None:
        """
        Add/replace a single expert's message.
        Accepts a message dict with keys 'role' and 'content'.
        """
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError("Message must be a dict with at least a 'content' key.")
        self.expert_messages[expert_id] = message["content"]

    def receive_messages(self, messages: Dict[str, Dict[str, str]]) -> None:
        """
        Bulk add/replace expert messages.
        Accepts a dict mapping expert_id -> message dict with keys 'role' and 'content'.
        """
        for expert_id, message in messages.items():
            if not isinstance(message, dict) or "content" not in message:
                raise ValueError(f"Message for expert {expert_id} must be a dict with at least a 'content' key.")
            self.expert_messages[expert_id] = message["content"]

    # ---- feedback ----
    def start_round(self, *, round_idx: int, question: Question, extra_context: Optional[str] = None) -> None:
        """Append a new round header + question context without clearing prior messages."""
        # Seed a system message once if none exists yet (but not for Claude, which handles system separately)
        from models import ClaudeLLM
        if not isinstance(self.llm, ClaudeLLM) and not any(m["role"] == "system" for m in self.conversation_manager.messages):
            self.conversation_manager.add_message(role="system", content=self.system_prompt)

        header = [f"=== DELPHI ROUND {round_idx} ==="]
        q_block = (
            f"QUESTION: {question.question}\n"
            f"BACKGROUND: {question.background}\n"
            f"RESOLUTION CRITERIA: {question.resolution_criteria}\n"
        )
        if getattr(question, "url", None):
            q_block += f"URL: {question.url}\n"
        if getattr(question, "freeze_datetime_value", None):
            q_block += f"MARKET FREEZE VALUE: {question.freeze_datetime_value}\n"
        if getattr(question, "freeze_datetime_value_explanation", None):
            q_block += f"MARKET FREEZE VALUE EXPLANATION: {question.freeze_datetime_value_explanation}\n"
        if extra_context:
            q_block += f"\nADDITIONAL CONTEXT:\n{extra_context}\n"

        self.conversation_manager.add_message(role="user", content="\n".join([*header, q_block]))


    async def generate_feedback(
        self,
        *,
        round_idx: Optional[int] = None,
        reset: bool = False
    ) -> str:
        """
        Append a feedback request based on all accumulated expert messages & prior rounds.
        Set reset=True to start a fresh thread (rare).
        """
        if reset:
            self.conversation_manager.messages.clear()

        # Ensure there is a system message (but not for Claude, which handles system separately)
        from models import ClaudeLLM
        if not isinstance(self.llm, ClaudeLLM) and not any(m["role"] == "system" for m in self.conversation_manager.messages):
            self.conversation_manager.add_message(role="system", content=self.system_prompt)

        # Append expert messages for this turn (anonymized, deterministic order)
        if self.expert_messages:
            # messages = []
            # for eid in sorted(self.expert_messages.keys()):
            #     message = self.expert_messages[eid]
            #     content = message['content'] if isinstance(message, dict) else message
            #     messages.append(f"[EXPERT {eid}] {content}\n\n")

            # self.conversation_manager.add_message(
            #     role="user",
            #     content=f"Here are the messages from the experts: {''.join(messages)}"
            # )
            for eid in sorted(self.expert_messages.keys()):
                message = self.expert_messages[eid]
                content = message['content'] if isinstance(message, dict) else message

                self.conversation_manager.add_message(
                    role="user",
                    content=f"[EXPERT {eid}] {content}"
                )

        # Append the mediator synthesis request (annotate round if provided)
        prefix = f"(Round {round_idx}) " if round_idx is not None else ""
        mediator_request = load_prompt(
            'mediator_feedback',
            'v1',
            prefix=prefix
        )
        self.conversation_manager.add_message(role="user", content=mediator_request)

        feedback = await self.conversation_manager.generate_response(
            mediator_request,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return feedback.strip()


class DelphiPanel:
    def __init__(
        self,
        loader: ForecastDataLoader,
        n_experts: int = 3,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: LLMModel = LLMModel.GPT_4O,
        system_prompt: Optional[str] = None,
        condition_on_data: bool = True,
        config: Optional[dict] = None
    ):
        self.loader = loader
        self.condition_on_data = condition_on_data
        self.config = config or {}
        
        # Load default system prompt if not provided
        if system_prompt is None:
            base_template = load_prompt('expert_system', 'v1')
            # Use just the base part without the user_profile placeholder for default
            system_prompt = base_template.replace('{user_profile}', '')

        # Load user profiles
        with open('user_profiles.json', 'r') as f:
            user_profiles = json.load(f)

        # Get all unique user IDs that have both profiles and forecasts
        all_user_ids = set()
        for forecasts in loader.super_forecasts.values():
            for forecast in forecasts:
                if forecast.user_id in user_profiles:
                    all_user_ids.add(forecast.user_id)

        # Create a pool of all possible experts
        self.expert_pool = {}  # user_id -> Expert
        self.n_experts = n_experts  # Store for later use

        for user_id in all_user_ids:
            user_profile = user_profiles[user_id]

            # Create personalized system prompt that includes the user profile
            expert_system_template = load_prompt('expert_system', 'v1')
            personalized_system_prompt = expert_system_template.format(
                user_profile=f" You should embody the following profile:\n{user_profile['expertise_profile']}\n"
            )

            llm = LLMFactory.create_llm(provider, model, system_prompt=personalized_system_prompt)
            expert = Expert(llm, user_profile=user_profile, config=self.config.get('model', {}))
            self.expert_pool[user_id] = expert

    async def forecast_question(self, question_id: str) -> dict:
        question = self.loader.get_question(question_id)
        if not question:
            raise ValueError(f"Question {question_id} not found")

        # Get all forecasts for this question
        question_forecasts = self.loader.get_super_forecasts(question_id=question_id)

        # Find experts who have forecasts with reasoning for this specific question
        available_experts = []
        expert_forecasts = {}

        for user_id, expert in self.expert_pool.items():
            for forecast in question_forecasts:
                if forecast.user_id == user_id and forecast.reasoning and forecast.reasoning.strip():
                    available_experts.append((expert, user_id))
                    expert_forecasts[user_id] = forecast
                    break

        if not available_experts:
            print(f"No experts found with reasoning for question {question_id[:8]}...")
            return {
                "aggregate": -1,
                "individual_forecasts": [],
                "selected_experts": [],
                "round1_responses": [],
                "round2_responses": [],
                "expert_details": []
            }

        # Sort available experts by user_id for consistent ordering before sampling
        available_experts = sorted(available_experts, key=lambda x: x[1])

        # Select up to n_experts from available experts
        selected_experts = random.sample(available_experts, min(self.n_experts, len(available_experts)))

        print(f"Found {len(available_experts)} experts with reasoning, selected {len(selected_experts)} for this question")

        # Initialize result structure
        result = {
            "aggregate": None,
            "individual_forecasts": [],
            "selected_experts": [user_id for _, user_id in selected_experts],
            "round1_responses": [],
            "round2_responses": [],
            "expert_details": []
        }

        # Check if we should do two-round Delphi
        delphi_rounds = self.config.get('panel', {}).get('delphi_rounds', 1)

        if delphi_rounds == 2:
            # Round 1: Collect initial responses from all experts
            print(f"\nStarting Delphi Round 1 for question {question_id[:8]}...")

            for expert, user_id in selected_experts:
                response = expert.generate_round_1_response(question, conditioning_forecast=expert_forecasts[user_id])
                result["round1_responses"].append({
                    'expert_id': user_id,
                    'response': response
                })
                print(f"Expert {user_id}: Round 1 response collected")

            # Round 2: Each expert sees all responses and makes final forecast
            print(f"\nStarting Delphi Round 2 (experts see all Round 1 responses)...")

            for expert, user_id in selected_experts:
                # Get the user's forecast for this question (we know it exists from filtering)
                user_forecast_for_question = expert_forecasts[user_id]

                # Create aggregated round 1 responses (anonymized)
                other_responses = "\n\n---\n\n".join([
                    f"Expert {i+1} Response:\n{r['response']}"
                    for i, r in enumerate(result["round1_responses"])
                ])

                # Get forecast with round 1 context
                expert_forecast, round2_response = expert.forecast_with_round1_context(
                    question,
                    other_responses,
                    user_forecast_for_question
                )
                result["individual_forecasts"].append(expert_forecast)
                result["round2_responses"].append({
                    'expert_id': user_id,
                    'response': round2_response
                })
                result["expert_details"].append({
                    "expert_id": user_id,
                    "final_forecast": expert_forecast,
                    "original_forecast": user_forecast_for_question.forecast,
                    "has_round1_response": True
                })
                print(f"Expert {user_id}: Final forecast = {expert_forecast:.3f}")

        else:
            # Original single-round process
            for expert, user_id in selected_experts:
                # Get the user's forecast for this question (we know it exists from filtering)
                user_forecast_for_question = expert_forecasts[user_id]

                print(f"Expert {user_id}: Using their actual forecast of {user_forecast_for_question.forecast} with reasoning")

                expert_forecast = await expert.forecast(question, user_forecast_for_question) # only used for reasoning
                result["individual_forecasts"].append(expert_forecast)
                result["expert_details"].append({
                    "expert_id": user_id,
                    "final_forecast": expert_forecast,
                    "original_forecast": user_forecast_for_question.forecast,
                    "has_round1_response": False
                })

        # Calculate aggregate
        result["aggregate"] = sum(result["individual_forecasts"]) / len(result["individual_forecasts"])

        # Calculate human performance for the sampled group
        sampled_human_forecasts = [expert_forecasts[user_id].forecast for _, user_id in selected_experts]
        result["human_group_mean"] = sum(sampled_human_forecasts) / len(sampled_human_forecasts) if sampled_human_forecasts else None
        result["human_group_std"] = float(np.std(sampled_human_forecasts)) if len(sampled_human_forecasts) > 1 else 0.0

        # Get resolution if available for calculating metrics
        resolution = self.loader.get_resolution(question_id)
        if resolution:
            outcome = float(resolution.resolved_to)
            result["outcome"] = outcome
            result["brier"] = (result["aggregate"] - outcome) ** 2
            result["mae"] = abs(result["aggregate"] - outcome)
            # Human group metrics
            if result["human_group_mean"] is not None:
                result["human_group_brier"] = (result["human_group_mean"] - outcome) ** 2
                result["human_group_mae"] = abs(result["human_group_mean"] - outcome)
            else:
                result["human_group_brier"] = None
                result["human_group_mae"] = None

        return result

    async def forecast_all_questions(self):
        results = {}
        for q in self.loader.get_all_questions():
            results[q.id] = await self.forecast_question(q.id)
        return results


async def main():
    loader = ForecastDataLoader()

    # Create panel with conditioning on human forecaster data
    panel = DelphiPanel(loader,
                        provider=LLMProvider.CLAUDE,
                        model=LLMModel.CLAUDE_4_HAIKU,
                        n_experts=3,  # Use fewer experts for testing
                        condition_on_data=True,
                        config={'panel': {'delphi_rounds': 2}})  # Enable 2-round Delphi

    print(f"Panel initialized with {len(panel.expert_pool)} potential experts: {list(panel.expert_pool.keys())}")

    # Get a resolved question for comparison
    resolved_questions = loader.get_resolved_questions()
    if resolved_questions:
        sample = resolved_questions[0]
    else:
        sample = loader.sample_random_question()

    print(f"\nQuestion: {sample.question}\n")

    # Show human forecasts if available
    human_forecasts = loader.get_super_forecasts(question_id=sample.id)
    if human_forecasts:
        human_avg = sum(f.forecast for f in human_forecasts) / len(human_forecasts)
        print(f"Superforecasts average: {human_avg:.3f} (n={len(human_forecasts)})")

    # Show resolution if available
    resolution = loader.get_resolution(sample.id)
    if resolution:
        print(f"Actual resolution: {resolution.resolved_to}")

    try:
        result = await panel.forecast_question(sample.id)
        print(f"\nAI Panel forecasts: {[f'{f:.3f}' for f in result['individual_forecasts']]}")
        print(f"AI Panel aggregate: {result['aggregate']:.3f}")

        # Print additional details
        print(f"\nSelected experts: {result['selected_experts']}")
        print(f"\nExpert details:")
        for detail in result['expert_details']:
            print(f"  - {detail['expert_id']}: original={detail['original_forecast']:.3f}, final={detail['final_forecast']:.3f}")

        if result['round1_responses']:
            print(f"\n{len(result['round1_responses'])} Round 1 responses stored")

        if result['round2_responses']:
            print(f"\n{len(result['round2_responses'])} Round 2 responses stored")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
