import os
from typing import List, Tuple, Optional, Dict
from models import LLMFactory, LLMProvider, LLMModel, BaseLLM, ConversationManager
from dataset.dataloader import ForecastDataLoader, Question, Forecast
import re
import random
import json
import dotenv
import numpy as np
import asyncio

dotenv.load_dotenv()

delphi_round_1_prompt = """
Delphi Round 1 — Expert Elicitation Survey

1. Ranking & Priorities
1.1 Rank the top 5 drivers that will influence the focal question (1 = strongest).
1.2 Which single driver is most overlooked by other experts? (max 50 words)

2. Early-Warning Indicators
List up to 3 observable signals that would make you revise any probability by ≥15 pp.

3. Risk & Uncertainty
3.1 What is the main "unknown unknown" you worry about? (max 50 words)
3.2 Probability that a black-swan event (>3 σ impact) occurs before the horizon: ___ %

4. Conditional Forecast (counterfactual)
"If [critical assumption] fails, what is your revised probability for the focal outcome?" ___ %

5. Key Forecasts (repeat table for each statement)
#Forecast StatementYour Probability (%)90% CI (low–high)2-Sentence RationaleKey Assumptions (bullets)Sources / URLs5.1………………
(Add rows as needed.)

Submission
Return the completed table and answers. Your identity will be anonymized; only aggregated statistics and rationales (paraphrased) will be shared with the panel.

Be concise and to the point.
"""

class Expert:
    def __init__(self, llm: BaseLLM, user_profile: Optional[dict] = None, config: Optional[dict] = None):
        self.llm = llm
        self.user_profile = user_profile
        self.config = config or {}
        self.conversation_manager = ConversationManager(llm)

    async def forecast(self, question: Question, conditioning_forecast: Optional[Forecast] = None) -> float:

        # Add the user's actual forecast for this question if available
        prior_forecast_info = ""
        if conditioning_forecast:
            prior_forecast_info = (
                f"Some of your notes on this question are: {conditioning_forecast.reasoning}\n"
            )

        prompt = (
            f"Based on the following question, analyze it carefully and provide your probability estimate.\n\n"
            f"Question: {question.question}\n"
            f"Background: {question.background}\n"
            f"Resolution criteria: {question.resolution_criteria}\n"
            f"URL: {question.url}\n"
            f"Market freeze value: {question.freeze_datetime_value}\n"
            f"{prior_forecast_info}\n"
            f"You may reason through the problem, but you MUST end your response with:\n"
            f"FINAL PROBABILITY: [your decimal number between 0 and 1]"
        )
        temperature = self.config.get('temperature', 0.3)
        self.conversation_manager.messages.clear()
        response = await self.conversation_manager.generate_response(prompt, max_tokens=500, temperature=temperature)
        response = response.strip()

        matches = list(re.finditer(r'FINAL PROBABILITY:\s*(0?\.\d+|1\.0|0|1)', response, re.IGNORECASE))
        match = matches[-1] if matches else None
        if match:
            return float(match.group(1))

        # Fallback: try to find any number at the end of the response
        numbers = re.findall(r'0?\.\d+|1\.0|0|1', response)
        if numbers:
            prob = float(numbers[-1])  # Take the last number found
            return max(0.0, min(1.0, prob))

        # Fallback if no valid number found
        return 0.5

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

        prompt = (
            f"{examples_text}"
            f"Now consider the following question and provide your forecast.\n\n"
            f"Question: {question.question}\n"
            f"Background: {question.background}\n"
            f"Resolution criteria: {question.resolution_criteria}\n"
            f"URL: {question.url}\n"
            f"Market freeze value: {question.freeze_datetime_value}\n"
            f"You may reason through the problem, but you MUST end your response with:\n"
            f"FINAL PROBABILITY: [your decimal number between 0 and 1]"
        )

        temperature = self.config.get('temperature', 0.3)
        self.conversation_manager.messages.clear()
        response = await self.conversation_manager.generate_response(prompt, max_tokens=500, temperature=temperature)
        response = response.strip()

        matches = list(re.finditer(r'FINAL PROBABILITY:\s*(0?\.\d+|1\.0|0|1)', response, re.IGNORECASE))
        match = matches[-1] if matches else None
        if match:
            return float(match.group(1))

        # Fallback: try to find any number at the end of the response
        numbers = re.findall(r'0?\.\d+|1\.0|0|1', response)
        if numbers:
            prob = float(numbers[-1])  # Take the last number found
            return max(0.0, min(1.0, prob))
        return 0.5

    def generate_round_1_response(self, question: Question, conditioning_forecast: Optional[Forecast] = None) -> str:
        prompt = (
            f"Here is the question, background, and resolution criteria. You should now think about it and fill out the Delphi Round 1 Expert Elicitation Survey.\n\n"
            f"Question: {question.question}\n"
            f"Background: {question.background}\n"
            f"Resolution criteria: {question.resolution_criteria}\n"
            f"Your forecast: {conditioning_forecast.forecast}. You should argue for this forecast and justify it using the data and your expertise unless you have a good reason to revise it.\n" if conditioning_forecast else ""
            f"{delphi_round_1_prompt}\n\n"
        )
        max_tokens = self.config.get('max_tokens', 2000)
        temperature = self.config.get('temperature', 0.3)
        self.conversation_manager.messages.clear()
        response = self.conversation_manager.generate_response(prompt, max_tokens=max_tokens, temperature=temperature).strip()
        return response

    def forecast_with_round1_context(self, question: Question, round1_responses: str, conditioning_forecast: Optional[Forecast] = None) -> Tuple[float, str]:
        """Make a forecast after seeing other experts' Round 1 responses."""
        prompt = (
            f"You may now review other experts' assessments and update your beliefs based on them and your own expertise.\n\n"
            f"Question: {question.question}\n"
            f"Background: {question.background}\n"
            f"Resolution criteria: {question.resolution_criteria}\n"
            f"URL: {question.url}\n"
            f"EXPERTS' ROUND 1 ASSESSMENTS:\n"
            f"{round1_responses}\n\n"
            f"--------------------------------\n"
            f"After considering the other experts' perspectives, think through your reasoning and provide your final probability estimate.\n"
            f"You may reason through the problem, but you MUST end your response with:\n"
            f"FINAL PROBABILITY: [your decimal number between 0 and 1]"
        )
        temperature = self.config.get('temperature', 0.3)
        response = self.conversation_manager.generate_response(prompt, max_tokens=800, temperature=temperature).strip()

        # Extract the final probability after "FINAL PROBABILITY:"
        match = re.search(r'FINAL PROBABILITY:\s*(0?\.\d+|1\.0|0|1)', response, re.IGNORECASE)
        if match:
            prob = float(match.group(1))
            return max(0.0, min(1.0, prob)), response  # Return both probability and response

        # Fallback: try to find any number at the end of the response
        numbers = re.findall(r'0?\.\d+|1\.0|0|1', response)
        if numbers:
            prob = float(numbers[-1])  # Take the last number found
            return max(0.0, min(1.0, prob)), response

        # Fallback if no valid number found
        return 0.5, response

    async def get_forecast_update(self, input_message) -> float:
        """Get a response without clearing the conversation, used after feedback."""
        if not self.conversation_manager.messages:
            raise RuntimeError("No conversation history found. Cannot update forecast without prior context.")
        response = await self.conversation_manager.generate_response(input_message, max_tokens=800, temperature=self.config.get('temperature', 0.3))
        response = response.strip()
        # Extract the last occurrence of "FINAL PROBABILITY:"
        matches = list(re.finditer(r'FINAL PROBABILITY:\s*(0?\.\d+|1\.0|0|1)', response, re.IGNORECASE))
        match = matches[-1] if matches else None
        if match:
            return float(match.group(1)), response
        return 0.5, response

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
        self.system_prompt = self.config.get(
            "system_prompt",
            "You are the Delphi mediator. Summarize areas of agreement and disagreement, "
            "identify key cruxes, and suggest what evidence or reasoning would most change minds. "
            "Be concise and neutral."
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
        # Seed a system message once if none exists yet
        if not any(m["role"] == "system" for m in self.conversation_manager.messages):
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

        # Ensure there is a system message
        if not any(m["role"] == "system" for m in self.conversation_manager.messages):
            self.conversation_manager.add_message(role="system", content=self.system_prompt)

        # Append expert messages for this turn (anonymized, deterministic order)
        if self.expert_messages:
            for eid in sorted(self.expert_messages.keys()):
                self.conversation_manager.add_message(
                    role="user",
                    content=f"[EXPERT {eid}] {self.expert_messages[eid]}"
                )

        # Append the mediator synthesis request (annotate round if provided)
        prefix = f"(Round {round_idx}) " if round_idx is not None else ""
        mediator_request = (
            f"{prefix}Synthesize the above into a concise feedback memo for all experts.\n"
            "Structure:\n"
            "1) Consensus (if any)\n"
            "2) Points of disagreement & cruxes\n"
            "3) Evidence/analyses that would most shift views\n"
            "4) Actionable next steps for the next Delphi round\n"
            "Keep it under ~300 words."
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
        system_prompt: str = "You are an expert superforecaster, familiar with Structured Analytic Techniques as well as Superforecasting by Philip Tetlock and the Delphi Method. You have the following profile:\n"
        f"You are participating in a Delphi panel. You will be given a question, background, some initial thoughts, and your initial forecast. You should rely on these (particularly your initial forecast) for the first round of the Delphi panel, typically not straying too far from them. You will do your best to faithfully participate in the panel.",
        condition_on_data: bool = True,
        config: Optional[dict] = None
    ):
        self.loader = loader
        self.condition_on_data = condition_on_data
        self.config = config or {}

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
            personalized_system_prompt = (
                f"{system_prompt}\n\n"
                f" You should embody the following profile:\n"
                f"{user_profile['expertise_profile']}\n"
            )

            llm = LLMFactory.create_llm(provider, model, system_prompt=personalized_system_prompt)
            expert = Expert(llm, user_profile=user_profile, config=self.config.get('model', {}))
            self.expert_pool[user_id] = expert

    def forecast_question(self, question_id: str) -> dict:
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
                "aggregate": 0.5,
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

                expert_forecast = expert.forecast(question, user_forecast_for_question) # only used for reasoning
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

    def forecast_all_questions(self):
        results = {}
        for q in self.loader.get_all_questions():
            results[q.id] = self.forecast_question(q.id)
        return results


if __name__ == "__main__":
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
    human_forecasts = loader.get_super_forecasts(sample.id)
    if human_forecasts:
        human_avg = sum(f.forecast for f in human_forecasts) / len(human_forecasts)
        print(f"Human forecasts average: {human_avg:.3f} (n={len(human_forecasts)})")

    # Show resolution if available
    resolution = loader.get_resolution(sample.id)
    if resolution:
        print(f"Actual resolution: {resolution.resolved_to}")

    try:
        result = panel.forecast_question(sample.id)
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

