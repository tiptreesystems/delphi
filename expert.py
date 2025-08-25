from typing import Optional, List, Tuple

from dataset.dataloader import Question, Forecast
from utils.models import BaseLLM, ConversationManager
from utils.probability_parser import extract_final_probability_with_retry
from utils.prompt_loader import load_prompt


class Expert:
    def __init__(self, llm: BaseLLM, user_profile: Optional[dict] = None, config: Optional[dict] = None):
        self.llm = llm
        print(f"Expert initialized with: {llm.system_prompt}")
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

        # Pass seed if provided for deterministic results
        kwargs = {'max_tokens': 500, 'temperature': temperature}
        if seed is not None:
            kwargs['seed'] = seed

        response = await self.conversation_manager.generate_response(prompt, **kwargs)
        response = response.strip()

        prob = await extract_final_probability_with_retry(
            response,
            self.conversation_manager,
            max_retries=3
        )
        if prob != -1:
            self.retry_count = 0  # Reset retry count on success
            return prob

        self.token_warnings.append(f"No valid probability found in response of {len(response)} chars")
        return -1

    async def forecast_with_examples_in_context(self, question: Question, examples: List[Tuple[Question, Forecast]]) -> float:
        # Build examples section with clear structure
        examples_text = "REFERENCE EXAMPLES OF EXPERT FORECASTS:\n" + "="*60 + "\n\n"

        for i, (ex_q, ex_f) in enumerate(examples):
            examples_text += (
                f"[EXAMPLE {i+1}]\n"
                f"Question: {ex_q.question}\n"
                f"Background: {ex_q.background}\n"
                f"Resolution: {ex_q.resolution_criteria}\n"
                f"Analysis: {ex_f.reasoning}\n"
                f"Probability: {ex_f.forecast}\n"
                f"{'-'*40}\n\n"
            )

        # Clear transition to the target question
        target_section = (
            "\n" + "="*60 + "\n"
            "YOUR TASK - PROVIDE FORECAST FOR THIS QUESTION:\n"
            + "="*60 + "\n\n"
            f"Question: {question.question}\n"
            f"Background: {question.background}\n"
            f"Resolution: {question.resolution_criteria}\n"
            f"URL: {question.url}\n"
            f"Freeze value: {question.freeze_datetime_value}\n"
            f"Freeze value explanation: {question.freeze_datetime_value_explanation}\n\n"
        )

        # Explicit instructions for output
        instructions = (
            "Based on the examples above, provide your forecast concluding with:\n"
            "FINAL PROBABILITY: [decimal between 0 and 1]"
        )

        prompt = examples_text + target_section + instructions

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

        return prob if prob != -1 else -1

    async def get_forecast_update(self, input_message) -> float:
        """Get a response without clearing the conversation, used after feedback."""
        if not self.conversation_manager.messages:
            raise RuntimeError("No conversation history found. Cannot update forecast without prior context.")
        response = await self.conversation_manager.generate_response(input_message, input_message_type="user", max_tokens=self.config.get('max_tokens', 800), temperature=self.config.get('temperature', 0.3))
        response = response.strip()
        prob = await extract_final_probability_with_retry(
            response,
            self.conversation_manager,
            max_retries=3
        )
        return (prob, response) if prob != -1 else (-1, response)

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
