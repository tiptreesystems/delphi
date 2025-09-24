from typing import Optional, List, Tuple

from dataset.dataloader import Question, Forecast
from utils.models import BaseLLM, ConversationManager
from utils.probability_parser import extract_final_probability_with_retry
from utils.prompt_formatters import (
    build_base_forecast_prompt,
    build_in_context_forecast_prompt,
)
import copy


class Expert:
    def __init__(
        self,
        llm: BaseLLM,
        user_profile: Optional[dict] = None,
        config: Optional[dict] = None,
    ):
        self.llm = llm
        self.user_profile = user_profile
        self.config = config or {}
        self.conversation_manager = ConversationManager(llm)
        self.token_warnings = []  # Track token usage warnings
        self.retry_count = 0  # Track retries for token issues

    async def forecast(
        self,
        question: Question,
        conditioning_forecast: Optional[Forecast] = None,
        seed: Optional[int] = None,
    ) -> float:
        # Render prompt via shared formatter
        prompt_version = self.config.get("prompt_version", "v1")
        prompt = build_base_forecast_prompt(
            question,
            prompt_version=prompt_version,
            conditioning_forecast=conditioning_forecast,
        )
        temperature = self.config.get("temperature", 0.3)
        self.conversation_manager.messages.clear()

        # Pass seed if provided for deterministic results
        kwargs = {
            "max_tokens": self.config.get("max_tokens", 500),
            "temperature": temperature,
        }
        if seed is not None:
            kwargs["seed"] = seed

        response = await self.conversation_manager.generate_response(prompt, **kwargs)
        response = response.strip()

        prob = await extract_final_probability_with_retry(
            response, self.conversation_manager, max_retries=3
        )
        if prob != -1:
            self.retry_count = 0  # Reset retry count on success
            return prob

        self.token_warnings.append(
            f"No valid probability found in response of {len(response)} chars"
        )
        return -1

    async def forecast_with_examples_in_context(
        self, question: Question, examples: List[Tuple[Question, Forecast]]
    ) -> float:
        # Build prompt using shared formatter (preserves existing structure)
        prompt = build_in_context_forecast_prompt(question, examples)

        temperature = self.config.get("temperature", 0.3)
        self.conversation_manager.messages.clear()
        response = await self.conversation_manager.generate_response(
            prompt,
            max_tokens=self.config.get("max_tokens", 500),
            temperature=temperature,
        )
        response = response.strip()

        # Parse probability
        prob = await extract_final_probability_with_retry(
            response, self.conversation_manager, max_retries=3
        )

        return prob if prob != -1 else -1

    async def get_forecast_update(self, input_message) -> float:
        """Get a response without clearing the conversation, used after feedback."""
        if not self.conversation_manager.messages:
            raise RuntimeError(
                "No conversation history found. Cannot update forecast without prior context."
            )
        response = await self.conversation_manager.generate_response(
            input_message,
            input_message_type="user",
            max_tokens=self.config.get("max_tokens", 800),
            temperature=self.config.get("temperature", 0.3),
        )
        response = response.strip()
        prob = await extract_final_probability_with_retry(
            response, self.conversation_manager, max_retries=3
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

    def duplicate(self) -> "Expert":
        """
        Return a new Expert with the same configuration and state but completely
        distinct memory (deep-copied internal structures). The LLM reference is
        shared (not deep-copied).
        """

        # Deep-copy simple configs and profiles
        new_user_profile = copy.deepcopy(self.user_profile)
        new_config = copy.deepcopy(self.config)

        # Construct new Expert with the same llm reference
        new = Expert(self.llm, new_user_profile, new_config)

        # Copy simple scalar fields
        new.token_warnings = copy.deepcopy(self.token_warnings)
        new.retry_count = self.retry_count

        # Create a fresh ConversationManager tied to the same llm and deep-copy
        # all of its attributes except the llm reference so memory is distinct.
        new.conversation_manager = ConversationManager(self.llm)
        for attr, value in vars(self.conversation_manager).items():
            if attr == "llm":
                continue
            setattr(new.conversation_manager, attr, copy.deepcopy(value))

        return new
