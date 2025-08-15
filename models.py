import os
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from abc import ABC, abstractmethod
import anthropic
import openai
import time
import logging
import inspect


# def retry_with_exponential_backoff(
#     func,
#     initial_delay: float = 1,
#     exponential_base: float = 2,
#     jitter: bool = True,
#     max_retries: int = 5,
#     errors: tuple = (anthropic.RateLimitError, openai.RateLimitError)
# ):
#     """Retry a function with exponential backoff."""
#     def wrapper(*args, **kwargs):
#         num_retries = 0
#         delay = initial_delay

#         while True:
#             try:
#                 return func(*args, **kwargs)
#             except errors as e:
#                 num_retries += 1

#                 if num_retries > max_retries:
#                     raise Exception(f"Maximum number of retries ({max_retries}) exceeded.") from e

#                 delay *= exponential_base * (1 + jitter * (0.1 * (2 * (0.5 - time.time() % 1))))

#                 logging.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds... (attempt {num_retries}/{max_retries})")
#                 time.sleep(delay)
#             except Exception as e:
#                 raise e

#     return wrapper


class LLMProvider(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"


class LLMModel(Enum):
    # Claude models
    CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_4_OPUS = "claude-opus-4-20250514"
    CLAUDE_4_HAIKU = "claude-3-5-haiku-latest"

    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O3_MINI = "o3-mini"
    O3 = "o3"




class BaseLLM(ABC):
    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        self.api_key = api_key
        self.system_prompt = system_prompt or "You are a helpful assistant."

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs):
        pass


class ClaudeLLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None, model: str = LLMModel.CLAUDE_4_SONNET.value):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"), system_prompt)
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, **kwargs) -> str:
        # @retry_with_exponential_backoff
        def _generate():
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.content[0].text if response.content else ""

        return _generate()

    def generate_stream(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, **kwargs):
        stream = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        for event in stream:
            if event.type == "content_block_delta":
                yield event.delta.text


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None, model: str = LLMModel.GPT_4O.value):
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"), system_prompt)
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model = model

    async def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, **kwargs) -> str:
        # @retry_with_exponential_backoff
        async def _generate():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content

        return _generate()

    def generate_stream(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, **kwargs):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LLMFactory:
    @staticmethod
    def create_llm(
        provider: Union[LLMProvider, str],
        model: Optional[Union[LLMModel, str]] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> BaseLLM:
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())

        if provider == LLMProvider.CLAUDE:
            if model is None:
                model = LLMModel.CLAUDE_4_SONNET.value
            elif isinstance(model, LLMModel):
                model = model.value
            return ClaudeLLM(api_key=api_key, system_prompt=system_prompt, model=model)

        elif provider == LLMProvider.OPENAI:
            if model is None:
                model = LLMModel.GPT_4O.value
            elif isinstance(model, LLMModel):
                model = model.value
            return OpenAILLM(api_key=api_key, system_prompt=system_prompt, model=model)

        else:
            raise ValueError(f"Unsupported provider: {provider}")


class ConversationManager:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def add_messages(self, messages: List[Dict[str, str]]):
        for message in messages:
            if "role" in message and "content" in message:
                self.add_message(message["role"], message["content"])
            else:
                raise ValueError("Each message must contain 'role' and 'content' keys.")

    async def generate_response(self, user_input: str, **kwargs) -> str:
        self.add_message("user", user_input)

        # @retry_with_exponential_backoff
        async def _generate():
            if isinstance(self.llm, ClaudeLLM):
                response = await self.llm.client.messages.create(
                    model=self.llm.model,
                    system=self.llm.system_prompt,
                    messages=self.messages,
                    **kwargs
                )
                return response.content[0].text if response.content else ""
            else:
                messages = [{"role": "system", "content": self.llm.system_prompt}] + self.messages
                response = await self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content

        content = await _generate()
        self.add_message("assistant", content)
        return content
