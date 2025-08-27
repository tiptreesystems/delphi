import os
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from abc import ABC, abstractmethod
import anthropic
import openai
from groq import Groq
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
    GROQ = "groq"


class LLMModel(Enum):
    # Claude models
    CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_4_OPUS = "claude-opus-4-20250514"
    CLAUDE_4_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"

    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13" #snapshot to avoid memory leaks
    GPT_4O_MINI = "gpt-4o-mini"
    O3_MINI = "o3-mini"
    O3 = "o3"
    O3_2025_04_16 = "o3-2025-04-16"
    GPT_5_2025_08_07 = "gpt-5-2025-08-07"
    O1_2024_12_17 = "o1-2024-12-17"
    
    # Groq models (OpenAI-compatible models hosted on Groq)
    GROQ_GPT_OSS_20B = "openai/gpt-oss-20b"
    GROQ_GPT_OSS_120B = "openai/gpt-oss-120b"
    GROQ_LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    GROQ_LLAMA_3_1_70B = "llama-3.1-70b-versatile"
    GROQ_MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GROQ_LLAMA_4_MAVERICK_17B = "meta-llama/llama-4-maverick-17b-128e-instruct"
    GROQ_QWEN3_32B = "qwen/qwen3-32b"
    GROQ_DEEPSEEK_R1_DISTILL_70B = "deepseek-r1-distill-llama-70b"




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


class GroqLLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None, model: str = LLMModel.GROQ_LLAMA_3_3_70B.value):
        super().__init__(api_key or os.getenv("GROQ_API_KEY"), system_prompt)
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.reasoning_effort = "medium"  # Default reasoning effort for models that support it
        self.last_usage = {}  # Track token usage from last request

    async def generate(self, prompt: str, max_tokens: int = 8192, temperature: float = 0.7, **kwargs) -> str:
        # Note: Groq uses synchronous API, but we wrap it in async for compatibility
        # Build parameters dict
        create_params = {
            'model': self.model,
            'messages': [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            'max_completion_tokens': max_tokens,
            'temperature': temperature,
        }
        
        # Add seed if provided for deterministic results
        if 'seed' in kwargs:
            create_params['seed'] = kwargs['seed']
            
        # Add reasoning effort for models that support it
        if 'gpt-oss' in self.model and kwargs.get('reasoning_effort'):
            create_params['reasoning_effort'] = kwargs.get('reasoning_effort', self.reasoning_effort)
            
        # Add any other kwargs except reasoning_effort and seed (already handled)
        for k, v in kwargs.items():
            if k not in ['reasoning_effort', 'seed']:
                create_params[k] = v
        
        try:
            response = self.client.chat.completions.create(**create_params)
            
            # Log token usage for monitoring
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                completion_tokens = usage.completion_tokens
                prompt_tokens = usage.prompt_tokens
                total_tokens = usage.total_tokens
                
                # Store usage info for retrieval
                self.last_usage = {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens,
                    'max_tokens_requested': max_tokens
                }
                
                # Warn if we're approaching token limits
                if completion_tokens >= max_tokens * 0.95:
                    print(f"⚠️  WARNING: Response used {completion_tokens}/{max_tokens} tokens (95%+ of limit). Consider increasing max_tokens.")
                elif completion_tokens >= max_tokens * 0.8:
                    print(f"🔶 NOTICE: Response used {completion_tokens}/{max_tokens} tokens (80%+ of limit).")
                    
                # Log token usage
                print(f"📊 Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Store error info
            self.last_usage = {'error': str(e)}
            print(f"❌ Error during generation: {e}")
            raise

    def generate_stream(self, prompt: str, max_tokens: int = 8192, temperature: float = 0.7, **kwargs):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=kwargs.get('reasoning_effort', self.reasoning_effort) if 'gpt-oss' in self.model else None,
            stream=True,
            **{k: v for k, v in kwargs.items() if k != 'reasoning_effort'}
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None, model: str = LLMModel.GPT_4O.value):
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"), system_prompt)
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model = model

    async def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, **kwargs) -> str:
        # @retry_with_exponential_backoff
        async def _generate():

            if (self.model.startswith('o1') or self.model.startswith('o3') or self.model.startswith('gpt-5')):
                # o1 and o3 models require max_completion_tokens instead of max_tokens
                create_params = {
                    'model': self.model,
                    'messages': [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_completion_tokens": max_tokens,
                }

                response = await self.client.chat.completions.create(**create_params)
                return response.choices[0].message.content

            else:
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

        return await _generate()



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
        
        elif provider == LLMProvider.GROQ:
            if model is None:
                model = LLMModel.GROQ_LLAMA_3_3_70B.value
            elif isinstance(model, LLMModel):
                model = model.value
            return GroqLLM(api_key=api_key, system_prompt=system_prompt, model=model)

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

    async def generate_response(self, user_input: str, add_to_history: bool = True, input_message_type: str = "user", **kwargs) -> str:
        if add_to_history:
            self.add_message(input_message_type, user_input)

        async def _generate_with_retry(max_retries=5, base_backoff=1.0):
            import asyncio
            import re

            for attempt in range(max_retries):
                try:
                    if isinstance(self.llm, ClaudeLLM):
                        # Filter out system messages for Claude API (system prompt is passed separately)
                        claude_messages = [msg for msg in self.messages if msg.get("role") != "system"]
                        response = self.llm.client.messages.create(
                            model=self.llm.model,
                            system=self.llm.system_prompt,
                            messages=claude_messages,
                            **kwargs
                        )
                        return response.content[0].text if response.content else ""
                    elif isinstance(self.llm, GroqLLM):
                        # Groq uses synchronous API but we handle it here
                        messages = [{"role": "system", "content": self.llm.system_prompt}] + self.messages
                        
                        # Build parameters dict
                        create_params = {
                            'model': self.llm.model,
                            'messages': messages,
                            'max_completion_tokens': kwargs.get('max_tokens', 8192),
                            'temperature': kwargs.get('temperature', 0.7),
                        }
                        
                        # Add seed if provided for deterministic results
                        if 'seed' in kwargs:
                            create_params['seed'] = kwargs['seed']
                            
                        # Add any other kwargs except those already handled
                        for k, v in kwargs.items():
                            if k not in ['max_tokens', 'temperature', 'seed', 'reasoning_effort']:
                                create_params[k] = v
                        
                        response = self.llm.client.chat.completions.create(**create_params)
                        return response.choices[0].message.content
                    else:
                        messages = [{"role": "system", "content": self.llm.system_prompt}] + self.messages
                        
                        # Check if this is an o1 or o3 model that requires max_completion_tokens
                        if isinstance(self.llm, OpenAILLM) and (self.llm.model.startswith('o1') or self.llm.model.startswith('o3') or self.llm.model.startswith('gpt-5')):
                            # o1 and o3 models require max_completion_tokens instead of max_tokens
                            create_params = {
                                'model': self.llm.model,
                                'messages': messages,
                            }
                            
                            # Make a copy of kwargs to avoid modifying the original
                            kwargs_copy = kwargs.copy()
                            
                            # Convert max_tokens to max_completion_tokens for o1/o3 models
                            if 'max_tokens' in kwargs_copy:
                                create_params['max_completion_tokens'] = kwargs_copy.pop('max_tokens')
                            
                            # Remove temperature as it's not supported by o1/o3 models
                            if 'temperature' in kwargs_copy:
                                kwargs_copy.pop('temperature')
                            
                            # Add remaining kwargs
                            create_params.update(kwargs_copy)
                            
                            response = await self.llm.client.chat.completions.create(**create_params)
                        else:
                            response = await self.llm.client.chat.completions.create(
                                model=self.llm.model,
                                messages=messages,
                                **kwargs
                            )
                        return response.choices[0].message.content
                        
                except Exception as e:
                    # Check if it's a rate limit error
                    is_rate_limit = (
                        'rate' in str(e).lower() and 'limit' in str(e).lower()
                    ) or (
                        hasattr(e, '__class__') and 'RateLimitError' in str(e.__class__.__name__)
                    )
                    
                    if is_rate_limit and attempt < max_retries - 1:
                        # Extract wait time from error message if available
                        wait_time_match = re.search(r'try again in ([\d.]+)\s*(ms|s)', str(e).lower())
                        if wait_time_match:
                            wait_time = float(wait_time_match.group(1))
                            if wait_time_match.group(2) == 'ms':
                                wait_time = wait_time / 1000.0  # Convert to seconds
                        else:
                            # Use exponential backoff if no specific wait time found
                            wait_time = base_backoff * (2 ** attempt)
                        
                        print(f"⏳ Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Re-raise if not a rate limit error or max retries exceeded
                        raise e
            
            # This should never be reached due to the raise in the except block
            raise Exception("Max retries exceeded")

        content = await _generate_with_retry()
        if add_to_history:
            self.add_message("assistant", content)
        return content
