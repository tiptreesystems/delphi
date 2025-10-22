import json
import os
import inspect
import logging
from typing import Optional, Dict, List, Union, Callable, Awaitable, Any
from enum import Enum
from abc import ABC, abstractmethod
import anthropic
import openai
from groq import Groq
from pydantic import BaseModel
from textwrap import dedent


logger = logging.getLogger(__name__)

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
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"  # snapshot to avoid memory leaks
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
    GROQ_KIMI_K2_INSTRUCT = "moonshotai/kimi-k2-instruct"


class BaseLLM(ABC):
    def __init__(
        self, api_key: Optional[str] = None, system_prompt: Optional[str] = None
    ):
        self.api_key = api_key
        self.system_prompt = system_prompt or "You are a helpful assistant."

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs):
        pass


class ClaudeLLM(BaseLLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: str = LLMModel.CLAUDE_4_SONNET.value,
    ):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"), system_prompt)
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def generate(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, **kwargs
    ) -> str:
        # @retry_with_exponential_backoff
        def _generate():
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.content[0].text if response.content else ""

        return _generate()

    def generate_stream(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, **kwargs
    ):
        stream = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs,
        )
        for event in stream:
            if event.type == "content_block_delta":
                yield event.delta.text


class GroqLLM(BaseLLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: str = LLMModel.GROQ_LLAMA_3_3_70B.value,
    ):
        super().__init__(api_key or os.getenv("GROQ_API_KEY"), system_prompt)
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.reasoning_effort = (
            "medium"  # Default reasoning effort for models that support it
        )
        self.last_usage = {}  # Track token usage from last request

    async def generate(
        self, prompt: str, max_tokens: int = 8192, temperature: float = 0.7, **kwargs
    ) -> str:
        # Note: Groq uses synchronous API, but we wrap it in async for compatibility
        # Build parameters dict
        create_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add seed if provided for deterministic results
        if "seed" in kwargs:
            create_params["seed"] = kwargs["seed"]

        # Add reasoning effort for models that support it
        if "gpt-oss" in self.model and kwargs.get("reasoning_effort"):
            create_params["reasoning_effort"] = kwargs.get(
                "reasoning_effort", self.reasoning_effort
            )

        # Add any other kwargs except reasoning_effort and seed (already handled)
        for k, v in kwargs.items():
            if k not in ["reasoning_effort", "seed"]:
                create_params[k] = v

        try:
            response = self.client.chat.completions.create(**create_params)

            # Log token usage for monitoring
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                completion_tokens = usage.completion_tokens
                prompt_tokens = usage.prompt_tokens
                total_tokens = usage.total_tokens

                # Store usage info for retrieval
                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "max_tokens_requested": max_tokens,
                }

                # Warn if we're approaching token limits
                if completion_tokens >= max_tokens * 0.95:
                    print(
                        f"⚠️  WARNING: Response used {completion_tokens}/{max_tokens} tokens (95%+ of limit). Consider increasing max_tokens."
                    )
                elif completion_tokens >= max_tokens * 0.8:
                    print(
                        f"🔶 NOTICE: Response used {completion_tokens}/{max_tokens} tokens (80%+ of limit)."
                    )

                # Log token usage
                print(
                    f"📊 Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}"
                )

            return response.choices[0].message.content

        except Exception as e:
            # Store error info
            self.last_usage = {"error": str(e)}
            print(f"❌ Error during generation: {e}")
            raise

    def generate_stream(
        self, prompt: str, max_tokens: int = 8192, temperature: float = 0.7, **kwargs
    ):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=kwargs.get("reasoning_effort", self.reasoning_effort)
            if "gpt-oss" in self.model
            else None,
            stream=True,
            **{k: v for k, v in kwargs.items() if k != "reasoning_effort"},
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: str = LLMModel.GPT_4O.value,
    ):
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"), system_prompt)
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model = model

    async def generate(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, **kwargs
    ) -> str:
        # @retry_with_exponential_backoff
        async def _generate():
            if (
                self.model.startswith("o1")
                or self.model.startswith("o3")
                or self.model.startswith("gpt-5")
            ):
                # o1 and o3 models require max_completion_tokens instead of max_tokens
                create_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
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
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return response.choices[0].message.content

        return await _generate()

    def generate_stream(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, **kwargs
    ):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
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
        system_prompt: Optional[str] = None,
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
        self.messages: List[Dict[str, Any]] = []
        self._registered_tools: Dict[
            str, Dict[str, Any]
        ] = {}  # name -> {"schema": dict, "handler": Callable}
        self._tool_schemas: List[Dict[str, Any]] = []

    @staticmethod
    def _preview(value: Any, limit: int = 400) -> str:
        if value is None:
            return "None"
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned[:limit] + ("…" if len(cleaned) > limit else "")
        try:
            serialized = json.dumps(value, default=str)
        except TypeError:
            serialized = str(value)
        return serialized[:limit] + ("…" if len(serialized) > limit else "")

    def _log_message(self, message: Dict[str, Any], *, prefix: str) -> None:
        role = message.get("role", "unknown")
        preview = self._preview(message.get("content"))
        extras = {k: v for k, v in message.items() if k not in {"role", "content"}}
        extras_json = json.dumps(extras, default=str) if extras else ""
        logger.info("%s role=%s content=%s extras=%s", prefix, role, preview, extras_json)

    def add_message(
        self, role: str, content: Optional[str], **kwargs
    ) -> Dict[str, Any]:
        message = {"role": role, "content": content}
        for key, value in kwargs.items():
            if value is not None:
                message[key] = value
        self.messages.append(message)
        self._log_message(message, prefix="conversation.add")
        return message

    def add_messages(self, messages: List[Dict[str, Any]]):
        for message in messages:
            role = message.get("role")
            if role is None:
                raise ValueError("Each message must contain a 'role'.")
            content = message.get("content")
            extras = {k: v for k, v in message.items() if k not in {"role", "content"}}
            self.add_message(role, content, **extras)

    def clear_messages(self):
        if self.messages:
            logger.info("conversation.clear count=%s", len(self.messages))
        self.messages.clear()

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        schema: Dict[str, Any],
    ):
        if not inspect.iscoroutinefunction(handler):
            raise ValueError(f"Tool '{name}' handler must be async.")
        if schema.get("type") != "function":
            raise ValueError("Tool schema must follow the OpenAI function schema.")
        function_block = schema.get("function") or {}
        if function_block.get("name") and function_block["name"] != name:
            raise ValueError(
                f"Tool schema name '{function_block['name']}' does not match registered name '{name}'."
            )
        self._registered_tools[name] = {"schema": schema, "handler": handler}
        self._refresh_tool_schemas()

    def register_tool_object(self, tool: Any):
        schema = tool.generate_openai_schema()
        self.register_tool(tool.name, tool.__call__, schema)

    def register_tools(
        self,
        tools: List[Dict[str, Any] | Any],
    ):
        for tool in tools:
            if hasattr(tool, "generate_openai_schema") and hasattr(tool, "__call__"):
                self.register_tool_object(tool)
            elif isinstance(tool, dict):
                name = (
                    tool.get("function", {}).get("name")
                    or tool.get("name")
                    or tool.get("id")
                )
                handler = tool.get("handler")
                schema = tool.get("schema") or tool
                if name is None or handler is None:
                    raise ValueError(
                        "Dict-based tool registration requires 'name', 'handler', and 'schema'."
                    )
                self.register_tool(name, handler, schema)
            else:
                raise ValueError(
                    "Each tool must either be a Tool-like object or a dict with schema/handler."
                )

    def clear_tools(self):
        self._registered_tools.clear()
        self._tool_schemas.clear()

    def list_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(self._tool_schemas)

    async def generate_response(
        self,
        user_input: str,
        add_to_history: bool = True,
        input_message_type: str = "user",
        response_model: Optional[BaseModel] = None,
        *,
        run_tools: bool = True,
        include_history: bool = True,
        max_tool_iterations: int = 5,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_executor: Optional[
            Callable[[str, Dict[str, Any], str], Awaitable[Any]]
        ] = None,
        **kwargs,
    ) -> str:
        """
        Generate an assistant response, optionally executing tool calls.

        Args:
            user_input: Latest user message to append to the conversation.
            add_to_history: When False, leaves ConversationManager state untouched after completion.
            input_message_type: Role for the incoming message (defaults to 'user').
            response_model: Optional Pydantic model guiding structured outputs.
            run_tools: When True, automatically executes tool calls returned by the model.
            include_history: When False, ignores existing conversation history during this call.
            max_tool_iterations: Guardrail for recursive tool execution loops.
            tool_choice: Optional explicit tool selection passed to the OpenAI-compatible API.
            tool_executor: Optional override for executing tool calls; receives (name, arguments, call_id).
            **kwargs: Forwarded to the underlying chat completion API.

        Returns:
            The assistant's textual response (empty string if model omits content).
        """
        if response_model:
            schema = response_model.model_json_schema()
            user_input = dedent(
                f"""{user_input}

                Please format your response as a single valid JSON object that matches this schema's *fields*:
                {json.dumps(schema["properties"], indent=2)}

                Rules:
                - The JSON object must include all required fields: {schema.get("required", [])}.
                - Do not include the schema itself, only the field values.
                - Do not include markdown code fences (like ```json), or any text before or after the JSON.
                - Return exactly one JSON object, nothing else.
                """
            )

        if add_to_history and not include_history:
            raise ValueError(
                "Cannot add to history when include_history is False; this call would discard new messages."
            )

        prior_history_count = len(self.messages)

        original_messages: Optional[List[Dict[str, Any]]] = None
        if not include_history:
            original_messages = list(self.messages)
            self.messages = []
        elif not add_to_history:
            original_messages = list(self.messages)

        response_model_name = getattr(response_model, "__name__", None) if response_model else None
        logger.info(
            "conversation.generate start run_tools=%s response_model=%s history=%s",
            run_tools,
            response_model_name,
            prior_history_count,
        )

        if add_to_history:
            self.add_message(input_message_type, user_input)
        else:
            # Still provide context for the LLM without mutating the original history
            temp_message = {"role": input_message_type, "content": user_input}
            self.messages.append(temp_message)
            self._log_message(temp_message, prefix="conversation.temp_add")

        async def _generate_with_retry(max_retries=10, base_backoff=2.0):
            import asyncio
            import re

            for attempt in range(max_retries):
                try:
                    if isinstance(self.llm, ClaudeLLM):
                        if run_tools and self._tool_schemas:
                            raise NotImplementedError(
                                "Tool calls are not yet supported for Claude models in ConversationManager."
                            )
                        # Filter out system messages for Claude API (system prompt is passed separately)
                        claude_messages = [
                            msg for msg in self.messages if msg.get("role") != "system"
                        ]
                        response = self.llm.client.messages.create(
                            model=self.llm.model,
                            system=self.llm.system_prompt,
                            messages=claude_messages,
                            **kwargs,
                        )
                        text = response.content[0].text if response.content else ""
                        logger.info(
                            "conversation.llm_response provider=claude tool_calls=False content=%s",
                            self._preview(text),
                        )
                        return {"role": "assistant", "content": text}
                    elif isinstance(self.llm, GroqLLM):
                        # Groq uses synchronous API but we handle it here
                        messages = [
                            {"role": "system", "content": self.llm.system_prompt}
                        ] + self.messages

                        # Build parameters dict
                        create_params = {
                            "model": self.llm.model,
                            "messages": messages,
                            "max_completion_tokens": kwargs.get("max_tokens", 8192),
                            "temperature": kwargs.get("temperature", 0.7),
                        }

                        if run_tools and self._tool_schemas:
                            create_params["tools"] = self._tool_schemas
                            if tool_choice is not None:
                                create_params["tool_choice"] = tool_choice

                        # Add seed if provided for deterministic results
                        if "seed" in kwargs:
                            create_params["seed"] = kwargs["seed"]

                        # Add any other kwargs except those already handled
                        for k, v in kwargs.items():
                            if k not in [
                                "max_tokens",
                                "temperature",
                                "seed",
                                "reasoning_effort",
                            ]:
                                create_params[k] = v

                        response = self.llm.client.chat.completions.create(
                            **create_params
                        )
                        message = response.choices[0].message
                        preview = self._preview(getattr(message, "content", None))
                        has_tool_calls = bool(getattr(message, "tool_calls", None))
                        logger.info(
                            "conversation.llm_response provider=groq tool_calls=%s content=%s",
                            has_tool_calls,
                            preview,
                        )
                        return self._normalize_assistant_message(
                            message, response_model=response_model
                        )
                    else:
                        messages = [
                            {"role": "system", "content": self.llm.system_prompt}
                        ] + self.messages

                        # Check if this is an o1 or o3 model that requires max_completion_tokens
                        if isinstance(self.llm, OpenAILLM) and (
                            self.llm.model.startswith("o1")
                            or self.llm.model.startswith("o3")
                            or self.llm.model.startswith("gpt-5")
                        ):
                            # o1 and o3 models require max_completion_tokens instead of max_tokens
                            create_params = {
                                "model": self.llm.model,
                                "messages": messages,
                            }

                            # Make a copy of kwargs to avoid modifying the original
                            kwargs_copy = kwargs.copy()

                            # Convert max_tokens to max_completion_tokens for o1/o3 models
                            if "max_tokens" in kwargs_copy:
                                create_params["max_completion_tokens"] = (
                                    kwargs_copy.pop("max_tokens")
                                )

                            # Remove temperature as it's not supported by o1/o3 models
                            if "temperature" in kwargs_copy:
                                kwargs_copy.pop("temperature")

                            # Add remaining kwargs
                            create_params.update(kwargs_copy)

                            response = await self.llm.client.chat.completions.create(
                                **create_params
                            )
                        else:
                            create_params = dict(kwargs)
                            create_params.update(
                                {"model": self.llm.model, "messages": messages}
                            )
                            if run_tools and self._tool_schemas:
                                create_params["tools"] = self._tool_schemas
                                if tool_choice is not None:
                                    create_params["tool_choice"] = tool_choice
                            response = await self.llm.client.chat.completions.create(
                                **create_params
                            )

                        message = response.choices[0].message
                        preview = self._preview(getattr(message, "content", None))
                        has_tool_calls = bool(getattr(message, "tool_calls", None))
                        logger.info(
                            "conversation.llm_response provider=openai tool_calls=%s content=%s",
                            has_tool_calls,
                            preview,
                        )
                        return self._normalize_assistant_message(
                            message, response_model=response_model
                        )

                except Exception as e:
                    # Check if it's a rate limit error
                    is_rate_limit = (
                        "rate" in str(e).lower() and "limit" in str(e).lower()
                    ) or (
                        hasattr(e, "__class__")
                        and "RateLimitError" in str(e.__class__.__name__)
                    )

                    if is_rate_limit and attempt < max_retries - 1:
                        # Extract wait time from error message if available
                        wait_time_match = re.search(
                            r"try again in ([\d.]+)\s*(ms|s)", str(e).lower()
                        )
                        if wait_time_match:
                            wait_time = float(wait_time_match.group(1))
                            if wait_time_match.group(2) == "ms":
                                wait_time = wait_time / 1000.0  # Convert to seconds
                        else:
                            # Use exponential backoff if no specific wait time found
                            wait_time = base_backoff * (2**attempt)

                        logger.warning(
                            "Rate limit encountered attempt=%s wait=%.1fs",
                            attempt + 1,
                            wait_time,
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Re-raise if not a rate limit error or max retries exceeded
                        raise e

            # This should never be reached due to the raise in the except block
            raise Exception("Max retries exceeded")

        try:
            tool_iterations = 0
            final_content: Optional[str] = None

            while True:
                assistant_message = await _generate_with_retry()
                tool_calls = assistant_message.get("tool_calls")
                content = assistant_message.get("content")

                stored_message = self.add_message(
                    "assistant",
                    content,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                )

                if tool_calls and run_tools:
                    await self._execute_tool_calls(
                        tool_calls,
                        tool_executor=tool_executor,
                    )
                    tool_iterations += 1
                    if tool_iterations >= max_tool_iterations:
                        raise RuntimeError(
                            "Maximum tool iterations exceeded without final assistant response."
                        )
                    continue

                final_content = content or ""
                logger.info(
                    "conversation.generate complete tool_calls_executed=%s final_content=%s",
                    tool_iterations,
                    self._preview(final_content),
                )
                break

            return final_content
        finally:
            if original_messages is not None:
                self.messages = original_messages
            logger.info(
                "conversation.generate end history=%s", len(self.messages)
            )

    def _refresh_tool_schemas(self):
        self._tool_schemas = [
            registered["schema"] for registered in self._registered_tools.values()
        ]

    def _normalize_assistant_message(
        self, message: Any, response_model: Optional[BaseModel] = None
    ) -> Dict[str, Any]:
        content = getattr(message, "content", None)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                text = getattr(part, "text", None)
                if text is not None:
                    text_parts.append(text)
                elif isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = "".join(text_parts) if text_parts else None

        if isinstance(content, str) and response_model:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:-3]
            elif cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned[3:-3]
            content = cleaned

        normalized: Dict[str, Any] = {"role": getattr(message, "role", "assistant")}
        normalized["content"] = content

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            normalized["tool_calls"] = []
            for call in tool_calls:
                function = getattr(call, "function", None) or {}
                normalized["tool_calls"].append(
                    {
                        "id": getattr(call, "id", None),
                        "type": getattr(call, "type", "function"),
                        "function": {
                            "name": getattr(function, "name", None)
                            if not isinstance(function, dict)
                            else function.get("name"),
                            "arguments": getattr(function, "arguments", "{}")
                            if not isinstance(function, dict)
                            else function.get("arguments", "{}"),
                        },
                    }
                )
        return normalized

    def _format_tool_result(self, result: Any) -> str:
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        if isinstance(result, str):
            return result
        if result is None:
            return "null"
        try:
            return json.dumps(result)
        except TypeError:
            return str(result)

    async def _invoke_registered_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> Any:
        registered = self._registered_tools.get(name)
        if registered is None:
            raise ValueError(
                f"Tool '{name}' is not registered with ConversationManager."
            )
        handler = registered["handler"]
        return await handler(**arguments)

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        tool_executor: Optional[
            Callable[[str, Dict[str, Any], str], Awaitable[Any]]
        ] = None,
    ):
        for call in tool_calls:
            function_block = call.get("function", {}) if isinstance(call, dict) else {}
            name = function_block.get("name")
            arguments_json = function_block.get("arguments", "{}")
            tool_call_id = call.get("id")

            if not name:
                raise ValueError("Tool call is missing a function name.")

            try:
                arguments = json.loads(arguments_json) if arguments_json else {}
            except json.JSONDecodeError:
                arguments = {"raw_arguments": arguments_json}

            logger.info(
                "conversation.tool_call start name=%s id=%s args=%s",
                name,
                tool_call_id,
                self._preview(arguments),
            )
            if tool_executor is not None:
                result = await tool_executor(name, arguments, tool_call_id)
            else:
                result = await self._invoke_registered_tool(name, arguments)

            content = self._format_tool_result(result)
            logger.info(
                "conversation.tool_call result name=%s id=%s content=%s",
                name,
                tool_call_id,
                self._preview(content),
            )
            self.add_message(
                "tool",
                content,
                name=name,
                tool_call_id=tool_call_id,
            )
