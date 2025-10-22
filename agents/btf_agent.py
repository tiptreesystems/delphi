from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable

from agents.btf_tools import register_btf_tools
from agents.btf_utils import BTFQuestion
from utils.models import BaseLLM, ConversationManager
from utils.prompt_loader import load_prompt
from utils.probability_parser import extract_final_probability_with_retry


@dataclass
class BTFForecast:
    question_id: str
    probability: float
    response: str
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)


class BTFAgent:
    """Agent wrapper that orchestrates BTF forecasting via tool-enabled LLM calls."""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        prompt_version: str = "v1",
        probability_retry_count: int = 2,
        conversation_manager: Optional[ConversationManager] = None,
        include_query_generator: bool = True,
        max_tool_iterations: int = 5,
    ) -> None:
        self.llm = llm
        self.prompt_version = prompt_version
        self.probability_retry_count = probability_retry_count
        self.conversation_manager = conversation_manager or ConversationManager(llm)
        self.max_tool_iterations = max_tool_iterations
        register_btf_tools(
            self.conversation_manager,
            include_query_generator=include_query_generator,
        )

    async def forecast(
        self,
        question: Union[BTFQuestion, Dict[str, Any]],
        *,
        prior_forecast_info: str = "",
        max_tool_iterations: Optional[int] = None,
        tool_executor: Optional[
            Callable[[str, Dict[str, Any], str], Awaitable[Any]]
        ] = None,
    ) -> BTFForecast:
        btf_question = self._ensure_question(question)
        self.conversation_manager.clear_messages()
        prompt = self._build_prompt(btf_question, prior_forecast_info)

        executor = (
            tool_executor
            if tool_executor is not None
            else self._build_tool_executor(btf_question)
        )
        response_text = await self.conversation_manager.generate_response(
            prompt,
            input_message_type="user",
            run_tools=True,
            max_tool_iterations=max_tool_iterations or self.max_tool_iterations,
            tool_executor=executor,
            temperature=0.3,
        )

        probability = await extract_final_probability_with_retry(
            response_text,
            self.conversation_manager,
            max_retries=self.probability_retry_count,
        )

        return BTFForecast(
            question_id=btf_question.id,
            probability=probability,
            response=response_text,
            tool_outputs=self._collect_tool_outputs(),
            messages=copy.deepcopy(self.conversation_manager.messages),
        )

    def _build_prompt(
        self,
        question: BTFQuestion,
        prior_forecast_info: str,
    ) -> str:
        present_date = question.present_date.isoformat()
        date_cutoff_start = (
            question.date_cutoff_start.isoformat()
            if question.date_cutoff_start is not None
            else "unknown"
        )
        date_cutoff_end = question.date_cutoff_end.isoformat()

        return load_prompt(
            "btf_agent",
            self.prompt_version,
            question=question.question,
            background=question.background,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            present_date=present_date,
            prior_forecast_info=prior_forecast_info,
            date_cutoff_start=date_cutoff_start,
            date_cutoff_end=date_cutoff_end,
        )

    def _collect_tool_outputs(self) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for message in self.conversation_manager.messages:
            if message.get("role") != "tool":
                continue
            raw_content = message.get("content")
            parsed_content: Any
            if isinstance(raw_content, str):
                try:
                    parsed_content = json.loads(raw_content)
                except json.JSONDecodeError:
                    parsed_content = raw_content
            else:
                parsed_content = raw_content

            outputs.append(
                {
                    "name": message.get("name"),
                    "content": parsed_content,
                    "raw": raw_content,
                    "tool_call_id": message.get("tool_call_id"),
                }
            )
        return outputs

    @staticmethod
    def _ensure_question(question: Union[BTFQuestion, Dict[str, Any]]) -> BTFQuestion:
        if isinstance(question, BTFQuestion):
            return question
        if isinstance(question, dict):
            return BTFQuestion.model_validate(question)
        raise TypeError("Question must be a BTFQuestion instance or compatible dict.")

    def _build_tool_executor(
        self, question: BTFQuestion
    ) -> Callable[[str, Dict[str, Any], str], Awaitable[Any]]:
        async def _executor(
            tool_name: str, arguments: Dict[str, Any], tool_call_id: str
        ) -> Any:
            enriched_arguments = self._enrich_tool_arguments(
                tool_name, arguments, question
            )
            entry = self.conversation_manager._registered_tools.get(tool_name)
            if entry is None:
                raise ValueError(f"Unknown tool requested: {tool_name}")
            handler = entry["handler"]
            return await handler(**enriched_arguments)

        return _executor

    def _enrich_tool_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        question: BTFQuestion,
    ) -> Dict[str, Any]:
        entry = self.conversation_manager._registered_tools.get(tool_name)
        if entry is None:
            return arguments

        schema = entry["schema"].get("function", {}).get("parameters", {})
        allowed_params = set(schema.get("properties", {}).keys())
        if not allowed_params:
            return arguments

        enriched = dict(arguments)
        for key, value in self._question_context_map(question).items():
            if key in allowed_params and key not in enriched and value is not None:
                enriched[key] = value
        return enriched

    @staticmethod
    def _question_context_map(question: BTFQuestion) -> Dict[str, Any]:
        return {
            "question_id": question.id,
            "question": question.question,
            "background": question.background,
            "resolution_criteria": question.resolution_criteria,
            "fine_print": question.fine_print,
            "present_date": BTFAgent._ensure_iso(question.present_date),
            "date_cutoff_start": BTFAgent._ensure_iso(question.date_cutoff_start),
            "date_cutoff_end": BTFAgent._ensure_iso(question.date_cutoff_end),
            "scoring_weight": question.scoring_weight,
            "resolution": question.resolution,
            "resolved_at": question.resolved_at,
        }

    @staticmethod
    def _ensure_iso(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)


__all__ = ["BTFAgent", "BTFForecast"]
