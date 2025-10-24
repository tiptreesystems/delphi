from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Tuple

from agents.btf_tools import register_btf_tools
from agents.btf_utils import (
    BTFQuestion,
    BTFForecastResult,
    BTFSearchResult,
    BTFSearchFact,
    calculate_brier_score,
    calculate_weighted_brier_score,
)
from utils.models import BaseLLM, ConversationManager
from utils.prompt_loader import load_prompt
from utils.probability_parser import extract_final_probability_with_retry


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
    ) -> BTFForecastResult:
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

        (
            tool_outputs,
            search_results,
            extracted_facts,
            fetched_url_count,
        ) = self._collect_tool_data()

        raw_resolution = (btf_question.resolution or "unknown").strip() or "unknown"
        normalized_resolution = raw_resolution.lower()
        if 0.0 <= probability <= 1.0 and normalized_resolution in {"yes", "no"}:
            brier_score = calculate_brier_score(probability, normalized_resolution)
            weighted_brier = calculate_weighted_brier_score(
                probability,
                normalized_resolution,
                btf_question.scoring_weight,
            )
        else:
            brier_score = math.nan
            weighted_brier = math.nan

        return BTFForecastResult(
            question_id=btf_question.id,
            question=btf_question.question,
            probability=probability,
            reasoning=response_text,
            search_results=search_results,
            extracted_facts=extracted_facts,
            fetched_url_count=fetched_url_count,
            resolution=raw_resolution,
            scoring_weight=btf_question.scoring_weight,
            brier_score=brier_score,
            weighted_brier_score=weighted_brier,
            tool_outputs=tool_outputs,
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

    def _collect_tool_data(
        self,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[BTFSearchResult],
        List[BTFSearchFact],
        int,
    ]:
        outputs: List[Dict[str, Any]] = []
        search_results: List[BTFSearchResult] = []
        extracted_facts: List[BTFSearchFact] = []
        fetched_urls: set[str] = set()
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

            tool_name = message.get("name")

            outputs.append(
                {
                    "name": tool_name,
                    "content": parsed_content,
                    "raw": raw_content,
                    "tool_call_id": message.get("tool_call_id"),
                }
            )
            if tool_name == "btf_fetch_url_content":
                if isinstance(parsed_content, dict):
                    url_value = parsed_content.get("url")
                    if url_value:
                        fetched_urls.add(str(url_value))
            self._harvest_structured_tool_data(
                tool_name,
                parsed_content,
                search_results,
                extracted_facts,
            )
        return outputs, search_results, extracted_facts, len(fetched_urls)

    def _harvest_structured_tool_data(
        self,
        tool_name: Optional[str],
        content: Any,
        search_results: List[BTFSearchResult],
        extracted_facts: List[BTFSearchFact],
    ) -> None:
        if not tool_name or content is None:
            return

        if tool_name in {"btf_fetch_page_content", "btf_fetch_url_content"}:
            if isinstance(content, dict):
                url = str(content.get("url") or "")
                text = str(
                    content.get("snippet")
                    or content.get("content")
                    or ""
                )
                title = str(content.get("title") or url or "Page content")
                if url or text:
                    search_results.append(
                        BTFSearchResult(title=title, url=url, content=text)
                    )
        elif tool_name in {"btf_retro_search", "btf_retro_search_urls"}:
            if isinstance(content, dict):
                for item in content.get("results", []):
                    if isinstance(item, dict):
                        title = str(item.get("title") or "Search result")
                        url = str(item.get("link") or item.get("url") or "")
                        snippet = str(
                            item.get("snippet")
                            or item.get("description")
                            or item.get("content")
                            or ""
                        )
                        search_results.append(
                            BTFSearchResult(title=title, url=url, content=snippet)
                        )
        elif tool_name == "btf_extract_evidence":
            if isinstance(content, dict):
                facts_data = content.get("facts", [])
                source_title = str(content.get("title") or "Evidence")
                source_url = str(content.get("url") or content.get("source_url") or "")
                for fact_item in facts_data:
                    if isinstance(fact_item, dict):
                        fact_title = str(fact_item.get("title") or source_title)
                        fact_url = str(fact_item.get("url") or source_url)
                        fact_text = str(
                            fact_item.get("fact") or fact_item.get("content") or ""
                        )
                        extracted_facts.append(
                            BTFSearchFact(
                                title=fact_title, url=fact_url, fact=fact_text
                            )
                        )
                    elif isinstance(fact_item, str):
                        extracted_facts.append(
                            BTFSearchFact(
                                title=source_title, url=source_url, fact=fact_item
                            )
                        )

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
            if key in allowed_params and value is not None:
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


__all__ = ["BTFAgent"]
