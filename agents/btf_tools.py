from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

from agents.btf_utils import (
    BTFQuestion,
    BTFSearchFact,
    BTFSearchResult,
    extract_evidence_from_page as _extract_evidence_with_manager,
    extract_evidence_from_page_narrow,
    generate_search_queries as _generate_search_queries_with_manager,
    generate_search_queries_narrow,
    get_retrosearch_results,
    get_result_content,
)
from agents.tools.tools import Tool

if TYPE_CHECKING:  # pragma: no cover
    from utils.models import ConversationManager


_CONVERSATION_MANAGER: Optional["ConversationManager"] = None


async def generate_search_queries(
    question: BTFQuestion,
    *,
    max_queries: Optional[int] = None,
    add_to_history: bool = False,
) -> List[str]:
    """Generate search queries using the configured conversation manager when available."""

    if _CONVERSATION_MANAGER is None:
        queries = await generate_search_queries_narrow(question)
        if isinstance(max_queries, int) and max_queries > 0:
            return queries[:max_queries]
        return queries

    return await _generate_search_queries_with_manager(
        _CONVERSATION_MANAGER,
        question,
        max_queries=max_queries,
        add_to_history=add_to_history,
        run_tools=False,
    )


async def extract_evidence_from_page(
    search_result: BTFSearchResult,
    question: BTFQuestion,
    *,
    max_facts: int = 5,
    add_to_history: bool = False,
) -> List[BTFSearchFact]:
    """Extract evidence using the configured conversation manager when available."""

    if _CONVERSATION_MANAGER is None:
        return await extract_evidence_from_page_narrow(
            search_result=search_result,
            question=question,
            max_facts=max_facts,
        )

    return await _extract_evidence_with_manager(
        _CONVERSATION_MANAGER,
        search_result,
        question,
        max_facts=max_facts,
        add_to_history=add_to_history,
        run_tools=False,
    )


def _parse_datetime(
    value: Optional[str], *, default: Optional[datetime] = None
) -> Optional[datetime]:
    if value is None:
        return default
    candidate = value.strip()
    if not candidate:
        return default
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return default


def _build_question(
    *,
    question_id: str,
    question: str,
    background: str = "",
    resolution_criteria: str = "",
    fine_print: str = "",
    present_date: Optional[str] = None,
    date_cutoff_start: Optional[str] = None,
    date_cutoff_end: Optional[str] = None,
    scoring_weight: float = 1.0,
    resolution: str = "unknown",
    resolved_at: Optional[str] = None,
) -> BTFQuestion:
    present_dt = _parse_datetime(present_date)
    end_dt = _parse_datetime(date_cutoff_end)
    start_dt = _parse_datetime(date_cutoff_start)

    return BTFQuestion(
        id=question_id,
        question=question,
        background=background,
        resolution_criteria=resolution_criteria,
        scoring_weight=scoring_weight,
        fine_print=fine_print,
        resolution=resolution,
        resolved_at=resolved_at or "",
        present_date=present_dt,
        date_cutoff_start=start_dt,
        date_cutoff_end=end_dt,
    )


async def btf_generate_search_queries(
    question_id: str = "",
    question: str = "",
    background: str = "",
    resolution_criteria: str = "",
    fine_print: str = "",
    present_date: Optional[str] = None,
    date_cutoff_start: Optional[str] = None,
    date_cutoff_end: Optional[str] = None,
    scoring_weight: float = 1.0,
    resolution: str = "unknown",
    resolved_at: Optional[str] = None,
    max_queries: int = 5,
) -> Dict[str, Any]:
    """Generate focused search queries for the provided forecasting question context."""

    if not question_id:
        raise ValueError("`question_id` must be provided for btf_generate_search_queries.")
    if not question:
        raise ValueError("`question` must be provided for btf_generate_search_queries.")

    btf_question = _build_question(
        question_id=question_id,
        question=question,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        present_date=present_date,
        date_cutoff_start=date_cutoff_start,
        date_cutoff_end=date_cutoff_end,
        scoring_weight=scoring_weight,
        resolution=resolution,
        resolved_at=resolved_at,
    )

    queries = await generate_search_queries(
        btf_question,
        max_queries=max_queries,
        add_to_history=False,
    )

    limited_queries = (
        queries[:max_queries]
        if isinstance(max_queries, int) and max_queries > 0
        else queries
    )
    return {"question_id": question_id, "queries": limited_queries}


async def btf_retro_search_urls(
    query: str,
    date_cutoff_start: Optional[str] = None,
    date_cutoff_end: Optional[str] = None,
    max_results: int = 5,
) -> Dict[str, Any]:
    """Retrieve current web result urls for a query using the RetroSearch service."""
    end_dt = _parse_datetime(date_cutoff_end)
    start_dt = _parse_datetime(date_cutoff_start)

    results = await get_retrosearch_results(
        query=query,
        date_cutoff_start=start_dt,
        date_cutoff_end=end_dt,
        max_results=max_results,
    )

    return {"query": query, "results": results}


async def btf_fetch_url_content(
    url: str,
    *,
    date_cutoff_start: Optional[str] = None,
    date_cutoff_end: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch article content for a specific URL within optional date bounds."""

    end_dt = _parse_datetime(date_cutoff_end)
    start_dt = _parse_datetime(date_cutoff_start)

    payload = await get_result_content(
        url=url,
        date_cutoff_start=start_dt,
        date_cutoff_end=end_dt,
    )
    content = payload.get("content", "")
    snippet = payload.get("snippet")
    if not snippet:
        snippet = content[:2500].rstrip()
        if len(content) > len(snippet):
            snippet += "…"
    word_count = payload.get("word_count")
    if word_count is None:
        word_count = len(content.split())
    snippet_word_count = payload.get("snippet_word_count")
    if snippet_word_count is None:
        snippet_word_count = len(snippet.split())
    truncated = payload.get("snippet_truncated")
    if truncated is None:
        truncated = len(payload.get("content", "")) > len(snippet)

    if truncated and word_count and snippet_word_count is not None:
        snippet = (
            f"{snippet}\n\n[Truncated preview: showing {snippet_word_count} "
            f"words out of ~{word_count}. Full content available via "
            f"`btf_extract_evidence`.]"
        )

    return {
        "url": url,
        "snippet": snippet,
        "word_count": word_count,
        "snippet_word_count": snippet_word_count,
        "snippet_truncated": truncated,
        "date_cutoff_start": payload.get("date_cutoff_start"),
        "date_cutoff_end": payload.get("date_cutoff_end"),
        "cached_at": payload.get("cached_at"),
    }


def _normalize_fact_payload(items: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, BTFSearchFact):
            normalized.append(item.model_dump())
        elif isinstance(item, BTFSearchResult):
            normalized.append(
                {
                    "title": item.title,
                    "url": item.url,
                    "content": item.content,
                    "fact": item.content,
                    "type": "search_result",
                }
            )
        else:
            normalized.append({"fact": str(item)})
    return normalized


async def btf_extract_evidence(
    url: str,
    *,
    content: Optional[str] = None,
    title: Optional[str] = None,
    question_id: str = "",
    question: str = "",
    background: str = "",
    resolution_criteria: str = "",
    fine_print: str = "",
    present_date: Optional[str] = None,
    date_cutoff_start: Optional[str] = None,
    date_cutoff_end: Optional[str] = None,
    scoring_weight: float = 1.0,
    resolution: str = "unknown",
    resolved_at: Optional[str] = None,
    max_facts: int = 5,
) -> Dict[str, Any]:
    """Extract decision-relevant facts from page content for a forecasting question."""

    if not url:
        raise ValueError("`url` must be provided for btf_extract_evidence.")
    if not question:
        raise ValueError("`question` must be provided for btf_extract_evidence.")
    if not question_id:
        raise ValueError("`question_id` must be provided for btf_extract_evidence.")

    btf_question = _build_question(
        question_id=question_id,
        question=question,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        present_date=present_date,
        date_cutoff_start=date_cutoff_start,
        date_cutoff_end=date_cutoff_end,
        scoring_weight=scoring_weight,
        resolution=resolution,
        resolved_at=resolved_at,
    )

    end_dt = _parse_datetime(date_cutoff_end)
    start_dt = _parse_datetime(date_cutoff_start)
    page_payload = await get_result_content(
        url=url,
        date_cutoff_start=start_dt,
        date_cutoff_end=end_dt,
    )
    page_content = page_payload.get("content", "")

    search_result = BTFSearchResult(
        title=title or url,
        url=url,
        content=page_content or "",
    )
    facts = await extract_evidence_from_page(
        search_result,
        btf_question,
        max_facts=max_facts,
        add_to_history=False,
    )
    return {
        "question_id": question_id,
        "url": url,
        "facts": _normalize_fact_payload(facts),
    }


def get_btf_tools(include_query_generator: bool = True) -> List[Tool]:
    """Return the suite of BTF-specific tools exposed for agentic forecasting."""

    tools: List[Tool] = [
        Tool.from_function(btf_retro_search_urls),
        Tool.from_function(btf_fetch_url_content),
        Tool.from_function(btf_extract_evidence),
    ]
    if include_query_generator:
        tools.insert(0, Tool.from_function(btf_generate_search_queries))
    return tools


def register_btf_tools(
    manager: "ConversationManager",
    *,
    include_query_generator: bool = True,
) -> None:
    """Register the BTF tool suite on an existing ConversationManager."""

    global _CONVERSATION_MANAGER
    _CONVERSATION_MANAGER = manager

    manager.register_tools(
        get_btf_tools(include_query_generator=include_query_generator)
    )


__all__ = [
    "generate_search_queries",
    "btf_generate_search_queries",
    "btf_retro_search_urls",
    "btf_fetch_url_content",
    "btf_extract_evidence",
    "get_btf_tools",
    "register_btf_tools",
]
