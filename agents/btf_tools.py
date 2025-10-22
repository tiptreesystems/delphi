from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

from agents.btf_utils import (
    BTFQuestion,
    BTFSearchFact,
    BTFSearchResult,
    extract_evidence_from_page,
    generate_search_queries,
    get_retrosearch_results,
    get_result_content,
)
from agents.tools.tools import Tool

if TYPE_CHECKING:  # pragma: no cover
    from utils.models import ConversationManager


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
    max_queries: int = 5,
) -> Dict[str, Any]:
    """Generate focused search queries for the provided forecasting question context."""

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

    queries = await generate_search_queries(btf_question)
    limited_queries = queries[:max_queries] if max_queries > 0 else queries
    return {"question_id": question_id, "queries": limited_queries}


async def btf_retro_search(
    query: str,
    date_cutoff_start: Optional[str] = None,
    date_cutoff_end: Optional[str] = None,
    max_results: int = 5,
) -> Dict[str, Any]:
    """Retrieve current web results for a query using the RetroSearch service."""
    end_dt = _parse_datetime(date_cutoff_end)
    start_dt = _parse_datetime(date_cutoff_start)

    results = await get_retrosearch_results(
        query=query,
        date_cutoff_start=start_dt,
        date_cutoff_end=end_dt,
        max_results=max_results,
    )

    return {"query": query, "results": results}


async def btf_fetch_page_content(
    url: str,
    date_cutoff_start: Optional[str] = None,
    date_cutoff_end: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch article content for a specific URL within optional date bounds."""

    end_dt = _parse_datetime(date_cutoff_end)
    start_dt = _parse_datetime(date_cutoff_start)

    content = await get_result_content(
        url=url,
        date_cutoff_start=start_dt,
        date_cutoff_end=end_dt,
    )
    return {"url": url, "content": content}


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
    question_id: str,
    question: str,
    content: str,
    url: str,
    title: str = "",
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

    search_result = BTFSearchResult(
        title=title or url,
        url=url,
        content=content,
    )
    facts = await extract_evidence_from_page(
        search_result=search_result,
        question=btf_question,
        max_facts=max_facts,
    )
    return {
        "question_id": question_id,
        "url": url,
        "facts": _normalize_fact_payload(facts),
    }


def get_btf_tools(include_query_generator: bool = True) -> List[Tool]:
    """Return the suite of BTF-specific tools exposed for agentic forecasting."""

    tools: List[Tool] = [
        Tool.from_function(btf_retro_search),
        Tool.from_function(btf_fetch_page_content),
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

    manager.register_tools(
        get_btf_tools(include_query_generator=include_query_generator)
    )


__all__ = [
    "btf_generate_search_queries",
    "btf_retro_search",
    "btf_fetch_page_content",
    "btf_extract_evidence",
    "get_btf_tools",
    "register_btf_tools",
]
