from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import os
import json
import httpx
from typing import Any
from pydantic import BaseModel, Field
import logging
import csv
from textwrap import dedent
import random
import re

from dotenv import load_dotenv
import asyncio

from utils.prompt_formatters import build_evidence_extraction_prompt
from utils.prompt_formatters import build_search_query_prompt
from utils.prompt_formatters import build_btf_forecast_prompt
from utils.models import ConversationManager

load_dotenv()

CONTENT_CACHE_DIR = Path(os.getenv("BTF_CONTENT_CACHE_DIR", "cache/btf_content"))

RETROSEARCH_API_TOKEN = os.getenv("RETROSEARCH_API_TOKEN")
RETROSEARCH_URL = os.getenv("RETROSEARCH_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# DEFAULT_QUESTIONS_LIMIT = int(os.getenv("BTF_QUESTIONS_LIMIT", "0"))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BTFQuestion(BaseModel):
    """A forecasting question with all relevant information."""

    id: str
    question: str
    background: str
    resolution_criteria: str
    scoring_weight: float
    fine_print: str
    resolution: str  # 'yes' or 'no'
    resolved_at: str
    present_date: datetime
    date_cutoff_start: datetime | None
    date_cutoff_end: datetime


class BTFSearchQueries(BaseModel):
    """Structured response for search query generation."""

    reasoning: str
    queries: list[str]


class BTFProbabilisticForecast(BaseModel):
    """Structured response for probabilistic forecasting."""

    reasoning: str
    probability: float


class BTFEvidenceExtraction(BaseModel):
    """Structured response for evidence extraction."""

    reasoning: str
    facts: list[str]


class BTFSearchResult(BaseModel):
    """A single search result with content."""

    title: str
    url: str
    content: str


class BTFSearchFact(BaseModel):
    """A single search result with extracted facts."""

    title: str
    url: str
    fact: str


class BTFForecastResult(BaseModel):
    """Complete forecast result with all associated data."""

    question_id: str
    question: str
    probability: float
    reasoning: str
    search_results: List[BTFSearchResult] = Field(default_factory=list)
    extracted_facts: List[BTFSearchFact] = Field(default_factory=list)
    fetched_url_content_count: int = 0
    resolution: str = "unknown"
    scoring_weight: float = 1.0
    brier_score: float = float("nan")
    weighted_brier_score: float = float("nan")
    tool_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    messages: List[Dict[str, Any]] = Field(default_factory=list)


@dataclass
class HybridForecastResult:
    """Container for hybrid expert outputs."""

    question_id: str
    question: str
    probability: float
    response: str
    search_results: List[BTFSearchResult] = field(default_factory=list)
    evidence: List[BTFSearchFact] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    resolution: Optional[str] = None
    scoring_weight: float = 1.0
    brier_score: float = math.nan
    weighted_brier_score: float = math.nan
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Union[str, float, int, dict, list]] = field(
        default_factory=dict
    )

    def to_btfforecast(self) -> BTFForecastResult:
        """Return a BTFForecastResult for compatibility with existing tooling."""

        return BTFForecastResult(
            question_id=self.question_id,
            question=self.question,
            probability=self.probability,
            reasoning=self.response,
            search_results=list(self.search_results),
            extracted_facts=list(self.evidence),
            resolution=self.resolution or "unknown",
            scoring_weight=self.scoring_weight,
            brier_score=self.brier_score,
            weighted_brier_score=self.weighted_brier_score,
        )

    def as_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "probability": self.probability,
            "response": self.response,
            "evidence": [result.model_dump() for result in self.evidence],
            "queries": list(self.queries),
            "resolution": self.resolution,
            "scoring_weight": self.scoring_weight,
            "brier_score": self.brier_score,
            "weighted_brier_score": self.weighted_brier_score,
            "generated_at": self.generated_at.isoformat(),
            "metadata": self.metadata,
        }


async def call_anthropic_llm[T: BaseModel](
    prompt: str,
    response_model: type[T] | None = None,
) -> str | T:
    """Call Anthropic API with optional structured response parsing."""

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    if response_model:
        schema = response_model.model_json_schema()
        prompt = dedent(
            f"""{prompt}

            Please format your response as a JSON object that matches this schema:
            {json.dumps(schema, indent=2)}

            Return only valid JSON, no additional text or formatting.
            """
        )

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4000,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        )
        if response.status_code != 200:
            error_text = response.text
            raise Exception(
                f"API call failed with status {response.status_code}: {error_text}"
            )

        result = response.json()
        content = result["content"][0]["text"]

        if response_model:
            # Clean up JSON response
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]

            return response_model.model_validate_json(content)

        return content


async def get_retrosearch_results(
    query: str,
    date_cutoff_start: datetime,
    date_cutoff_end: datetime,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search for content with optional date filtering.
    Returns list of results, but does not include page content."""

    headers = {
        "Authorization": f"Bearer {RETROSEARCH_API_TOKEN}",
        "Content-Type": "application/json",
    }

    params = {
        "query": query,
        "max_results": max_results,
        "date_cutoff_start": date_cutoff_start.isoformat(),
        "date_cutoff_end": date_cutoff_end.isoformat(),
    }

    max_attempts = int(os.getenv("RETROSEARCH_MAX_RETRIES", "3"))
    max_attempts = max(max_attempts, 1)
    backoff = float(os.getenv("RETROSEARCH_RETRY_BACKOFF", "2.0"))

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    f"{RETROSEARCH_URL}/search",
                    headers=headers,
                    json=params,
                )
                if response.status_code != 200:
                    error_text = response.text
                    raise Exception(
                        f"Search failed with status {response.status_code}: {error_text}"
                    )

                result = response.json()
                return result.get("organic_results", [])
        except (
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.RemoteProtocolError,
        ) as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                break
            delay = backoff**attempt
            logger.warning(
                "RetroSearch read timeout (attempt %s/%s) for query '%s'; retrying in %.1fs",
                attempt + 1,
                max_attempts,
                query,
                delay,
            )
            await asyncio.sleep(delay)
        except Exception as exc:
            last_error = exc
            break

    assert last_error is not None
    raise last_error


async def get_result_content(
    url: str,
    date_cutoff_start: datetime | None,
    date_cutoff_end: datetime | None,
) -> dict[str, Any]:
    """Get page content with optional date filtering.

    Returns a dictionary containing the full content, word count metadata,
    and whether the content was truncated for snippet generation.
    """

    cache_key = _make_content_cache_key(url, date_cutoff_start, date_cutoff_end)
    cached = _read_cached_payload(cache_key)
    if cached is not None:
        logger.debug("Content cache hit for %s", url)
        cached.setdefault("cache_key", cache_key)
        content = cached.get("content", "")
        if not cached.get("snippet") and content:
            snippet = _build_snippet(content)
            cached["snippet"] = snippet
            cached.setdefault("snippet_word_count", _word_count(snippet))
            if cached.get("snippet_truncated") is None:
                cached["snippet_truncated"] = len(content) > len(snippet)
        if cached.get("word_count") is None and content:
            cached["word_count"] = _word_count(content)
        cached.setdefault("snippet_word_count", 0)
        cached.setdefault("snippet_truncated", False)
        return cached

    headers = {
        "Authorization": f"Bearer {RETROSEARCH_API_TOKEN}",
        "Content-Type": "application/json",
    }

    params: dict[str, Any] = {"url": url}
    if date_cutoff_end is not None:
        params["date_cutoff_end"] = date_cutoff_end.isoformat()
    if date_cutoff_start is not None:
        params["date_cutoff_start"] = date_cutoff_start.isoformat()

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            f"{RETROSEARCH_URL}/get-page",
            headers=headers,
            json=params,
        )
        if response.status_code != 200:
            error_text = response.text
            raise Exception(
                f"Page fetch failed with status {response.status_code}: {error_text}"
            )

        result = response.json()
        if result.get("status") == "success":
            content = result.get("content", "")
            snippet = _build_snippet(content)
            payload = {
                "url": url,
                "date_cutoff_start": _iso_or_none(date_cutoff_start),
                "date_cutoff_end": _iso_or_none(date_cutoff_end),
                "cached_at": datetime.utcnow().isoformat(),
                "content": content,
                "word_count": _word_count(content),
                "snippet": snippet,
                "snippet_word_count": _word_count(snippet),
                "snippet_truncated": len(content) > len(snippet),
                "cache_key": cache_key,
            }
            _write_cached_content(
                cache_key,
                payload,
            )
            return payload
        else:
            logger.warning(
                f"get_result_content() Failed to fetch content for URL {url}: {result.get('error')}"
            )
            return {
                "url": url,
                "content": "",
                "word_count": 0,
                "snippet": "",
                "snippet_word_count": 0,
                "snippet_truncated": False,
                "cache_key": cache_key,
            }


def _iso_or_none(value: datetime | None) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _make_content_cache_key(
    url: str, date_cutoff_start: datetime | None, date_cutoff_end: datetime | None
) -> str:
    key_payload = json.dumps(
        {
            "url": url,
            "start": _iso_or_none(date_cutoff_start),
            "end": _iso_or_none(date_cutoff_end),
        },
        sort_keys=True,
    )
    return hashlib.sha256(key_payload.encode("utf-8")).hexdigest()


def _get_cache_path(cache_key: str) -> Path:
    return CONTENT_CACHE_DIR / f"{cache_key}.json"


def _read_cached_payload(cache_key: str) -> Optional[dict[str, Any]]:
    path = _get_cache_path(cache_key)
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return payload
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug("Failed to read cache %s: %s", path, exc)
        return None


def _write_cached_content(cache_key: str, payload: dict[str, Any]) -> None:
    try:
        CONTENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _get_cache_path(cache_key)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
        tmp_path.replace(path)
    except Exception as exc:
        logger.warning("Failed to write content cache for key %s: %s", cache_key, exc)


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _build_snippet(content: str, max_chars: int = 2500) -> str:
    if not content:
        return ""
    if len(content) <= max_chars:
        return content
    trimmed = content[:max_chars].rstrip()
    return trimmed + "…"


def load_questions_from_csv(
    csv_path: str = "dataset/btf_data/questions.csv",
    *,
    limit: int | None = None,
    sample: bool = False,
) -> list[BTFQuestion]:
    """Load questions from CSV file.

    Args:
        csv_path: Path to the CSV file.
        limit: Maximum number of questions to return. Defaults to the
            `BTF_QUESTIONS_LIMIT` environment variable when set.
        sample: When True and a limit is provided, randomly sample that many
            distinct questions instead of taking the first N.
    """

    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        questions = [BTFQuestion.model_validate(row) for row in reader]

    effective_limit = limit
    if effective_limit is None:
        effective_limit = len(questions)

    if effective_limit is not None and effective_limit > 0:
        count = min(effective_limit, len(questions))
        if sample:
            questions = random.sample(questions, count)
        else:
            questions = questions[:count]

    return questions


async def extract_evidence_from_page_narrow(
    search_result: BTFSearchResult, question: BTFQuestion, max_facts: int = 5
) -> list[BTFSearchFact]:
    """Extract key facts from a web page that are relevant to the forecasting question.

    Args:
        search_result: The search result containing the page content
        question: The forecasting question for context
        max_facts: Maximum number of facts to extract

    Returns:
        List of BTFSearchResult objects, each containing one extracted fact
    """

    # Limit content to avoid token limits
    content = search_result.content[:65535]

    prompt = build_evidence_extraction_prompt(question, max_facts, content)

    try:
        result = await call_anthropic_llm(prompt, BTFEvidenceExtraction)

        # Create BTFSearchFact objects for each extracted fact
        extracted_facts = []
        for fact in result.facts[:max_facts]:
            if fact.strip():  # Only include non-empty facts
                extracted_facts.append(
                    BTFSearchFact(
                        title=search_result.title,
                        url=search_result.url,
                        fact=fact.strip(),
                    )
                )

        logger.debug(
            f"Extracted {len(extracted_facts)} facts from {search_result.title}"
        )
        return extracted_facts

    except Exception as e:
        logger.warning(f"Evidence extraction failed for {search_result.url}: {e}")
        # Return the original search result if extraction fails
        return [search_result]


async def extract_evidence_from_page(
    conversation_manager: ConversationManager,
    search_result: BTFSearchResult,
    question: BTFQuestion,
    *,
    max_facts: int = 5,
    add_to_history: bool = False,
    run_tools: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> List[BTFSearchFact]:
    """Extract key facts from a web page using the provided conversation manager.

    Args:
        conversation_manager: Conversation manager that wraps the target LLM.
        search_result: The search result containing the page content.
        question: The forecasting question for context.
        max_facts: Maximum number of facts to extract.
        add_to_history: Whether to persist the prompt/response to the conversation log.
        run_tools: Whether tool calls are permitted during extraction.
        temperature: Sampling temperature when calling the LLM.
        max_tokens: Maximum tokens an LLM response may use.
    """

    if conversation_manager is None:
        raise ValueError(
            "A ConversationManager instance is required to extract evidence."
        )

    content = search_result.content[:65535]
    prompt = build_evidence_extraction_prompt(question, max_facts, content)

    effective_max_tokens = (
        max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else 1024
    )

    raw_response = await conversation_manager.generate_response(
        prompt,
        add_to_history=add_to_history,
        include_history=False,
        response_model=BTFEvidenceExtraction,
        max_tokens=effective_max_tokens,
        temperature=temperature,
        run_tools=run_tools,
    )

    facts: List[str] = []
    if isinstance(raw_response, BTFEvidenceExtraction):
        facts = raw_response.facts
    else:
        response_text = (
            raw_response if isinstance(raw_response, str) else str(raw_response)
        )
        try:
            structured = BTFEvidenceExtraction.model_validate_json(response_text)
            facts = structured.facts
        except Exception:
            facts = _parse_facts_from_text(response_text)

    structured_facts: List[BTFSearchFact] = []
    for fact in facts:
        candidate = fact.strip()
        if not candidate:
            continue
        structured_facts.append(
            BTFSearchFact(
                title=search_result.title,
                url=search_result.url,
                fact=candidate,
            )
        )
        if len(structured_facts) >= max_facts:
            break

    if not structured_facts:
        # Mirror narrow behavior by returning original page when no facts are produced.
        return [search_result]

    return structured_facts


async def generate_search_queries_narrow(question: BTFQuestion) -> list[str]:
    """Generate search queries for a given question."""

    prompt = build_search_query_prompt(question)

    result = await call_anthropic_llm(prompt, BTFSearchQueries)
    return result.queries


async def generate_search_queries(
    conversation_manager: ConversationManager,
    question: BTFQuestion,
    *,
    max_queries: Optional[int] = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
    add_to_history: bool = True,
    run_tools: bool = False,
) -> List[str]:
    """Generate search queries for a given question using the provided LLM.

    Args:
        conversation_manager: Conversation manager that wraps the target LLM.
        question: Forecasting question to build queries for.
        max_queries: Optional cap on the number of queries returned (defaults to 3).
        temperature: Sampling temperature when calling the LLM.
        max_tokens: Maximum tokens an LLM response may use.
        add_to_history: Whether to persist the prompt/response to the conversation log.
        run_tools: Whether tool calls are permitted during query generation.
    """

    if conversation_manager is None:
        raise ValueError(
            "A ConversationManager instance is required to generate queries."
        )

    if max_queries is None:
        query_limit: Optional[int] = 3
    else:
        try:
            requested = int(max_queries)
        except (TypeError, ValueError):
            requested = 3
        if requested <= 0:
            query_limit = None
        else:
            query_limit = requested

    prompt = build_search_query_prompt(question)

    effective_max_tokens = (
        max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else 512
    )

    raw_response = await conversation_manager.generate_response(
        prompt,
        add_to_history=add_to_history,
        include_history=False,
        response_model=BTFSearchQueries,
        max_tokens=effective_max_tokens,
        temperature=temperature,
        run_tools=run_tools,
    )

    queries: List[str] = []
    if isinstance(raw_response, BTFSearchQueries):
        queries = raw_response.queries
    else:
        response_text = (
            raw_response if isinstance(raw_response, str) else str(raw_response)
        )
        try:
            structured = BTFSearchQueries.model_validate_json(response_text)
            queries = structured.queries
        except Exception:
            queries = _parse_queries_from_text(response_text)

    cleaned: List[str] = []
    seen: set[str] = set()
    for query in queries:
        candidate = query.strip()
        if not candidate or candidate.lower().startswith("reasoning"):
            continue
        if candidate in seen:
            continue
        cleaned.append(candidate)
        seen.add(candidate)
        if query_limit is not None and len(cleaned) >= query_limit:
            break

    return cleaned


def _parse_queries_from_text(raw_response: str) -> List[str]:
    """Fallback parser to extract queries from unconstrained text output."""
    queries: List[str] = []

    quoted = re.findall(r'"([^"]+)"', raw_response)
    for match in quoted:
        candidate = match.strip()
        if candidate:
            queries.append(candidate)
    if queries:
        return queries

    for line in raw_response.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("reasoning"):
            continue
        line = re.sub(r"^[\d\-\*\.)\s]+", "", line)
        if line:
            queries.append(line)
    return queries


def _parse_facts_from_text(raw_response: str) -> List[str]:
    """Fallback parser to extract fact snippets from unconstrained text output."""
    facts: List[str] = []

    # First look for JSON-like enumerations within brackets
    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            candidates = parsed.get("facts") or parsed.get("items") or []
            if isinstance(candidates, list):
                for item in candidates:
                    if isinstance(item, str):
                        candidate = item.strip()
                        if candidate:
                            facts.append(candidate)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    candidate = item.strip()
                    if candidate:
                        facts.append(candidate)
        if facts:
            return facts
    except Exception:
        pass

    for line in raw_response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("reasoning"):
            continue
        stripped = re.sub(r"^[\d\-\*\.)\s]+", "", stripped)
        if stripped:
            facts.append(stripped)

    return facts


async def fetch_search_results(query, date_cutoff_start, date_cutoff_end):
    try:
        logger.debug(f"Searching for: {query}")
        search_results = await get_retrosearch_results(
            query,
            max_results=5,
            date_cutoff_start=date_cutoff_start,
            date_cutoff_end=date_cutoff_end,
        )
        return query, search_results
    except Exception as e:
        logger.warning(f"Search failed for query '{query}': {e}")
        return query, []


async def fetch_page_content(result, date_cutoff_start, date_cutoff_end):
    try:
        payload = await get_result_content(
            result.get("link", ""),
            date_cutoff_start=date_cutoff_start,
            date_cutoff_end=date_cutoff_end,
        )
        content = payload.get("content", "") if isinstance(payload, dict) else str(payload or "")
        if content:
            content = content[:5000] + "..." if len(content) > 5000 else content
            search_result = BTFSearchResult(
                title=result.get("title", "No title"),
                url=result.get("link", ""),
                content=content,
            )
            logger.debug(f"Successfully fetched: {search_result.title}")
            return search_result
    except Exception as e:
        logger.warning(f"Failed to fetch page {result.get('link', '')}: {e}")
    return None


async def gather_evidence(
    queries: list[str],
    question: BTFQuestion,
    date_cutoff_start: datetime,
    date_cutoff_end: datetime,
) -> list[BTFSearchFact]:
    """Gather evidence by searching and reading pages."""

    all_results = []

    # Gather all search results concurrently
    search_tasks = [
        fetch_search_results(query, date_cutoff_start, date_cutoff_end)
        for query in queries
    ]
    search_results_per_query = await asyncio.gather(*search_tasks)

    # Gather all page fetches concurrently for top 3 results per query
    page_tasks = []
    for _, search_results in search_results_per_query:
        for result in search_results[:3]:
            page_tasks.append(
                fetch_page_content(result, date_cutoff_start, date_cutoff_end)
            )

    page_contents = await asyncio.gather(*page_tasks)

    # Filter out None results and add to all_results
    for search_result in page_contents:
        if search_result is not None:
            all_results.append(search_result)

    # Extract evidence from all pages concurrently using the new function
    extraction_tasks = []
    for search_result in all_results:
        extraction_tasks.append(
            extract_evidence_from_page_narrow(search_result, question)
        )

    extracted_results = await asyncio.gather(*extraction_tasks)

    # Flatten the results (each extraction returns a list of facts)
    final_results = []
    for fact_list in extracted_results:
        final_results.extend(fact_list)

    return final_results


async def make_forecast(
    question: BTFQuestion, evidence: list[BTFSearchFact]
) -> BTFProbabilisticForecast:
    """Make a probabilistic forecast based on the question and evidence."""

    # Prepare evidence text
    evidence_text = "\n".join(
        [
            dedent(
                f"""<item>
                <title>
                {result.title}
                </title>
                <url>
                {result.url}
                </url>
                <content>
                {result.fact}
                </content>
            </item>
            """
            )
            for result in evidence
        ]
    )

    prompt = build_btf_forecast_prompt(question, evidence_text)

    result = await call_anthropic_llm(prompt, BTFProbabilisticForecast)

    # Ensure probability is in valid range
    result.probability = max(0.0, min(1.0, result.probability))

    return result


async def produce_forecast_for_question(
    question: BTFQuestion,
    forecast_semaphore: asyncio.Semaphore | None = None,
) -> BTFForecastResult:
    """Complete forecasting pipeline for a single question."""

    semaphore = forecast_semaphore or asyncio.Semaphore(1)

    async with semaphore:
        logger.info(f"Processing question: {question.id} ({question.question})")

        # Step 1: Generate search queries
        logger.debug("Generating search queries...")
        queries = await generate_search_queries_narrow(question)
        logger.debug(f"Generated {len(queries)} search queries")

        # Step 2: Gather evidence
        logger.debug("Gathering evidence...")
        evidence = await gather_evidence(
            queries,
            question,
            date_cutoff_start=question.date_cutoff_start,
            date_cutoff_end=question.date_cutoff_end,
        )
        logger.debug(f"Gathered {len(evidence)} pieces of evidence")

        # Step 3: Make forecast
        logger.debug("Making forecast...")
        forecast = await make_forecast(question, evidence)
        logger.debug(f"Forecast: {forecast.probability:.3f} probability")

        # Step 4: Calculate Brier score
        brier_score = calculate_brier_score(forecast.probability, question.resolution)
        weighted_brier_score = calculate_weighted_brier_score(
            forecast.probability, question.resolution, question.scoring_weight
        )
        logger.debug(f"Brier score: {brier_score:.4f}")

        # Return complete result
        return BTFForecastResult(
            question_id=question.id,
            question=question.question,
            probability=forecast.probability,
            reasoning=forecast.reasoning,
            search_results=evidence,
            resolution=question.resolution,
            scoring_weight=question.scoring_weight,
            brier_score=brier_score,
            weighted_brier_score=weighted_brier_score,
        )


def calculate_brier_score(forecast_probability: float, actual_outcome: str) -> float:
    """Calculate the Brier score for a forecast.

    Args:
        forecast_probability: Probability assigned to YES outcome (0.0 to 1.0)
        actual_outcome: Actual resolution ('yes' or 'no')

    Returns:
        Brier score (lower is better, 0.0 is perfect)
    """
    actual_value = 1.0 if actual_outcome.lower() == "yes" else 0.0
    return (forecast_probability - actual_value) ** 2


def calculate_weighted_brier_score(
    forecast_probability: float, actual_outcome: str, weight: float
) -> float:
    actual_value = 1.0 if actual_outcome.lower() == "yes" else 0.0
    return weight * (forecast_probability - actual_value) ** 2
