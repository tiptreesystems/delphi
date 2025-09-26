import os
import json
import httpx
from datetime import datetime
from typing import Any
from pydantic import BaseModel
import logging
import csv
from textwrap import dedent
import random

from dotenv import load_dotenv
import asyncio

load_dotenv()

RETROSEARCH_API_TOKEN = os.getenv("RETROSEARCH_API_TOKEN")
RETROSEARCH_URL = os.getenv("RETROSEARCH_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


DEFAULT_QUESTIONS_LIMIT = int(os.getenv("BTF_QUESTIONS_LIMIT", "0"))


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


class BTFForecastResult(BaseModel):
    """Complete forecast result with all associated data."""

    question_id: str
    question: str
    probability: float
    reasoning: str
    search_results: list[BTFSearchResult]
    resolution: str
    scoring_weight: float
    brier_score: float
    weighted_brier_score: float


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


async def search(
    query: str,
    date_cutoff_start: datetime,
    date_cutoff_end: datetime,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search for content with optional date filtering."""

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


async def get_page(
    url: str,
    date_cutoff_start: datetime,
    date_cutoff_end: datetime,
) -> str:
    """Get page content with optional date filtering."""

    headers = {
        "Authorization": f"Bearer {RETROSEARCH_API_TOKEN}",
        "Content-Type": "application/json",
    }

    params = {
        "url": url,
        "date_cutoff_end": date_cutoff_end.isoformat(),
        "date_cutoff_start": date_cutoff_start.isoformat(),
    }

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
            return result.get("content", "")
        else:
            raise Exception(
                f"Failed to fetch page {url}: {result.get('error', 'Unknown error')}"
            )
            return ""


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
    if effective_limit is None and DEFAULT_QUESTIONS_LIMIT > 0:
        effective_limit = DEFAULT_QUESTIONS_LIMIT

    if effective_limit is not None and effective_limit > 0:
        count = min(effective_limit, len(questions))
        if sample:
            questions = random.sample(questions, count)
        else:
            questions = questions[:count]

    return questions


async def extract_evidence_from_page(
    search_result: BTFSearchResult, question: BTFQuestion, max_facts: int = 5
) -> list[BTFSearchResult]:
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

    prompt = dedent(
        f"""You are an expert research assistant helping to forecast the outcome of a question.

        <question>
        {question.question}
        </question>

        <background>
        {question.background}
        </background>

        <resolution_criteria>
        {question.resolution_criteria}
        </resolution_criteria>

        <web_page_content>
        {content}
        </web_page_content>

        From the web page content above, extract the most relevant facts that could help answer the forecasting question.

        Focus on:
        - Specific data points, statistics, or measurements
        - Recent developments or announcements
        - Expert opinions or official statements
        - Trends or patterns that relate to the question

        Provide your reasoning for why these facts are relevant, then list exactly {max_facts} facts.
        Each fact should be:
        - Concise (1-2 sentences maximum)
        - Specific and actionable for forecasting
        - Directly related to the question

        If the page contains fewer than {max_facts} relevant facts, provide as many as you can find.
        """
    )

    try:
        result = await call_anthropic_llm(prompt, BTFEvidenceExtraction)

        # Create BTFSearchResult objects for each extracted fact
        extracted_facts = []
        for fact in result.facts[:max_facts]:
            if fact.strip():  # Only include non-empty facts
                extracted_facts.append(
                    BTFSearchResult(
                        title=search_result.title,
                        url=search_result.url,
                        content=fact.strip(),
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


async def generate_search_queries(question: BTFQuestion) -> list[str]:
    """Generate search queries for a given question."""

    prompt = dedent(
        f"""You are a research assistant helping to forecast the outcome of a question.

        <question>
        {question.question}
        </question>

        <background>
        {question.background}
        </background>

        <resolution_criteria>
        {question.resolution_criteria}
        </resolution_criteria>

        <fine-print>
        {question.fine_print}
        </fine-print>

        Today's date is {question.present_date}.

        Generate exactly 3 Google search queries that would help gather relevant information to forecast this question.
        The queries should be:
        1. Specific and targeted to the question
        2. Designed to find recent, relevant information
        3. Diverse in approach to cover different aspects of the question

        Also, provide a brief explanation of your reasoning for selecting these queries.
        """
    )

    result = await call_anthropic_llm(prompt, BTFSearchQueries)
    return result.queries


async def gather_evidence(
    queries: list[str],
    question: BTFQuestion,
    date_cutoff_start: datetime,
    date_cutoff_end: datetime,
) -> list[BTFSearchResult]:
    """Gather evidence by searching and reading pages."""

    all_results = []

    async def fetch_search_results(query):
        try:
            logger.debug(f"Searching for: {query}")
            search_results = await search(
                query,
                max_results=5,
                date_cutoff_start=date_cutoff_start,
                date_cutoff_end=date_cutoff_end,
            )
            return query, search_results
        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")
            return query, []

    async def fetch_page_content(result):
        try:
            content = await get_page(
                result.get("link", ""),
                date_cutoff_start=date_cutoff_start,
                date_cutoff_end=date_cutoff_end,
            )
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

    # Gather all search results concurrently
    search_tasks = [fetch_search_results(query) for query in queries]
    search_results_per_query = await asyncio.gather(*search_tasks)

    # Gather all page fetches concurrently for top 3 results per query
    page_tasks = []
    for _, search_results in search_results_per_query:
        for result in search_results[:3]:
            page_tasks.append(fetch_page_content(result))

    page_contents = await asyncio.gather(*page_tasks)

    # Filter out None results and add to all_results
    for search_result in page_contents:
        if search_result is not None:
            all_results.append(search_result)

    # Extract evidence from all pages concurrently using the new function
    extraction_tasks = []
    for search_result in all_results:
        extraction_tasks.append(extract_evidence_from_page(search_result, question))

    extracted_results = await asyncio.gather(*extraction_tasks)

    # Flatten the results (each extraction returns a list of facts)
    final_results = []
    for fact_list in extracted_results:
        final_results.extend(fact_list)

    return final_results


async def make_forecast(
    question: BTFQuestion, evidence: list[BTFSearchResult]
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
                {result.content}
                </content>
            </item>
            """
            )
            for result in evidence
        ]
    )

    prompt = dedent(
        f"""You are an expert forecaster. Based on the information provided, you need to make a probabilistic forecast.

        <question>
        {question.question}
        </question>

        <background>
        {question.background}
        </background>

        <resolution-criteria>
        {question.resolution_criteria}
        </resolution-criteria>

        <fine-print>
        {question.fine_print}
        </fine-print>

        <evidence>
        {evidence_text}
        </evidence>

        Today's date is {question.present_date}.

        Based on this information, provide:
        1. A probability (between 0.0 and 1.0) that this question will resolve YES
        2. A brief explanation of your reasoning and any calculation used to formulate
           this probability

        Be thoughtful and analytical. Consider both supporting and contradicting evidence.
        Always provide probabilities as decimals between 0.0 and 1.0 (e.g., 0.65 for 65% chance).
        Be calibrated in your forecasts - if you say 70%, you should be right about 70% of the time."""
    )

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
        queries = await generate_search_queries(question)
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
