from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from dataset.dataloader import Question
from utils.models import BaseLLM, ConversationManager
from utils.prompt_formatters import (
    build_evidence_based_forecasting_prompt,
    build_evidence_extraction_prompt,
    build_search_query_prompt,
)
from utils.probability_parser import extract_final_probability_with_retry


import asyncio
from agents.btf_utils import (
    BTFQuestion,
    BTFSearchQueries,
    BTFSearchResult,
    BTFSearchFact,
    BTFForecastResult,
    BTFEvidenceExtraction,
    calculate_brier_score,
    calculate_weighted_brier_score,
    load_questions_from_csv,
    fetch_search_results,
    fetch_page_content,
)


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
            search_results=self.evidence,
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


class HybridBTFExpert:
    """Expert that combines Delphi conversation flow with BTF retrieval."""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        user_profile: Optional[dict] = None,
        config: Optional[dict] = None,
        retrieval_config: Optional[dict] = None,
        questions_csv: Optional[Union[str, Path]] = None,
    ) -> None:
        if llm is None:
            raise ValueError("HybridBTFExpert requires an LLM instance")

        self.llm = llm
        self.user_profile = user_profile
        self.config = config or {}
        self.retrieval_config = retrieval_config or {}
        self.conversation_manager = ConversationManager(llm)

        self.prompt_version = self._cfg_get("prompt_version", "v1")
        self.temperature = self._cfg_get("temperature", 0.3)
        self.max_tokens = self._cfg_get("max_tokens", 600)
        self.retry_count = self._cfg_get("probability_retry_count", 2)

        self._questions_csv = Path(questions_csv or "dataset/btf_data/questions.csv")
        self._btf_question_lookup: Optional[Dict[str, BTFQuestion]] = None
        self._retrieval_cache: Dict[
            str, Tuple[List[str], List[BTFSearchResult], List[BTFSearchFact]]
        ] = {}

        self.last_result: Optional[HybridForecastResult] = None

    async def return_forecast_probability(
        self,
        question: Union[Question, BTFQuestion, str],
        conditioning_forecast=None,
        seed: Optional[int] = None,
    ) -> float:
        result = await self.retrieve_then_forecast(
            question,
            conditioning_forecast=conditioning_forecast,
            seed=seed,
        )
        return result.probability

    async def retrieve_then_forecast(
        self,
        question: Union[Question, BTFQuestion, str],
        conditioning_forecast=None,
        seed: Optional[int] = None,
    ) -> HybridForecastResult:
        btf_question = self._ensure_btf_question(question)

        queries = await self.generate_search_queries(btf_question)

        queries, all_results, evidence = await self._retrieve_evidence(
            btf_question, queries
        )

        forecast_prompt = build_evidence_based_forecasting_prompt(
            btf_question,
            evidence=evidence,
            conditioning_forecast=conditioning_forecast,
        )

        response = await self.conversation_manager.generate_response(
            forecast_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=seed,
        )
        response = response.strip()

        probability = await extract_final_probability_with_retry(
            response,
            self.conversation_manager,
            max_retries=max(1, self.retry_count),
        )

        if 0 > probability or probability > 1:
            print(f"Invalid probability: {probability}")
            probability = -1

        resolution = (btf_question.resolution or "").strip().lower() or None
        if resolution in {"yes", "no"} and 0 <= probability <= 1:
            brier_score = calculate_brier_score(probability, resolution)
            weighted_brier = calculate_weighted_brier_score(
                probability,
                resolution,
                btf_question.scoring_weight,
            )
        else:
            brier_score = math.nan
            weighted_brier = math.nan

        result = HybridForecastResult(
            question_id=btf_question.id,
            question=btf_question.question,
            probability=probability,
            response=response,
            search_results=all_results,
            evidence=evidence,
            queries=queries,
            resolution=btf_question.resolution,
            scoring_weight=btf_question.scoring_weight,
            brier_score=brier_score,
            weighted_brier_score=weighted_brier,
        )
        self.last_result = result
        return result

    async def generate_search_queries(self, question: BTFQuestion) -> List[str]:
        """Generate search queries for a given question using the LLM."""
        max_queries = self.retrieval_config.get("max_queries", 3)
        try:
            max_queries = int(max_queries)
        except (TypeError, ValueError):
            max_queries = 3
        if max_queries <= 0:
            max_queries = 3

        prompt = build_search_query_prompt(question)
        query_temperature = self.retrieval_config.get(
            "query_temperature", self.temperature
        )
        query_max_tokens = self.retrieval_config.get("query_max_tokens", 512)
        if not isinstance(query_max_tokens, int) or query_max_tokens <= 0:
            query_max_tokens = 512

        raw_response = await self.conversation_manager.generate_response(
            prompt,
            add_to_history=True,
            response_model=BTFSearchQueries,
            max_tokens=query_max_tokens,
            temperature=query_temperature,
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
                queries = self._parse_queries_from_text(response_text)

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
            if len(cleaned) >= max_queries:
                break

        return cleaned

    @staticmethod
    def _parse_queries_from_text(raw_response: str) -> List[str]:
        """Fallback parser to extract queries from unconstrained text."""
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

    def clear_cache(self) -> None:
        """Erase in-memory caches for retrieval results and question lookups."""

        self._retrieval_cache.clear()
        self._btf_question_lookup = None

    def duplicate(self) -> "HybridBTFExpert":
        """Create a copy with the same configuration but fresh state."""

        return HybridBTFExpert(
            self.llm,
            user_profile=copy.deepcopy(self.user_profile),
            config=copy.deepcopy(self.config),
            retrieval_config=copy.deepcopy(self.retrieval_config),
            questions_csv=self._questions_csv,
        )

    async def _retrieve_evidence(
        self, question: BTFQuestion, queries: List[str]
    ) -> tuple[List[BTFSearchResult], List[BTFSearchFact]]:
        use_cache = self.retrieval_config.get("use_cache", True)
        if use_cache and question.id in self._retrieval_cache:
            return self._retrieval_cache[question.id]

        date_cutoff_start = (
            question.date_cutoff_start
            if question.date_cutoff_start is not None
            else question.present_date
            - timedelta(days=self.retrieval_config.get("days_back", 365))
        )
        date_cutoff_end = question.date_cutoff_end or question.present_date

        nonempty_search_results = []

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
                nonempty_search_results.append(search_result)

        # Extract evidence from all pages concurrently using the new function
        extraction_tasks = []
        for search_result in nonempty_search_results:
            extraction_tasks.append(
                self.extract_evidence_from_page(search_result, question)
            )

        extracted_results = await asyncio.gather(*extraction_tasks)

        # Flatten the results (each extraction returns a list of facts)
        evidence = []
        for fact_list in extracted_results:
            evidence.extend(fact_list)

        max_evidence = self.retrieval_config.get("max_evidence", 0)
        if isinstance(max_evidence, int) and max_evidence > 0:
            evidence = evidence[:max_evidence]

        if use_cache:
            self._retrieval_cache[question.id] = (
                queries,
                nonempty_search_results,
                evidence,
            )

        return queries, nonempty_search_results, evidence

    async def extract_evidence_from_page(
        self, search_result: BTFSearchResult, question: BTFQuestion, max_facts: int = 5
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

        evidence_temperature = self.retrieval_config.get(
            "evidence_temperature", self.temperature
        )
        evidence_max_tokens = self.retrieval_config.get("evidence_max_tokens", 2048)
        if not isinstance(evidence_max_tokens, int) or evidence_max_tokens <= 0:
            evidence_max_tokens = 2048

        raw_response = await self.conversation_manager.generate_response(
            prompt,
            add_to_history=True,
            response_model=BTFEvidenceExtraction,
            max_tokens=evidence_max_tokens,
            temperature=evidence_temperature,
        )
        extracted_facts: List[BTFSearchFact] = []
        if isinstance(raw_response, BTFEvidenceExtraction):
            for fact in raw_response.facts[:max_facts]:
                if fact.strip():  # Only include non-empty facts
                    extracted_facts.append(
                        BTFSearchFact(
                            title=search_result.title,
                            url=search_result.url,
                            fact=fact.strip(),
                        )
                    )
        else:
            response_text = (
                raw_response if isinstance(raw_response, str) else str(raw_response)
            )
            try:
                structured = BTFEvidenceExtraction.model_validate_json(response_text)
                for fact in structured.facts[:max_facts]:
                    if fact.strip():  # Only include non-empty facts
                        extracted_facts.append(
                            BTFSearchFact(
                                title=search_result.title,
                                url=search_result.url,
                                fact=fact.strip(),
                            )
                        )
            except Exception:
                # Fallback: split by lines and take non-empty ones
                for line in response_text.splitlines():
                    line = line.strip()
                    if line and len(extracted_facts) < max_facts:
                        extracted_facts.append(
                            BTFSearchFact(
                                title=search_result.title,
                                url=search_result.url,
                                fact=line,
                            )
                        )

        print(f"Extracted {len(extracted_facts)} facts from {search_result.title}")
        return extracted_facts

    def _ensure_btf_question(
        self, question: Union[Question, BTFQuestion, str]
    ) -> BTFQuestion:
        if isinstance(question, BTFQuestion):
            return question
        if isinstance(question, str):
            lookup = self._load_btf_lookup()
            if question not in lookup:
                raise ValueError(
                    f"Question ID '{question}' not found in {self._questions_csv}"
                )
            return lookup[question]
        if isinstance(question, Question):
            return self._from_dataset_question(question)
        raise TypeError(
            "HybridBTFExpert expects a Question, BTFQuestion, or question ID string"
        )

    def _load_btf_lookup(self) -> Dict[str, BTFQuestion]:
        if self._btf_question_lookup is None:
            if not self._questions_csv.exists():
                raise FileNotFoundError(
                    f"BTF questions CSV not found at {self._questions_csv}"
                )
            questions = load_questions_from_csv(str(self._questions_csv))
            self._btf_question_lookup = {
                question.id: question for question in questions
            }
        return self._btf_question_lookup

    def _from_dataset_question(self, question: Question) -> BTFQuestion:
        present_date = (
            self._safe_parse_datetime(getattr(question, "freeze_datetime", None))
            or datetime.utcnow()
        )
        open_dt = self._safe_parse_datetime(
            getattr(question, "market_info_open_datetime", None)
        )
        start_dt = open_dt or present_date - timedelta(days=365)

        fine_print_parts: List[str] = []
        if getattr(question, "market_info_resolution_criteria", None):
            fine_print_parts.append(str(question.market_info_resolution_criteria))
        if getattr(question, "freeze_datetime_value_explanation", None):
            fine_print_parts.append(str(question.freeze_datetime_value_explanation))
        fine_print = "\n\n".join(fine_print_parts)

        return BTFQuestion(
            id=str(question.id),
            question=question.question,
            background=question.background,
            resolution_criteria=question.resolution_criteria,
            scoring_weight=1.0,
            fine_print=fine_print,
            resolution="unknown",
            resolved_at=str(present_date),
            present_date=present_date,
            date_cutoff_start=start_dt,
            date_cutoff_end=present_date,
        )

    @staticmethod
    def _safe_parse_datetime(
        value: Optional[Union[str, datetime]],
    ) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    def _cfg_get(self, key: str, default):
        cfg = self.config
        try:
            return cfg.get(key, default)
        except AttributeError:
            return getattr(cfg, key, default)
