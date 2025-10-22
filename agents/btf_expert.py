from __future__ import annotations

import math
import copy
from typing import Optional, List, Union
from datetime import datetime, timedelta

from .BTFQuestion import BTFQuestion
from dataset.dataloader import Question
from utils.models import BaseLLM, ConversationManager

from .btf_utils import (
    BTFForecastResult,
    generate_search_queries_narrow,
    gather_evidence,
    make_forecast,
    calculate_brier_score,
    calculate_weighted_brier_score,
)


class BTFExpert:
    """Expert that performs retrieval-augmented BTF-style forecasting.

    This expert uses utilities in `agents/btf_utils.py` to:
    - generate targeted search queries
    - gather evidence via RetroSearch
    - produce a probabilistic forecast via the Anthropic API

    The interface mirrors `Expert.forecast` (returns a probability float) so it can
    be slotted into existing code paths where appropriate.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        user_profile: Optional[dict] = None,
        config: Optional[dict] = None,
    ):
        # The BTF pipeline talks directly to external HTTP APIs; `llm` is optional
        # to keep the interface consistent with other Experts.
        self.llm = llm
        self.user_profile = user_profile
        self.config = config or {}
        self.conversation_manager = (
            ConversationManager(llm) if llm is not None else None
        )

        # Diagnostics similar to Expert
        self.token_warnings: List[str] = []
        self.retry_count: int = 0
        self.last_result: Optional[BTFForecastResult] = None

    async def forecast(
        self,
        question: Union[Question, BTFQuestion],
        conditioning_forecast=None,
        seed: Optional[int] = None,
    ) -> float:
        """Run retrieval + forecasting and return probability in [0,1]."""

        result = await self.forecast_with_details(
            question,
            conditioning_forecast=conditioning_forecast,
            seed=seed,
        )
        return result.probability

    async def forecast_with_details(
        self,
        question: Union[Question, BTFQuestion],
        conditioning_forecast=None,
        seed: Optional[int] = None,
    ) -> BTFForecastResult:
        """Full pipeline returning structured data.

        Conditioning forecasts and seeds are accepted for interface parity but are not
        currently used because the external APIs do not expose deterministic seeding.
        """

        _ = conditioning_forecast, seed  # explicitly unused

        btf_q = self._ensure_btf_question(question)
        result = await self.forecast_btf_question(btf_q)
        self.last_result = result
        return result

    async def forecast_btf_question(self, question: BTFQuestion) -> BTFForecastResult:
        """Convenience: full BTF pipeline for an already-structured BTF question."""
        queries = await generate_search_queries_narrow(question)
        evidence = await gather_evidence(
            queries,
            question,
            date_cutoff_start=question.date_cutoff_start
            or (question.present_date - timedelta(days=365)),
            date_cutoff_end=question.date_cutoff_end,
        )
        forecast = await make_forecast(question, evidence)

        probability = float(max(0.0, min(1.0, forecast.probability)))

        resolution = (question.resolution or "").strip().lower()
        if resolution in {"yes", "no"}:
            brier_score = calculate_brier_score(probability, resolution)
            weighted_brier_score = calculate_weighted_brier_score(
                probability,
                resolution,
                question.scoring_weight,
            )
        else:
            brier_score = math.nan
            weighted_brier_score = math.nan

        # Build a complete result with evidence and scoring metrics
        return BTFForecastResult(
            question_id=question.id,
            question=question.question,
            probability=probability,
            reasoning=forecast.reasoning,
            search_results=evidence,
            resolution=question.resolution,
            scoring_weight=question.scoring_weight,
            brier_score=brier_score,
            weighted_brier_score=weighted_brier_score,
        )

    def duplicate(self) -> "BTFExpert":
        """Return a copy with identical configuration."""

        new = BTFExpert(
            llm=self.llm,
            user_profile=copy.deepcopy(self.user_profile),
            config=copy.deepcopy(self.config),
        )
        return new

    def _ensure_btf_question(
        self, question: Union[Question, BTFQuestion]
    ) -> BTFQuestion:
        if isinstance(question, BTFQuestion):
            return question
        return self._to_btf_question(question)

    def _to_btf_question(self, q: Question) -> BTFQuestion:
        """Map our dataset Question to the BTFQuestion shape expected by utils.

        - present_date/date_cutoff_end use the dataset freeze date when available.
        - date_cutoff_start uses the market open date if provided, otherwise 365 days prior.
        - scoring_weight defaults to 1.0; fine_print pulls from market info fields when present.
        """

        def _parse_dt(value: Optional[str]) -> Optional[datetime]:
            if not value:
                return None
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except Exception:
                return None

        present_date = _parse_dt(q.freeze_datetime) or datetime.utcnow()
        open_dt = _parse_dt(getattr(q, "market_info_open_datetime", None))
        start_dt = open_dt or (present_date - timedelta(days=365))

        fine_print_parts = []
        if getattr(q, "market_info_resolution_criteria", None):
            fine_print_parts.append(str(q.market_info_resolution_criteria))
        if getattr(q, "freeze_datetime_value_explanation", None):
            fine_print_parts.append(str(q.freeze_datetime_value_explanation))
        fine_print = "\n\n".join(fine_print_parts)

        # Resolution fields are unknown at forecast time; placeholders are fine for forecasting.
        return BTFQuestion(
            id=str(q.id),
            question=q.question,
            background=q.background,
            resolution_criteria=q.resolution_criteria,
            scoring_weight=1.0,
            fine_print=fine_print,
            resolution="unknown",
            resolved_at=str(present_date),
            present_date=present_date,
            date_cutoff_start=start_dt,
            date_cutoff_end=present_date,
        )
