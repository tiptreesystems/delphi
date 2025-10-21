"""Prompt formatting utilities shared across agents."""

from __future__ import annotations

from textwrap import dedent
from typing import List, Optional, Tuple

from utils.prompt_loader import load_prompt

# Local types are referenced by attribute usage only to avoid import cycles
# Question: expects .question, .background, .resolution_criteria, .url,
#           .freeze_datetime_value, .freeze_datetime_value_explanation
# Forecast: expects .reasoning, .forecast


def build_base_forecast_prompt(
    question,
    *,
    prompt_version: str = "v1",
    conditioning_forecast: Optional[object] = None,
) -> str:
    """Render the standard expert forecast prompt with optional conditioning."""

    prior_forecast_info = ""
    if conditioning_forecast is not None:
        prior_reasoning = getattr(conditioning_forecast, "reasoning", "")
        prior_forecast_info = (
            f"Some of your notes on this question are: {prior_reasoning}\n"
        )

    return load_prompt(
        "expert_forecast",
        prompt_version,
        question=getattr(question, "question", ""),
        background=getattr(question, "background", ""),
        resolution_criteria=getattr(question, "resolution_criteria", ""),
        url=getattr(question, "url", None),
        freeze_datetime_value=getattr(question, "freeze_datetime_value", None),
        freeze_datetime_value_explanation=getattr(
            question, "freeze_datetime_value_explanation", None
        ),
        prior_forecast_info=prior_forecast_info,
    )


def build_in_context_forecast_prompt(
    question, examples: List[Tuple[object, object]]
) -> str:
    """
    Build the prompt used by the Expert when forecasting with examples in context.

    The output string mirrors the existing Expert.forecast_with_examples_in_context
    composition so behavior remains unchanged.
    """
    # Examples section
    examples_text = "REFERENCE EXAMPLES OF EXPERT FORECASTS:\n" + "=" * 60 + "\n\n"
    for i, (ex_q, ex_f) in enumerate(examples):
        q_text = getattr(ex_q, "question", "")
        q_bg = getattr(ex_q, "background", "")
        q_res = getattr(ex_q, "resolution_criteria", "")
        f_reason = getattr(ex_f, "reasoning", "")
        f_prob = getattr(ex_f, "forecast", "")
        examples_text += (
            f"[EXAMPLE {i + 1}]\n"
            f"Question: {q_text}\n"
            f"Background: {q_bg}\n"
            f"Resolution: {q_res}\n"
            f"Analysis: {f_reason}\n"
            f"Probability: {f_prob}\n"
            f"{'-' * 40}\n\n"
        )

    # Target question section
    url = getattr(question, "url", None)
    freeze_val = getattr(question, "freeze_datetime_value", None)
    freeze_expl = getattr(question, "freeze_datetime_value_explanation", None)

    target_section = (
        "\n" + "=" * 60 + "\n"
        "YOUR TASK - PROVIDE FORECAST FOR THIS QUESTION:\n" + "=" * 60 + "\n\n"
        f"Question: {getattr(question, 'question', '')}\n"
        f"Background: {getattr(question, 'background', '')}\n"
        f"Resolution: {getattr(question, 'resolution_criteria', '')}\n"
        f"URL: {url}\n"
        f"Freeze value: {freeze_val}\n"
        f"Freeze value explanation: {freeze_expl}\n\n"
    )

    # Instructions
    instructions = (
        "Based on the examples above, provide your forecast concluding with:\n"
        "FINAL PROBABILITY: [decimal between 0 and 1]"
    )

    return examples_text + target_section + instructions


def build_evidence_extraction_prompt(question, max_facts, content):
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

    return prompt


def build_search_query_prompt(question):
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

    return prompt


def build_btf_forecast_prompt(question, evidence_text):
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

    return prompt


# def format_evidence_system_message(
#     question: BTFQuestion,
#     queries: List[str],
#     evidence: List[BTFSearchResult],
# ) -> str:
#     lines = [
#         "You have access to external research used for this forecast.",
#         f"Question ID: {question.id}",
#         "Search queries:",
#     ]
#     lines.extend(f"- {query}" for query in queries)
#     if evidence:
#         lines.append("Evidence snippets:")
#         for idx, item in enumerate(evidence, start=1):
#             lines.append(f"  {idx}. {item.title} ({item.url}) -> {item.content}")
#     return "\n".join(lines)


def build_evidence_based_forecasting_prompt(
    question,
    *,
    evidence,
    prompt_version: str = "v1",
    conditioning_forecast=None,
) -> str:
    prior_forecast_info = ""
    if conditioning_forecast is not None:
        prior_reasoning = getattr(conditioning_forecast, "reasoning", "")
        prior_forecast_info = (
            f"Some of your notes on this question are: {prior_reasoning}\n"
        )

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

    return load_prompt(
        "evidence_based_forecast",
        prompt_version,
        question=getattr(question, "question", ""),
        background=getattr(question, "background", ""),
        resolution_criteria=getattr(question, "resolution_criteria", ""),
        fine_print=getattr(question, "fine_print", ""),
        present_date=getattr(question, "present_date", ""),
        date_cutoff_start=getattr(question, "date_cutoff_start", ""),
        date_cutoff_end=getattr(question, "date_cutoff_end", ""),
        prior_forecast_info=prior_forecast_info,
        evidence_text=evidence_text,
    )
