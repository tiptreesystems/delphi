"""Prompt formatting utilities shared across agents."""

from __future__ import annotations

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
            "Some of your notes on this question are: "
            f"{prior_reasoning}\n"
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


def build_in_context_forecast_prompt(question, examples: List[Tuple[object, object]]) -> str:
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
