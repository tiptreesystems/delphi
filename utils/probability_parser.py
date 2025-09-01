"""
Utility module for parsing probability values from LLM responses.

This module provides a centralized function for extracting FINAL PROBABILITY values
from text responses, handling various formatting styles including markdown.
"""

import re
from typing import Optional, Protocol, Awaitable

# Unified regex for FINAL PROBABILITY with optional markdown and scientific notation
# Matches: 0.75, .5, 1, 1.0, 5e-05, 1e-3, 2E-1, etc.
PROB_RE = re.compile(
    r"\*{0,2}FINAL PROBABILITY:\*{0,2}\s*"  # label with optional ** or *
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def extract_final_probability(text: str) -> float:
    """
    Extract the final probability from text containing "FINAL PROBABILITY:" pattern.

    Args:
        text: The text to parse for probability values

    Returns:
        float: The extracted probability value between 0.0 and 1.0, or -1 if not found

    Examples:
        >>> extract_final_probability("FINAL PROBABILITY: 0.75")
        0.75
        >>> extract_final_probability("**FINAL PROBABILITY:** 0.545")
        0.545
        >>> extract_final_probability("*FINAL PROBABILITY:* 0.25")
        0.25
        >>> extract_final_probability("No probability here")
        -1
    """
    if not text or not isinstance(text, str):
        return -1

    # Find all matches and take the last one (most recent/final)
    matches = PROB_RE.findall(text)
    if matches:
        try:
            prob = float(matches[-1])
            # Clamp to valid probability range [0.0, 1.0]
            return max(0.0, min(1.0, prob))
        except ValueError:
            pass


    # No valid probability found
    return -1


def extract_final_probability_with_context(text: str) -> tuple[float, Optional[str]]:
    """
    Extract the final probability and return the surrounding context.

    Args:
        text: The text to parse for probability values

    Returns:
        tuple: (probability, context_string) where context_string contains
               the line with FINAL PROBABILITY or None if not found
    """
    if not text or not isinstance(text, str):
        return -1, None

    # Find all matches with full context
    matches = list(PROB_RE.finditer(text))
    if matches:
        last_match = matches[-1]
        try:
            prob = float(last_match.group(1))
            context = last_match.group(0)  # Full matched string
            return max(0.0, min(1.0, prob)), context
        except ValueError:
            pass

    # Fallback without context
    prob = extract_final_probability(text)
    return prob, None


class LLMRetryProtocol(Protocol):
    """Protocol for LLM objects that can retry generating responses."""
    async def generate_response(self, message: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        ...


async def extract_final_probability_with_retry(
    text: str,
    llm_retry_func: Optional[LLMRetryProtocol] = None,
    max_retries: int = 1
) -> float:
    """
    Extract final probability with retry mechanism if parsing fails.

    Args:
        text: The text to parse for probability values
        llm_retry_func: Optional LLM object that can generate new responses
        max_retries: Maximum number of retry attempts

    Returns:
        float: The extracted probability value between 0.0 and 1.0, or -1 if failed
    """
    # First attempt with original text
    prob = extract_final_probability(text)

    if prob != -1:
        return prob

    # If parsing failed and we have retry capability
    if llm_retry_func is None or max_retries <= 0:
        return -1

    retry_message = "Please provide your probability estimate like 'FINAL PROBABILITY: 0.75'\n\n"

    for attempt in range(max_retries):
        try:
            retry_response = await llm_retry_func.generate_response(
                retry_message,
                max_tokens=100,
                temperature=0.1
            )

            # Try to parse the retry response
            retry_prob = extract_final_probability(retry_response)
            if retry_prob != -1:
                return retry_prob

        except Exception:
            pass

    return -1


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "FINAL PROBABILITY: 0.75",
        "**FINAL PROBABILITY:** 0.545",
        "*FINAL PROBABILITY:* 0.25",
        "The answer is clear.\n\n**FINAL PROBABILITY:** 0.37",
        "No probability here",
        "Random number 0.85 but no final probability",
        "FINAL PROBABILITY: 1.0",
        "FINAL PROBABILITY: 0",
    ]

    print("Testing probability extraction:")
    for test in test_cases:
        result = extract_final_probability(test)
        print(f"'{test}' -> {result}")
