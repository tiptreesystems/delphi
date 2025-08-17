"""
Utility module for parsing probability values from LLM responses.

This module provides a centralized function for extracting FINAL PROBABILITY values
from text responses, handling various formatting styles including markdown.
"""

import re
from typing import Optional, Protocol, Awaitable


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
    
    # Pattern to match FINAL PROBABILITY with optional markdown formatting
    # Handles: FINAL PROBABILITY:, **FINAL PROBABILITY:**, *FINAL PROBABILITY:*
    probability_pattern = re.compile(
        r'\*{0,2}FINAL PROBABILITY:\*{0,2}\s*(0?\.\d+|1\.0|0|1)', 
        re.IGNORECASE
    )
    
    # Find all matches and take the last one (most recent/final)
    matches = probability_pattern.findall(text)
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
    
    # Pattern to match FINAL PROBABILITY with optional markdown formatting
    probability_pattern = re.compile(
        r'(\*{0,2}FINAL PROBABILITY:\*{0,2}\s*(0?\.\d+|1\.0|0|1))', 
        re.IGNORECASE
    )
    
    # Find all matches with full context
    matches = list(probability_pattern.finditer(text))
    if matches:
        last_match = matches[-1]
        try:
            prob = float(last_match.group(2))
            context = last_match.group(1)  # Full matched string
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
    
    print(f"⚠️  Failed to parse probability from response. Asking LLM to clarify...")
    
    retry_message = (
        "I couldn't find a clear 'FINAL PROBABILITY:' in your response. "
        "Please provide your probability estimate in this exact format:\n\n"
        "FINAL PROBABILITY: [your decimal number between 0 and 1]\n\n"
        "For example: FINAL PROBABILITY: 0.75"
    )
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Retry attempt {attempt + 1}/{max_retries}")
            retry_response = await llm_retry_func.generate_response(
                retry_message, 
                max_tokens=200, 
                temperature=0.1
            )

            print("--------------------------------")
            print(f"Retry response: {retry_response}")
            llm_retry_func.add_message("user", retry_response)
            print(f"LLM messages: {llm_retry_func.messages}")
            print("--------------------------------")

            # Try to parse the retry response
            retry_prob = extract_final_probability(retry_response)
            if retry_prob != -1:
                print(f"✅ Successfully extracted probability on retry: {retry_prob}")
                return retry_prob
            
            print(f"❌ Retry attempt {attempt + 1} still failed to provide valid probability")
            
        except Exception as e:
            print(f"❌ Error during retry attempt {attempt + 1}: {e}")
    
    print(f"❌ All retry attempts failed. Could not extract probability.")
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