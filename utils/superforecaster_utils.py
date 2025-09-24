"""
Superforecaster Utilities

This module provides functionality to load and manage superforecaster reasoning examples
and templates for use in prompt enhancement and evolution.
"""

import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .prompt_loader import load_prompt


@dataclass
class SuperforecasterExample:
    """A single example of superforecaster reasoning and prediction."""

    question: str
    reasoning: str
    forecast: float
    confidence: Optional[str] = None
    base_rate: Optional[float] = None
    key_factors: Optional[List[str]] = None
    resolution: Optional[bool] = None
    topic: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert example to dictionary."""
        return {
            "question": self.question,
            "reasoning": self.reasoning,
            "forecast": self.forecast,
            "confidence": self.confidence,
            "base_rate": self.base_rate,
            "key_factors": self.key_factors,
            "resolution": self.resolution,
            "topic": self.topic,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuperforecasterExample":
        """Create example from dictionary."""
        return cls(
            question=data.get("question", ""),
            reasoning=data.get("reasoning", ""),
            forecast=data.get("forecast", 0.5),
            confidence=data.get("confidence"),
            base_rate=data.get("base_rate"),
            key_factors=data.get("key_factors"),
            resolution=data.get("resolution"),
            topic=data.get("topic"),
        )


class SuperforecasterManager:
    """
    Manages superforecaster examples and reasoning templates for prompt enhancement.

    Integrates with the existing prompt loader system and provides utilities for
    enhancing prompts with superforecaster context.
    """

    def __init__(self, examples_file: Optional[str] = None):
        """
        Initialize the superforecaster manager.

        Args:
            examples_file: Path to JSON file containing superforecaster examples
        """
        self.examples_file = examples_file
        self.examples: List[SuperforecasterExample] = []
        self.reasoning_template = ""

        if examples_file and Path(examples_file).exists():
            self.load_examples(examples_file)

        self.load_reasoning_template()

    def load_examples(self, examples_file: str) -> None:
        """
        Load superforecaster examples from JSON file.

        Args:
            examples_file: Path to JSON file with examples
        """
        try:
            with open(examples_file, "r") as f:
                data = json.load(f)

            self.examples = []
            examples_data = data if isinstance(data, list) else data.get("examples", [])

            for item in examples_data:
                example = SuperforecasterExample.from_dict(item)
                self.examples.append(example)

            print(f"Loaded {len(self.examples)} superforecaster examples")

        except FileNotFoundError:
            print(
                f"Warning: Could not find superforecaster examples file: {examples_file}"
            )
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse superforecaster examples file: {e}")
        except Exception as e:
            print(f"Warning: Error loading superforecaster examples: {e}")

    def load_reasoning_template(self) -> None:
        """Load the superforecaster reasoning template from prompts."""
        try:
            self.reasoning_template = load_prompt(
                "genetic_evolution", "superforecaster_reasoning"
            )
        except Exception as e:
            print(f"Warning: Could not load superforecaster reasoning template: {e}")
            # Fallback template
            self.reasoning_template = """You are an expert forecaster with exceptional calibration. Follow these principles:

1. Consider base rates of similar historical events
2. Identify and analyze key driving factors  
3. Use multiple perspectives and decomposition
4. Generate well-calibrated probability estimates
5. Explain your reasoning systematically

Provide your forecast as a decimal probability between 0 and 1."""

    def get_random_examples(self, n: int = 3) -> List[SuperforecasterExample]:
        """
        Get n random examples from the loaded dataset.

        Args:
            n: Number of examples to return

        Returns:
            List of randomly selected examples
        """
        if not self.examples:
            return []

        return random.sample(self.examples, min(n, len(self.examples)))

    def get_examples_by_topic(
        self, topic: str, n: int = 3
    ) -> List[SuperforecasterExample]:
        """
        Get examples related to a specific topic.

        Args:
            topic: Topic to filter by (searches in question text and topic field)
            n: Number of examples to return

        Returns:
            List of topic-related examples
        """
        topic_lower = topic.lower()
        topic_examples = []

        for ex in self.examples:
            # Check both question text and topic field
            if topic_lower in ex.question.lower() or (
                ex.topic and topic_lower in ex.topic.lower()
            ):
                topic_examples.append(ex)

        if not topic_examples:
            # Fall back to random examples if no topic matches
            return self.get_random_examples(n)

        return random.sample(topic_examples, min(n, len(topic_examples)))

    def format_examples_for_icl(self, examples: List[SuperforecasterExample]) -> str:
        """
        Format examples for in-context learning.

        Args:
            examples: List of examples to format

        Returns:
            Formatted string for ICL
        """
        if not examples:
            return ""

        formatted_examples = []
        for i, example in enumerate(examples, 1):
            formatted = f"""Example {i}:
Question: {example.question}

Reasoning: {example.reasoning}

Forecast: {example.forecast}"""

            # Add optional fields if available
            if example.base_rate is not None:
                formatted += f"\nBase Rate: {example.base_rate}"
            if example.key_factors:
                formatted += f"\nKey Factors: {', '.join(example.key_factors)}"

            formatted_examples.append(formatted)

        return "\n\n".join(formatted_examples)

    def enhance_prompt(
        self,
        base_prompt: str,
        include_reasoning: bool = True,
        include_examples: bool = True,
        n_examples: int = 3,
        topic: Optional[str] = None,
        examples_position: str = "before",  # "before" or "after" base prompt
    ) -> str:
        """
        Enhance a prompt with superforecaster reasoning and examples.

        Args:
            base_prompt: The base prompt to enhance
            include_reasoning: Whether to include reasoning template
            include_examples: Whether to include ICL examples
            n_examples: Number of examples to include
            topic: Specific topic to filter examples by
            examples_position: Where to place examples ("before" or "after")

        Returns:
            Enhanced prompt with superforecaster context
        """
        parts = []

        # Add reasoning template at the beginning if requested
        if include_reasoning and self.reasoning_template:
            parts.append(self.reasoning_template)

        # Prepare ICL examples
        examples_text = ""
        if include_examples and self.examples:
            if topic:
                examples = self.get_examples_by_topic(topic, n_examples)
            else:
                examples = self.get_random_examples(n_examples)

            if examples:
                examples_text = f"Here are examples of expert forecasting:\n\n{self.format_examples_for_icl(examples)}"

        # Add examples before or after base prompt
        if examples_position == "before" and examples_text:
            parts.append(examples_text)
            parts.append(base_prompt)
        else:
            parts.append(base_prompt)
            if examples_text:
                parts.append(examples_text)

        return "\n\n".join(parts)

    def add_example(self, example: SuperforecasterExample) -> None:
        """
        Add a new superforecaster example to the collection.

        Args:
            example: SuperforecasterExample to add
        """
        self.examples.append(example)

    def save_examples(self, filepath: str) -> None:
        """
        Save current examples to JSON file.

        Args:
            filepath: Path to save the examples
        """
        data = {"examples": [ex.to_dict() for ex in self.examples]}

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.examples)} examples to {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded examples."""
        if not self.examples:
            return {"total_examples": 0}

        forecasts = [ex.forecast for ex in self.examples]
        resolved_examples = [ex for ex in self.examples if ex.resolution is not None]
        topics = set(ex.topic for ex in self.examples if ex.topic)

        stats = {
            "total_examples": len(self.examples),
            "mean_forecast": sum(forecasts) / len(forecasts),
            "resolved_examples": len(resolved_examples),
            "unique_topics": len(topics),
            "topics": list(topics),
            "has_reasoning_template": bool(self.reasoning_template),
        }

        if resolved_examples:
            # Calculate accuracy (simplified binary accuracy)
            correct = sum(
                1
                for ex in resolved_examples
                if (ex.forecast > 0.5 and ex.resolution)
                or (ex.forecast <= 0.5 and not ex.resolution)
            )
            stats["accuracy"] = correct / len(resolved_examples)

            # Calculate Brier score
            brier_scores = [
                (ex.forecast - (1 if ex.resolution else 0)) ** 2
                for ex in resolved_examples
            ]
            stats["brier_score"] = sum(brier_scores) / len(brier_scores)

        return stats


# Singleton instance for easy access
_superforecaster_manager: Optional[SuperforecasterManager] = None


def get_superforecaster_manager(
    examples_file: Optional[str] = None,
) -> SuperforecasterManager:
    """
    Get or create the singleton superforecaster manager.

    Args:
        examples_file: Path to examples file (only used on first call)

    Returns:
        SuperforecasterManager instance
    """
    global _superforecaster_manager

    if _superforecaster_manager is None:
        _superforecaster_manager = SuperforecasterManager(examples_file)

    return _superforecaster_manager


def enhance_prompt_with_superforecaster_context(
    prompt: str,
    include_reasoning: bool = True,
    include_examples: bool = True,
    n_examples: int = 3,
    topic: Optional[str] = None,
    examples_file: Optional[str] = None,
) -> str:
    """
    Convenience function to enhance a prompt with superforecaster context.

    Args:
        prompt: Base prompt to enhance
        include_reasoning: Include reasoning template
        include_examples: Include ICL examples
        n_examples: Number of examples
        topic: Topic to filter examples by
        examples_file: Path to examples file

    Returns:
        Enhanced prompt
    """
    manager = get_superforecaster_manager(examples_file)
    return manager.enhance_prompt(
        prompt,
        include_reasoning=include_reasoning,
        include_examples=include_examples,
        n_examples=n_examples,
        topic=topic,
    )
