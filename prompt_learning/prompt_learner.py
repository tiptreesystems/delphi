"""
Prompt Learning Module for Sequential Learning Pipeline

This module handles the learning and evolution of prompts based on
prediction outcomes from forecasting tasks.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np

from utils.models import BaseLLM, ConversationManager
from utils.prompt_loader import load_prompt


@dataclass
class PredictionRecord:
    """Record of a single prediction and its outcome."""

    question_id: str
    question_text: str
    topic: Optional[str]
    predicted_prob: float
    actual_outcome: Optional[float]
    superforecaster_median: Optional[float]
    reasoning: str
    timestamp: str
    phase: str  # 'train', 'valid'
    epoch: int

    def to_dict(self):
        return asdict(self)


@dataclass
class BatchUpdateRecord:
    """Record for batched prompt updates."""

    predictions: List[PredictionRecord]
    epoch: int
    batch_num: int
    avg_error: float


class PromptLearner:
    """Learns and evolves a strategic prompt based on batched prediction outcomes."""

    def __init__(self, llm: BaseLLM, role: str = "expert", prompt_version: str = "v1"):
        self.llm = llm
        self.conversation_manager = ConversationManager(llm)
        self.role = role
        self.prompt_version = prompt_version
        self.learned_prompt = ""
        self.prediction_history: List[PredictionRecord] = []
        self.evolution_history: List[str] = []
        self.batch_history: List[BatchUpdateRecord] = []

    async def batch_update_prompt(
        self, batch_records: List[PredictionRecord], epoch: int, batch_num: int
    ):
        """Update the learned prompt based on a batch of prediction outcomes."""

        # Filter records with actual outcomes
        valid_records = [r for r in batch_records if r.actual_outcome is not None]
        if not valid_records:
            return

        # Calculate batch statistics
        errors = [(r.predicted_prob - r.actual_outcome) for r in valid_records]
        avg_error = np.mean(errors)
        abs_avg_error = np.mean(np.abs(errors))

        # Build batch update prompt with questions and reasoning
        batch_prompt = self._build_batch_analysis_prompt(
            valid_records, avg_error, abs_avg_error, epoch, batch_num
        )

        # Use LLM to analyze and update the strategic prompt
        self.conversation_manager.messages.clear()
        response = await self.conversation_manager.generate_response(
            batch_prompt, max_tokens=6000, temperature=0.3
        )

        # Apply diff-based update to the learned prompt
        updated_prompt = self._apply_diff_update(response, self.learned_prompt)
        if updated_prompt:  # Only update if we successfully parsed the diff
            self.learned_prompt = updated_prompt
            self.evolution_history.append(self.learned_prompt)

        # Store batch update record
        batch_record = BatchUpdateRecord(
            predictions=valid_records,
            epoch=epoch,
            batch_num=batch_num,
            avg_error=abs_avg_error,
        )
        self.batch_history.append(batch_record)

        # Add to prediction history
        self.prediction_history.extend(valid_records)

    def _build_batch_analysis_prompt(
        self,
        records: List[PredictionRecord],
        avg_error: float,
        abs_avg_error: float,
        epoch: int,
        batch_num: int,
    ) -> str:
        """Build prompt for batch update analysis."""

        # Summarize batch performance
        batch_summary = f"""
Batch Statistics (Epoch {epoch}, Batch {batch_num}):
- Number of predictions: {len(records)}
- Average absolute error: {abs_avg_error:.3f}
- Average directional error: {avg_error:+.3f}
"""

        # Build detailed predictions list with questions and reasoning
        predictions_detail = []
        for i, r in enumerate(records, 1):
            error = r.predicted_prob - r.actual_outcome
            predictions_detail.append(f"""
### Prediction {i}
**Question:** {r.question_text}
**Topic:** {r.topic or "unknown"}
**Predicted:** {r.predicted_prob:.2f}
**Actual:** {r.actual_outcome:.2f}
**Error:** {error:+.3f}
**Reasoning:** {r.reasoning}
""")

        predictions_text = "\n".join(predictions_detail)

        # Use the prompt template
        prompt = load_prompt(
            "prompt_learner",
            "batch_analysis",
            current_guide=self.learned_prompt
            if self.learned_prompt
            else "[No guide developed yet]",
            batch_summary=batch_summary,
            predictions_detail=predictions_text,
            n_predictions=len(records),
        )

        return prompt

    def _apply_diff_update(self, response: str, current_prompt: str) -> str:
        """Apply diff-based updates from LLM response to current prompt."""
        try:
            # Parse simple diff-based updates
            diff_lines = []
            in_diff = False

            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("```") or "YOUR DIFF UPDATE:" in line:
                    in_diff = not in_diff
                    continue
                if in_diff and (line.startswith("+ ") or line.startswith("- ")):
                    diff_lines.append(line)

            # If this is the first prompt (empty), create initial from additions only
            if not current_prompt.strip():
                initial_lines = ["# Forecasting Strategy Guide", ""]
                for diff_line in diff_lines:
                    if diff_line.startswith("+ "):
                        initial_lines.append(diff_line[2:])  # Add without the + prefix
                return "\n".join(initial_lines)

            # Apply the simple diff to existing prompt
            if diff_lines:
                return self._apply_simple_diff(diff_lines, current_prompt)

            # If no diff format found, fall back to extracting complete guide
            return self._extract_learned_prompt(response)

        except Exception as e:
            print(
                f"Warning: Could not parse diff update, keeping current prompt. Error: {e}"
            )
            return current_prompt

    def _apply_simple_diff(self, diff_lines: list, current_prompt: str) -> str:
        """Apply simple +/- diff lines to the current prompt."""
        lines = current_prompt.split("\n")

        # Apply removals first
        for diff_line in diff_lines:
            if diff_line.startswith("- "):
                text_to_remove = diff_line[2:].strip()
                # Remove lines that contain this text
                lines = [line for line in lines if text_to_remove not in line]

        # Then apply additions
        for diff_line in diff_lines:
            if diff_line.startswith("+ "):
                text_to_add = diff_line[2:]
                lines.append(text_to_add)

        return "\n".join(lines)

    def _extract_learned_prompt(self, response: str) -> str:
        """Extract the learned prompt from LLM response (fallback method)."""
        if "# Forecasting Strategy Guide" in response:
            start_idx = response.find("# Forecasting Strategy Guide")
            return response[start_idx:].strip()
        return response.strip()
