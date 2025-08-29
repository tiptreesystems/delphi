"""
Sequential Prompt Learning System V3

This script implements batched updates with epochs:
1. Multiple epochs of training with batched prompt updates
2. Validation after each epoch to track progress
3. Early stopping based on validation performance
"""

import asyncio
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import random

from dataset.dataloader import ForecastDataLoader, Question
from agents.expert import Expert
from agents.mediator import Mediator
from utils.llm_config import get_llm_from_config
from utils.probability_parser import extract_final_probability_with_retry
from utils.prompt_loader import load_prompt
from utils.sampling import sample_questions
from utils.utils import split_train_valid
from prompt_learning import PromptLearner, PredictionRecord




class SequentialLearningPipeline:
    """Pipeline for sequential prompt learning with epochs and batched updates."""

    def __init__(self, config_path: str):
        self.config = yaml.safe_load(open(config_path))
        self.loader = ForecastDataLoader()

        # Initialize models
        self.expert = Expert(
            llm=get_llm_from_config(self.config, 'expert'),
            config=self.config['model'].get('expert', {})
        )

        # Initialize prompt learner
        prompt_version = self.config['model'].get('expert', {}).get('prompt_version', 'v1')
        self.expert_prompt_learner = PromptLearner(
            llm=get_llm_from_config(self.config, 'learner'),
            prompt_version=prompt_version,
            role='expert'
        )

        # Training configuration
        self.n_epochs = self.config.get('training', {}).get('n_epochs', 3)
        self.batch_size = self.config.get('training', {}).get('batch_size', 10)
        self.early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 2)

        # Results storage
        self.epoch_results = defaultdict(lambda: {'train': [], 'valid': []})
        self.best_valid_error = float('inf')
        self.best_epoch = 0
        self.best_prompt = ""

        # Create output directory
        self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)



    async def run_expert_with_learned_prompt(self, question: Question,
                                            resolution_date: str) -> Tuple[float, str]:
        """Run expert with the current learned prompt."""

        # Get base prompt
        base_prompt = load_prompt(
            'expert_forecast',
            'v1',
            question=question.question,
            background=question.background,
            resolution_criteria=question.resolution_criteria,
            url=question.url,
            freeze_datetime_value=question.freeze_datetime_value,
            prior_forecast_info=""
        )

        # Inject learned prompt if available
        if self.expert_prompt_learner.learned_prompt:
            enhanced_prompt = f"""{self.expert_prompt_learner.learned_prompt}

---

Now apply these strategies to the following question:

{base_prompt}"""
        else:
            enhanced_prompt = base_prompt

        # Get prediction
        self.expert.conversation_manager.messages.clear()
        temperature = self.config['model'].get('expert', {}).get('temperature', 0.3)
        response = await self.expert.conversation_manager.generate_response(
            enhanced_prompt,
            max_tokens=self.config['model'].get('expert', {}).get('max_tokens', 6000),
            temperature=temperature
        )

        prob = await extract_final_probability_with_retry(
            response,
            self.expert.conversation_manager,
            max_retries=3
        )

        return prob, response

    async def process_batch(self, questions: List[Question], resolution_date: str,
                           phase: str, epoch: int) -> List[PredictionRecord]:
        """Process a batch of questions without updating the prompt."""
        batch_records = []

        for question in questions:

            # Make prediction with current learned prompt
            pred_prob, reasoning = await self.run_expert_with_learned_prompt(
                question, resolution_date
            )

            # Get actual outcome
            resolution = self.loader.get_resolution(question.id, resolution_date)
            actual_outcome = resolution.resolved_to if resolution and resolution.resolved else None

            # Get superforecaster data
            sf_forecasts = self.loader.get_super_forecasts(
                question_id=question.id,
                resolution_date=resolution_date
            )
            sf_median = np.median([f.forecast for f in sf_forecasts]) if sf_forecasts else None

            # Create record
            record = PredictionRecord(
                question_id=question.id,
                question_text=question.question,
                topic=question.topic,
                predicted_prob=pred_prob,
                actual_outcome=actual_outcome,
                superforecaster_median=sf_median,
                reasoning=reasoning,
                timestamp=datetime.now().isoformat(),
                phase=phase,
                epoch=epoch
            )

            batch_records.append(record)

            # Print progress
            if actual_outcome is not None:
                error = pred_prob - actual_outcome
                print(f"  [{phase.upper()}] {question.question[:50]}... "
                      f"Pred: {pred_prob:.2f}, Actual: {actual_outcome:.2f}, Error: {error:+.3f}")

        return batch_records

    async def run_epoch(self, train_questions: List[Question], valid_questions: List[Question],
                       resolution_date: str, epoch: int):
        """Run a single epoch of training with batched updates."""

        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{self.n_epochs}")
        print("="*60)

        # Shuffle training data for this epoch
        epoch_train = train_questions.copy()
        random.shuffle(epoch_train)

        # Process training data in batches
        train_records = []
        for batch_num, i in enumerate(range(0, len(epoch_train), self.batch_size)):
            batch = epoch_train[i:i + self.batch_size]
            print(f"\n[Batch {batch_num + 1}] Processing {len(batch)} training questions...")

            # Process batch
            batch_records = await self.process_batch(batch, resolution_date, 'train', epoch)
            train_records.extend(batch_records)

            # Calculate and report batch training error
            batch_errors = [abs(r.predicted_prob - r.actual_outcome)
                          for r in batch_records if r.actual_outcome is not None]
            if batch_errors:
                batch_error = np.mean(batch_errors)
                print(f"  → Batch training error: {batch_error:.3f}")

            # Update prompt after each batch
            print(f"Updating prompt based on batch performance...")
            await self.expert_prompt_learner.batch_update_prompt(
                batch_records, epoch, batch_num
            )

            # Save intermediate results
            self.save_intermediate_results(epoch, batch_num)

        # Store training results
        self.epoch_results[epoch]['train'] = train_records

        # Calculate and report overall training error for the epoch
        all_train_errors = [abs(r.predicted_prob - r.actual_outcome)
                           for r in train_records if r.actual_outcome is not None]
        if all_train_errors:
            epoch_train_error = np.mean(all_train_errors)
            print(f"\n✓ Epoch {epoch + 1} Training Error (overall): {epoch_train_error:.3f}")

        # Validation phase
        print(f"\n[Validation] Processing {len(valid_questions)} questions...")
        valid_records = await self.process_batch(valid_questions, resolution_date, 'valid', epoch)
        self.epoch_results[epoch]['valid'] = valid_records

        # Calculate validation metrics
        valid_errors = [abs(r.predicted_prob - r.actual_outcome)
                       for r in valid_records if r.actual_outcome is not None]

        if valid_errors:
            avg_valid_error = np.mean(valid_errors)
            print(f"\n✓ Epoch {epoch + 1} Validation Error: {avg_valid_error:.3f}")

            # Check for improvement
            if avg_valid_error < self.best_valid_error:
                self.best_valid_error = avg_valid_error
                self.best_epoch = epoch
                self.best_prompt = self.expert_prompt_learner.learned_prompt
                print(f"  → New best validation error!")

            return avg_valid_error

        return float('inf')

    async def run_training_loop(self):
        """Run the full training loop with multiple epochs."""

        # Get and prepare data
        all_questions = self.loader.get_questions_with_topics()
        sampled_questions = sample_questions(self.config, all_questions, self.loader)

        # Split into train and validation
        valid_ratio = self.config.get('training', {}).get('valid_ratio', 0.2)
        seed = self.config['experiment']['seed']
        train_questions, valid_questions = split_train_valid(sampled_questions, valid_ratio, seed)

        resolution_date = self.config['data']['resolution_date']

        print(f"\n{'='*60}")
        print("TRAINING CONFIGURATION")
        print(f"  Epochs: {self.n_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Training set: {len(train_questions)} questions")
        print(f"  Validation set: {len(valid_questions)} questions")
        print(f"  Early stopping patience: {self.early_stopping_patience}")
        print("="*60)

        # Training loop or baseline evaluation
        if self.n_epochs == 0:
            # Baseline evaluation - no training, just validation
            print(f"\n{'='*60}")
            print("BASELINE EVALUATION (No Prompt Learning)")
            print("="*60)

            print(f"\n[Baseline Validation] Processing {len(valid_questions)} questions...")
            valid_records = await self.process_batch(valid_questions, resolution_date, 'valid', 0)
            self.epoch_results[0]['valid'] = valid_records

            # Calculate validation metrics
            valid_errors = [abs(r.predicted_prob - r.actual_outcome)
                           for r in valid_records if r.actual_outcome is not None]

            if valid_errors:
                self.best_valid_error = np.mean(valid_errors)
                self.best_epoch = 0
                self.best_prompt = ""  # No learned prompt for baseline
                print(f"\n✓ Baseline Validation Error: {self.best_valid_error:.3f}")
        else:
            # Normal training loop
            patience_counter = 0
            prev_valid_error = float('inf')

            for epoch in range(self.n_epochs):
                # Run epoch
                valid_error = await self.run_epoch(
                    train_questions, valid_questions, resolution_date, epoch
                )

                # Early stopping check
                if valid_error >= prev_valid_error:
                    patience_counter += 1
                    print(f"  No improvement. Patience: {patience_counter}/{self.early_stopping_patience}")

                    if patience_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                        break
                else:
                    patience_counter = 0

                prev_valid_error = valid_error

        # Save final results
        self.save_final_results()

        print(f"\n{'='*60}")
        if self.n_epochs == 0:
            print("BASELINE EVALUATION COMPLETED")
            print(f"  Validation error: {self.best_valid_error:.3f}")
            print(f"  Results saved to: {self.output_dir}")
        else:
            print("TRAINING COMPLETED")
            print(f"  Best epoch: {self.best_epoch + 1}")
            print(f"  Best validation error: {self.best_valid_error:.3f}")
            print(f"  Final prompt saved to: {self.output_dir}")
        print("="*60)

    def save_intermediate_results(self, epoch: int, batch_num: int):
        """Save intermediate results and learned prompt."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save learned prompt
        prompt_path = self.output_dir / f"learned_prompt_epoch{epoch}_batch{batch_num}_{timestamp}.md"
        with open(prompt_path, 'w') as f:
            f.write(self.expert_prompt_learner.learned_prompt)

    def save_final_results(self):
        """Save final results and best prompt."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save best prompt
        best_prompt_path = self.output_dir / f"best_prompt_{timestamp}.md"
        with open(best_prompt_path, 'w') as f:
            f.write(self.best_prompt)

        # Save all results
        results_path = self.output_dir / f"training_results_{timestamp}.json"
        results = {
            'config': self.config,
            'best_epoch': self.best_epoch,
            'best_valid_error': self.best_valid_error,
            'epoch_results': {
                str(k): {
                    'train': [r.to_dict() for r in v['train']],
                    'valid': [r.to_dict() for r in v['valid']]
                }
                for k, v in self.epoch_results.items()
            },
            'evolution_history': self.expert_prompt_learner.evolution_history
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_path}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Sequential Prompt Learning with Epochs')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    pipeline = SequentialLearningPipeline(args.config)
    await pipeline.run_training_loop()


if __name__ == "__main__":
    asyncio.run(main())