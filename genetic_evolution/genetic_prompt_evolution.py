"""
Genetic Prompt Evolution Runner

This script implements genetic algorithm-based prompt evolution for the Delphi forecasting system.
It integrates with the existing pipeline and provides an evolutionary alternative to diff-based learning.
"""

import asyncio
import json
import yaml
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import random

from dataset.dataloader import ForecastDataLoader, Question
from agents.expert import Expert
from utils.forecast_loader import load_forecasts
from utils.llm_config import get_llm_from_config
from utils.probability_parser import extract_final_probability_with_retry
from utils.prompt_loader import load_prompt
from utils.sampling import sample_questions
from utils.utils import split_train_valid
from utils.superforecaster_utils import get_superforecaster_manager
from genetic_evolution import GeneticPromptOptimizer, FitnessConfig
from genetic_evolution.fitness_smooth_improvement import (
    evaluate_prompt_smooth_improvement,
    calculate_delphi_fitness,
)
import importlib


# import debugpy
# if not debugpy.is_client_connected():
#     debugpy.listen(5679)
#     print("Waiting for debugger to attach...")
#     debugpy.wait_for_client()
#     print("Debugger attached.")


class GeneticEvolutionPipeline:
    """Pipeline for genetic prompt evolution integrated with Delphi forecasting system."""

    def __init__(self, config_path: str, dry_run: bool = False):
        self.config = yaml.safe_load(open(config_path))
        self.dry_run = dry_run
        self.loader = ForecastDataLoader()

        # Initialize models
        if self.dry_run:
            # Use a lightweight in-process Dummy LLM for zero-network runs
            from utils.models import BaseLLM

            class DummyLLM(BaseLLM):
                async def generate(
                    self,
                    prompt: str,
                    max_tokens: int = 512,
                    temperature: float = 0.0,
                    **kwargs,
                ) -> str:
                    text = (prompt or "").strip().replace("\n", " ")
                    return f"[DRY-RUN MUTATION] {text[:200]}"

                def generate_stream(
                    self,
                    prompt: str,
                    max_tokens: int = 512,
                    temperature: float = 0.0,
                    **kwargs,
                ):
                    if False:
                        yield ""

            dummy_llm = DummyLLM(system_prompt="")
            self.expert = Expert(
                llm=dummy_llm, config=self.config["model"].get("expert", {})
            )
            optimizer_llm = dummy_llm
        else:
            self.expert = Expert(
                llm=get_llm_from_config(self.config, "expert"),
                config=self.config["model"].get("expert", {}),
            )
            optimizer_llm = get_llm_from_config(self.config, "learner")

        # Genetic algorithm configuration
        evolution_config = self.config.get("evolution", {})
        self.population_size = evolution_config.get("population_size", 8)
        self.max_generations = evolution_config.get("max_generations", 40)
        self.elitism_size = evolution_config.get("elitism_size", 2)
        self.tournament_size = evolution_config.get("tournament_size", 3)
        self.initial_mutation_rate = evolution_config.get("initial_mutation_rate", 0.5)

        # Fitness evaluation configuration
        fitness_config = evolution_config.get("fitness", {})
        self.length_penalty_weight = fitness_config.get("length_penalty_weight", 0.1)
        self.target_length = fitness_config.get("target_length", 50)
        self.superforecaster_examples_file = fitness_config.get(
            "superforecaster_examples_file"
        )
        self.include_reasoning = fitness_config.get("include_reasoning", True)
        self.include_examples = fitness_config.get("include_examples", False)
        self.n_examples = fitness_config.get("n_examples", 3)

        # Delphi evaluation configuration
        self.use_delphi_evaluation = fitness_config.get("use_delphi_evaluation", False)
        self.optimize_component = fitness_config.get(
            "optimize_component", "mediator"
        )  # 'expert', 'mediator', or 'both'
        self.use_smooth_improvement = fitness_config.get(
            "use_smooth_improvement", False
        )
        self.variance_weight = fitness_config.get("variance_weight", 0.3)
        self.smoothness_weight = fitness_config.get("smoothness_weight", 0.2)
        self.improvement_weight = fitness_config.get("improvement_weight", 0.5)

        # Optional: override per-question smooth improvement evaluator
        # Default remains `evaluate_prompt_smooth_improvement`.
        self.smooth_improvement_eval_fn_name = (
            fitness_config.get("smooth_improvement_eval_fn")
            or fitness_config.get("smooth_improvement_function")
            or "evaluate_prompt_smooth_improvement"
        )
        self._delphi_per_question_evaluator = self._resolve_smooth_improvement_function(
            self.smooth_improvement_eval_fn_name
        )

        # Training configuration
        training_config = self.config.get("training", {})
        self.validation_batch_size = training_config.get("validation_batch_size", 50)

        # Processing configuration
        processing_config = self.config.get("processing", {})
        self.max_concurrent_mutations = processing_config.get(
            "max_concurrent_mutations", 5
        )

        # Initialize superforecaster manager
        self.superforecaster_manager = None
        if self.superforecaster_examples_file:
            self.superforecaster_manager = get_superforecaster_manager(
                self.superforecaster_examples_file
            )
        else:
            self.superforecaster_manager = get_superforecaster_manager()

        # Create genetic optimizer
        population_config = {
            "population_size": self.population_size,
            "elitism_size": self.elitism_size,
            "tournament_size": self.tournament_size,
            "initial_mutation_rate": self.initial_mutation_rate,
        }

        fitness_config_obj = FitnessConfig(
            length_penalty_weight=self.length_penalty_weight,
            target_length=self.target_length,
            superforecaster_manager=self.superforecaster_manager,
            include_reasoning=self.include_reasoning,
            include_examples=self.include_examples,
            n_examples=self.n_examples,
        )

        # Create timestamped, unique run directory under the configured output_dir
        self.base_output_dir = Path(self.config["experiment"]["output_dir"])
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.run_timestamp}_{uuid.uuid4().hex[:6]}"
        self.output_dir = self.base_output_dir / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # In dry-run, route artifacts to a subfolder
        if self.dry_run:
            self.log_dir = self.output_dir / "dry_run"
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = self.output_dir

        # Persist a copy of the effective configuration used for this run
        try:
            cfg_out = self.log_dir / "config_used.yml"
            with open(cfg_out, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    self.config,
                    f,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True,
                )
            # Also place a copy at output_dir/config_used to meet path requirement
            cfg_out_root = self.output_dir / "config_used"
            with open(cfg_out_root, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    self.config,
                    f,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True,
                )
        except Exception as e:
            print(f"Warning: failed to write config_used.yml: {e}")

        # Optional monitoring parameters
        monitor_params = self.config.get("evolution", {}).get("monitor", {})

        self.optimizer = GeneticPromptOptimizer(
            llm=optimizer_llm,
            population_config=population_config,
            fitness_config=fitness_config_obj,
            log_dir=str(self.log_dir),
            max_concurrent_mutations=self.max_concurrent_mutations,
            component_type=self.optimize_component,
            monitor_params=monitor_params,
        )

        # Results storage
        self.validation_questions = []
        self.best_prompt = ""
        self.evolution_results = {}

    def _resolve_smooth_improvement_function(self, fn_name: str):
        """Resolve a callable by name for per-question Delphi evaluation.

        Resolution order:
        1) genetic_evolution.fitness_smooth_improvement.<fn_name>
        2) Any module in the `genetic_evolution` package exposing <fn_name>
        Fallback to `evaluate_prompt_smooth_improvement` if not found.
        """
        # Fast-path: default
        if not fn_name or fn_name == "evaluate_prompt_smooth_improvement":
            return evaluate_prompt_smooth_improvement

        # 1) Try fitness_smooth_improvement first
        try:
            mod = importlib.import_module(
                "genetic_evolution.fitness_smooth_improvement"
            )
            cand = getattr(mod, fn_name, None)
            if callable(cand):
                print(
                    f"Using custom smooth-improvement evaluator: {fn_name} (from fitness_smooth_improvement)"
                )
                return cand
        except Exception:
            pass

        # 2) Search other modules in genetic_evolution package
        try:
            pkg = importlib.import_module("genetic_evolution")
            pkg_path = Path(pkg.__file__).parent
            for py_file in pkg_path.glob("*.py"):
                name = py_file.stem
                if name.startswith("__"):
                    continue
                try:
                    mod = importlib.import_module(f"genetic_evolution.{name}")
                    cand = getattr(mod, fn_name, None)
                    if callable(cand):
                        print(
                            f"Using custom smooth-improvement evaluator: {fn_name} (from genetic_evolution.{name})"
                        )
                        return cand
                except Exception:
                    continue
        except Exception:
            pass

        print(
            f"Warning: evaluator '{fn_name}' not found; falling back to evaluate_prompt_smooth_improvement"
        )
        return evaluate_prompt_smooth_improvement

    async def run_evolution(self):
        """Run the genetic evolution process."""

        print(f"\n{'=' * 80}")
        print("GENETIC PROMPT EVOLUTION")
        print(f"{'=' * 80}")

        # Get and prepare data, Split into train and validation
        # Use a subset of validation questions for fitness evaluation to speed up evolution
        all_questions = self.loader.get_questions_with_topics()
        sampled_questions = sample_questions(self.config, all_questions, self.loader)

        valid_ratio = self.config.get("training", {}).get("valid_ratio", 0.3)
        seed = self.config["experiment"]["seed"]
        train_questions, valid_questions = split_train_valid(
            sampled_questions, valid_ratio, seed
        )

        print(f"\nEvolution Configuration:")
        print(f"  Population size: {self.population_size}")
        print(f"  Max generations: {self.max_generations}")
        print(f"  Elitism size: {self.elitism_size}")
        print(f"  Tournament size: {self.tournament_size}")
        print(f"  Initial mutation rate: {self.initial_mutation_rate}")
        print(f"  Length penalty weight: {self.length_penalty_weight}")
        print(f"  Include superforecaster reasoning: {self.include_reasoning}")
        print(f"  Include superforecaster examples: {self.include_examples}")

        if self.superforecaster_manager:
            stats = self.superforecaster_manager.get_statistics()
            print(f"  Superforecaster examples loaded: {stats['total_examples']}")
            print(f"  Superforecaster topics: {stats.get('topics', [])}")

        # Create seed prompts
        seed_prompts = self.create_seed_prompts()
        print(f"\nSeed prompts ({len(seed_prompts)}):")
        for i, prompt in enumerate(seed_prompts, 1):
            print(f"  {i}. {prompt}")

        # Prepare train/validation batches
        # Use a subset of training for speed; use full validation set for selection
        batch_size = min(self.validation_batch_size, len(train_questions))
        self.train_batch = random.sample(train_questions, batch_size)
        self.validation_questions = valid_questions
        print(f"\nData splits:")
        print(
            f"  Train batch size (for guidance): {len(self.train_batch)} of {len(train_questions)}"
        )
        print(f"  Validation size (for selection): {len(self.validation_questions)}")

        # Set evaluation functions based on configuration
        if self.dry_run:
            print("Using DRY-RUN evaluator (no external calls).")
            self.optimizer.set_evaluation_functions(
                lambda prompts, _: self._dry_run_evaluate(prompts),
                lambda prompts, _: self._dry_run_evaluate(prompts),
            )
        elif self.use_delphi_evaluation:
            print(
                f"Using Delphi-based fitness evaluation (optimizing: {self.optimize_component})"
            )
            if self.use_smooth_improvement:
                print(
                    f"  Fitness metric: Smooth improvement (variance_weight={self.variance_weight}, "
                    f"smoothness_weight={self.smoothness_weight}, improvement_weight={self.improvement_weight})\n"
                    f"  Evaluator: {getattr(self._delphi_per_question_evaluator, '__name__', 'unknown')}"
                )
            # Train evaluation attaches traces; validation evaluation does not
            self.optimizer.set_evaluation_functions(
                lambda prompts, _: self.evaluate_prompt_fitness_delphi(
                    prompts, self.train_batch, attach_traces=True, split_label="train"
                ),
                lambda prompts, _: self.evaluate_prompt_fitness_delphi(
                    prompts,
                    self.validation_questions,
                    attach_traces=False,
                    split_label="val",
                ),
            )
        else:
            print("Using simple expert-based fitness evaluation")
            self.optimizer.set_evaluation_functions(
                lambda prompts, _: self.evaluate_prompt_fitness(
                    prompts, self.train_batch, attach_traces=True
                ),
                lambda prompts, _: self.evaluate_prompt_fitness(
                    prompts, self.validation_questions, attach_traces=False
                ),
            )

        # Run evolution
        print(f"\nStarting genetic evolution...")
        print(f"{'=' * 80}")

        self.evolution_results = await self.optimizer.evolve(
            seed_prompts=seed_prompts,
            train_batch=self.train_batch,
            val_batch=self.validation_questions,
            max_generations=self.max_generations,
            save_every=5,
        )

        # Get best prompt
        self.best_prompt = self.optimizer.get_best_prompt()

        print(f"\n{'=' * 80}")
        print("EVOLUTION COMPLETED")
        print(f"  Total generations: {self.evolution_results['total_generations']}")
        print(f"  Best fitness: {self.evolution_results['best_fitness']:.3f}")
        print(
            f"  Final mutation rate: {self.evolution_results['final_mutation_rate']:.3f}"
        )
        print(f"{'=' * 80}")

        if not self.dry_run:
            # Evaluate best prompt on validation set
            print(
                f"\nEvaluating best prompt on validation set of ({len(valid_questions)} questions)..."
            )
            await self.evaluate_final_performance(valid_questions, split_label="val")

            # Also evaluate on evolution test set
            evolution_test_questions = sample_questions(
                self.config,
                all_questions,
                self.loader,
                method_override="evolution_evaluation",
            )
            print(
                f"\nEvaluating best prompt on evolution test set of ({len(evolution_test_questions)} questions...)"
            )
            await self.evaluate_final_performance(
                evolution_test_questions, split_label="test"
            )
        else:
            print("\nDRY-RUN: Skipping final validation and test evaluations.")

        # Save results
        self.save_results()

    def create_seed_prompts(self) -> List[str]:
        """Create initial seed prompts for evolution."""
        evolution_config = self.config.get("evolution", {})

        # Check if custom seed prompts are provided
        if "seed_prompts" in evolution_config:
            return evolution_config["seed_prompts"]

        # Create different seed prompts based on what we're optimizing
        if self.optimize_component == "mediator":
            # Mediator-focused seed prompts
            seed_prompts = [
                "Analyze the expert forecasts and provide structured feedback to improve accuracy.",
                "Review the predictions and highlight key considerations that may have been overlooked.",
                "Synthesize the expert opinions and identify areas where reasoning could be strengthened.",
                "Examine the forecasts for potential biases and suggest more calibrated approaches.",
                "Provide constructive feedback to help experts refine their probability estimates.",
                "Guide the experts toward better-calibrated predictions through targeted questions.",
                "Identify inconsistencies in reasoning and promote convergence on well-justified forecasts.",
                "Facilitate expert discussion by highlighting important factors and evidence.",
            ]
        elif self.optimize_component == "both":
            # Mixed prompts for both expert and mediator
            seed_prompts = [
                "Apply systematic forecasting principles with structured reasoning and expert feedback.",
                "Use analytical thinking for predictions while facilitating constructive expert discussion.",
                "Combine careful probability estimation with effective synthesis of multiple viewpoints.",
                "Balance individual forecasting skills with collaborative refinement processes.",
                "Integrate base rate analysis with expert consensus-building techniques.",
                "Apply both predictive reasoning and mediative guidance for optimal outcomes.",
                "Use evidence-based forecasting enhanced by structured expert collaboration.",
                "Combine systematic analysis with effective expert feedback mechanisms.",
            ]
        else:
            # Expert-focused seed prompts (default)
            seed_prompts = [
                "Forecast the probability of this event occurring by considering base rates and key factors.",
                "Predict the outcome by systematically analyzing the question and available information.",
                "Estimate the likelihood by examining historical patterns and current conditions.",
                "Assess the probability using structured forecasting principles and careful reasoning.",
                "Determine the forecast by applying systematic analysis and probabilistic thinking.",
                "Consider base rates, reference classes, and specific factors to predict the probability.",
                "Use analytical reasoning and historical context to forecast the outcome.",
                "Apply forecasting best practices to estimate the probability of this event.",
            ]

        # Take only what we need for population size
        return seed_prompts[: self.population_size]

    async def evaluate_prompt_fitness_delphi(
        self,
        prompts: List[str],
        validation_batch: List[Question],
        is_final: bool = False,
        attach_traces: bool = True,
        split_label: str = "val",
    ):
        """
        Evaluate fitness using full Delphi process with evolved prompts.
        Creates temporary prompt files and uses existing prompt version system.
        """

        def _create_temp_prompt_file(prompt: str, temp_id: str):
            """Create a temporary prompt file for evolved prompt evaluation."""
            from pathlib import Path

            if self.optimize_component == "expert" or self.optimize_component == "both":
                # Create temporary expert system prompt directory and file
                prompt_dir = Path(f"prompts/expert_system/{temp_id}")
                prompt_dir.mkdir(parents=True, exist_ok=True)

                temp_file = prompt_dir / "v1.md"
                with open(temp_file, "w") as f:
                    f.write(prompt)

            if (
                self.optimize_component == "mediator"
                or self.optimize_component == "both"
            ):
                # Create temporary mediator system prompt directory and file
                prompt_dir = Path(f"prompts/{temp_id}")
                prompt_dir.mkdir(parents=True, exist_ok=True)

                temp_file = prompt_dir / "md.md"
                with open(temp_file, "w") as f:
                    f.write(prompt)

        def _cleanup_temp_prompt_file(temp_id: str):
            """Clean up temporary prompt file after evaluation."""
            from pathlib import Path
            import shutil

            if self.optimize_component == "expert" or self.optimize_component == "both":
                temp_dir = Path(f"prompts/expert_system/{temp_id}")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

            if (
                self.optimize_component == "mediator"
                or self.optimize_component == "both"
            ):
                temp_dir = Path(f"prompts/{temp_id}")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        async def _generate_minimal_forecasts(question, config):
            """Generate minimal initial forecasts for evaluation when none exist."""
            from utils.prompt_loader import load_prompt

            n_experts = config["delphi"]["n_experts"]
            llm = get_llm_from_config(config, role="expert")

            # Generate simple initial forecasts
            llmcasts_by_sfid = {}

            for i in range(n_experts):
                sfid = f"eval_expert_{i}"

                # Create a basic forecast prompt
                prompt = load_prompt(
                    "expert_forecast",
                    config["model"].get("expert", {}).get("prompt_version", "v1"),
                    question=question.question,
                    background=question.background,
                    resolution_criteria=question.resolution_criteria,
                    url=question.url,
                    freeze_datetime_value=question.freeze_datetime_value,
                    prior_forecast_info="",
                )

                # Generate a single forecast
                try:
                    response = await llm.generate(
                        prompt,
                        max_tokens=config["model"]
                        .get("expert", {})
                        .get("max_tokens", 6000),
                        temperature=config["model"]
                        .get("expert", {})
                        .get("temperature", 0.3),
                    )

                    # Ensure the response has a valid FINAL PROBABILITY
                    from utils.probability_parser import extract_final_probability

                    prob = extract_final_probability(response)

                    if prob == -1:
                        print("=" * 40)
                        print("=" * 40)
                        print("=" * 40)
                        print(
                            "FALLBACK: No valid FINAL PROBABILITY found in expert response."
                        )
                        print("=" * 40)
                        print("=" * 40)
                        print("=" * 40)
                        # LLM didn't generate valid probability, append one
                        fallback_prob = 0.3 + (
                            i * 0.15
                        )  # Spread experts across 0.3-0.75
                        response += f"\n\nFINAL PROBABILITY: {fallback_prob:.2f}"

                    # Create conversation structure expected by initialize_experts
                    llmcasts_by_sfid[sfid] = [
                        {
                            "full_conversation": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": response},
                            ]
                        }
                    ]

                except Exception as e:
                    print(
                        f"    Warning: Failed to generate forecast for expert {i}: {e}"
                    )
                    # Fallback to a simple mock forecast with guaranteed valid probability
                    fallback_prob = 0.3 + (i * 0.15)
                    llmcasts_by_sfid[sfid] = [
                        {
                            "full_conversation": [
                                {
                                    "role": "user",
                                    "content": f"Forecast for: {question.question}",
                                },
                                {
                                    "role": "assistant",
                                    "content": f"Based on the available information, I estimate the probability at approximately {fallback_prob:.2f}.\n\nFINAL PROBABILITY: {fallback_prob:.2f}",
                                },
                            ]
                        }
                    ]

            return llmcasts_by_sfid

        from runners.delphi_runner import (
            initialize_experts,
            run_delphi_rounds,
            select_experts,
        )

        resolution_date = self.config["data"]["resolution_date"]
        fitness_scores = []
        all_traces: List[str] = []
        all_perf_summaries: List[str] = []
        all_abs_errors: Dict[str, List[float]] = defaultdict(list)
        all_briers: Dict[str, List[float]] = defaultdict(list)
        all_median_superforecast_briers: Dict[str, List[float]] = defaultdict(list)
        all_median_public_briers: Dict[str, List[float]] = defaultdict(list)
        all_delphi_logs: Dict[str, List[Dict]] = defaultdict(list)
        all_question_metrics: Dict[str, List[Dict]] = defaultdict(list)

        print(
            f"Evaluating {len(prompts)} prompts using full Delphi process on {len(validation_batch)} questions..."
        )

        per_prompt_metrics: List[Dict] = []
        for i, prompt in enumerate(prompts):
            print(
                f"  Evaluating prompt {i + 1}/{len(prompts)} via Delphi: {prompt[:50]}..."
            )

            # Debug: Check if base config has mediator with provider
            if i == 0:  # Only print once
                print(
                    f"DEBUG: Base config mediator keys: {list(self.config.get('model', {}).get('mediator', {}).keys())}"
                )

            # Create a stable identifier that captures generation and prompt content
            try:
                import hashlib

                gen_idx = getattr(self.optimizer.population, "generation", 0)
                prompt_hash = hashlib.sha1((prompt or "").encode("utf-8")).hexdigest()[
                    :8
                ]
            except Exception:
                gen_idx = getattr(self.optimizer.population, "generation", 0)
                prompt_hash = f"i{i}"

            # Create temporary prompt file for this evolved prompt (for runtime only)
            # Prefix with a run-specific path to avoid cross-run collisions
            temp_prompt_suffix = (
                f"evolved_g{gen_idx}_i{i}_{prompt_hash}"
                if not is_final
                else f"best_evolved_g{gen_idx}_{prompt_hash}"
            )
            temp_prompt_id = f"_evolution/{self.run_id}/{temp_prompt_suffix}"
            _create_temp_prompt_file(prompt, temp_prompt_id)

            # Also persist the evolved prompt and logs under a generation-aware directory
            prompt_log_base = (
                self.log_dir
                / "evolved_prompts"
                / f"gen_{gen_idx:03d}"
                / f"cand_{i:02d}_{prompt_hash}"
            )
            delphi_logs_dir = prompt_log_base / "delphi_logs" / split_label
            prompt_log_base.mkdir(parents=True, exist_ok=True)
            delphi_logs_dir.mkdir(parents=True, exist_ok=True)
            # Save the prompt text so it can be inspected later
            try:
                with open(prompt_log_base / "prompt.md", "w", encoding="utf-8") as pf:
                    pf.write(prompt)
            except Exception as e:
                print(
                    f"    Warning: failed to write prompt log for {temp_prompt_id}: {e}"
                )

            total_score = 0.0
            valid_predictions = 0
            abs_errors: List[float] = []
            briers: List[float] = []
            median_superforecast_briers: List[float] = []
            median_public_briers: List[float] = []
            prompt_traces: List[str] = []
            delphi_logs: List[Dict] = []
            question_metrics: Dict = {}

            # Create config that uses the evolved prompt
            eval_config = {
                **self.config,
                "delphi": {
                    **self.config.get("delphi", {}),
                    "n_rounds": 2,  # Shorter for faster evaluation
                    "n_experts": min(
                        3, self.config.get("delphi", {}).get("n_experts", 5)
                    ),
                },
            }

            # Ensure model config has provider and name for backward compatibility
            if "provider" not in eval_config["model"]:
                # Use mediator's provider as default (since we're optimizing mediator)
                if (
                    "mediator" in eval_config["model"]
                    and "provider" in eval_config["model"]["mediator"]
                ):
                    eval_config["model"]["provider"] = eval_config["model"]["mediator"][
                        "provider"
                    ]
                    eval_config["model"]["name"] = eval_config["model"]["mediator"].get(
                        "model", ""
                    )
                elif (
                    "expert" in eval_config["model"]
                    and "provider" in eval_config["model"]["expert"]
                ):
                    eval_config["model"]["provider"] = eval_config["model"]["expert"][
                        "provider"
                    ]
                    eval_config["model"]["name"] = eval_config["model"]["expert"].get(
                        "model", ""
                    )

            # Always include full conversation histories in Delphi logs
            try:
                eval_config.setdefault("output", {}).setdefault("save", {})
            except Exception:
                pass

            # Configure which component uses the evolved prompt
            if self.optimize_component == "expert":
                eval_config["model"] = {
                    **eval_config["model"],
                    "expert": {
                        **eval_config["model"].get("expert", {}),
                        "system_prompt_name": f"expert_system/{temp_prompt_id}",
                        "system_prompt_version": "v1",
                    },
                }
            elif self.optimize_component == "mediator":
                # Ensure mediator config has all required fields
                existing_mediator = eval_config["model"].get("mediator", {})
                eval_config["model"] = {
                    **eval_config["model"],
                    "mediator": {
                        **existing_mediator,
                        "system_prompt_name": temp_prompt_id,
                        "system_prompt_version": "md",
                    },
                }
                # Debug: verify mediator config has provider
                if "provider" not in eval_config["model"]["mediator"]:
                    print(
                        f"WARNING: mediator config missing provider. Config keys: {list(eval_config['model']['mediator'].keys())}"
                    )
                    print(f"Full mediator config: {eval_config['model']['mediator']}")
            else:  # both
                eval_config["model"] = {
                    **eval_config["model"],
                    "expert": {
                        **eval_config["model"].get("expert", {}),
                        "system_prompt_name": f"expert_system/{temp_prompt_id}",
                        "system_prompt_version": "v1",
                    },
                    "mediator": {
                        **eval_config["model"].get("mediator", {}),
                        "system_prompt_name": temp_prompt_id,
                        "system_prompt_version": "md",
                    },
                }

            # Load real initial forecasts for this question if available
            initial_forecasts_path = self.config["experiment"]["initial_forecasts_dir"]

            # Try to load existing forecasts
            llm = get_llm_from_config(eval_config, role="expert")
            _, llmcasts_by_qid_sfid, _ = await load_forecasts(
                self.config, self.loader, llm
            )

            # Test prompt on each validation question using full Delphi
            for question in validation_batch:
                # Get forecasts for this specific question
                llmcasts_for_question = llmcasts_by_qid_sfid.get(question.id, {})

                if llmcasts_for_question:
                    # Use real forecasts but limit to evaluation size
                    limited_llmcasts = {}
                    for sfid, forecasts in list(llmcasts_for_question.items())[
                        : eval_config["delphi"]["n_experts"]
                    ]:
                        limited_llmcasts[sfid] = forecasts
                    llmcasts_by_sfid = limited_llmcasts
                else:
                    # No existing forecasts - generate minimal ones for evaluation
                    raise AssertionError("No existing forecasts found")

                # Initialize experts and run Delphi with evolved prompts
                llm = get_llm_from_config(eval_config, role="expert")
                experts = initialize_experts(llmcasts_by_sfid, eval_config, llm)
                experts = select_experts(experts, eval_config)

                # Run Delphi rounds with evolved prompt configuration
                delphi_log = await run_delphi_rounds(question, experts, eval_config, {})
                delphi_logs.append(delphi_log)

                # Persist the per-question Delphi log for this prompt
                try:
                    qlog_path = delphi_logs_dir / f"{question.id}.json"
                    with open(qlog_path, "w", encoding="utf-8") as qf:
                        json.dump(delphi_log, qf, indent=2)
                except Exception as e:
                    print(
                        f"    Warning: failed to write Delphi log for question {question.id}: {e}"
                    )

                # Get ground truth
                resolution = self.loader.get_resolution(question.id, resolution_date)
                actual_outcome = resolution.resolved_to

                if self.use_smooth_improvement:
                    # Use smooth improvement fitness (configurable evaluator)
                    fitness_score, metrics = self._delphi_per_question_evaluator(
                        delphi_log, actual_outcome
                    )
                    question_metrics[question.id] = metrics
                    # Be robust to custom metric keys
                    _tot_imp = metrics.get("total_improvement")
                    _var = metrics.get("improvement_variance")
                    _smooth = metrics.get("smoothness") or metrics.get(
                        "smoothness_score"
                    )
                    if (
                        _tot_imp is not None
                        and _var is not None
                        and _smooth is not None
                    ):
                        print(
                            f"      Q{question.id[:8]}: improvement={_tot_imp:.3f}, "
                            f"variance={_var:.3f}, smoothness={_smooth:.3f}, "
                            f"fitness={fitness_score:.4f}"
                        )
                    else:
                        print(
                            f"      Q{question.id[:8]}: custom fitness={fitness_score:.4f}"
                        )

                    # Create trace with improvement details if available
                    rounds_trace = []
                    median_briers = metrics.get("median_briers_by_round") or []
                    improvements_by_round = metrics.get("improvements_by_round") or []
                    for i, (brier, imp) in enumerate(
                        zip(median_briers, [0] + list(improvements_by_round))
                    ):
                        try:
                            rounds_trace.append(
                                f"R{i + 1}: Brier={float(brier):.3f}, Δ={float(imp):+.3f}"
                            )
                        except Exception:
                            pass

                    if rounds_trace:
                        ti = _tot_imp if _tot_imp is not None else 0.0
                        sm = _smooth if _smooth is not None else 0.0
                        prompt_traces.append(
                            f"Question: {question.question[:60]}...\n"
                            f"Rounds: {' → '.join(rounds_trace)}\n"
                            f"Total Improvement: {ti:.3f}, Smoothness: {sm:.3f}"
                        )
                else:
                    # Original fitness calculation
                    final_round = delphi_log["rounds"][-1]
                    final_probs = [
                        expert["prob"] for expert in final_round["experts"].values()
                    ]
                    pred_prob = np.median(final_probs)

                    # Calculate Brier score (lower is better, so we negate it for fitness)
                    brier_score = (pred_prob - actual_outcome) ** 2
                    fitness_score = (
                        1.0 - brier_score
                    )  # Convert to fitness (higher is better)

                    print(
                        f"      Q{question.id[:8]}: pred={pred_prob:.3f}, actual={actual_outcome:.1f}, brier={brier_score:.4f}, fitness={fitness_score:.4f}"
                    )

                    # Create trace
                    rounds_trace = []
                    for round_data in delphi_log["rounds"]:
                        round_probs = [
                            exp["prob"] for exp in round_data["experts"].values()
                        ]
                        rounds_trace.append(
                            f"Round {round_data['round']}: {np.median(round_probs):.3f}"
                        )

                    prompt_traces.append(
                        f"Question: {question.question[:60]}...\n"
                        f"Rounds: {' → '.join(rounds_trace)}\n"
                        f"Final: {pred_prob:.3f} vs Actual: {actual_outcome:.3f} (Brier: {brier_score:.3f})"
                    )

                total_score += fitness_score
                valid_predictions += 1

                # Track for summary
                final_round = delphi_log["rounds"][-1]
                final_probs = [
                    expert["prob"] for expert in final_round["experts"].values()
                ]
                pred_prob = np.median(final_probs)
                abs_errors.append(abs(pred_prob - actual_outcome))
                briers.append((pred_prob - actual_outcome) ** 2)
                super_forecasts = self.loader.get_super_forecasts(
                    question_id=question.id, resolution_date=resolution_date
                )
                super_forecast_values = [sf.forecast for sf in super_forecasts]
                median_superforecast = np.median(super_forecast_values)
                median_superforecast_brier = (
                    median_superforecast - actual_outcome
                ) ** 2
                median_superforecast_briers.append(median_superforecast_brier)
                public_forecasts = self.loader.get_public_forecasts(
                    question_id=question.id, resolution_date=resolution_date
                )
                public_forecast_values = [pf.forecast for pf in public_forecasts]
                median_public_forecast = np.median(public_forecast_values)
                median_public_brier = (median_public_forecast - actual_outcome) ** 2
                median_public_briers.append(median_public_brier)
                print(
                    f"        Superforecaster median: {median_superforecast:.3f}, pred_prob: {pred_prob:.3f}, actual: {actual_outcome:.1f}"
                )

            all_abs_errors[temp_prompt_id].extend(abs_errors)
            all_briers[temp_prompt_id].extend(briers)
            all_median_superforecast_briers[temp_prompt_id].extend(
                median_superforecast_briers
            )
            all_median_public_briers[temp_prompt_id].extend(median_public_briers)
            all_delphi_logs[temp_prompt_id].extend(delphi_logs)
            all_question_metrics[temp_prompt_id] = question_metrics

            # Clean up temporary prompt file
            _cleanup_temp_prompt_file(temp_prompt_id)

            # Calculate average fitness for this prompt
            if valid_predictions > 0:
                avg_fitness = total_score / valid_predictions
            else:
                avg_fitness = 0.0

            fitness_scores.append(avg_fitness)
            avg_brier = 1.0 - avg_fitness if valid_predictions > 0 else 1.0
            print(
                f"    Delphi fitness: {avg_fitness:.3f} (from {valid_predictions} questions"
            )

            # Performance summary
            if abs_errors:
                abs_avg_error = float(np.mean(abs_errors))
                avg_brier = float(np.mean(briers))
                batch_summary_avg_error = f"Delphi predictions: {valid_predictions}\nAverage absolute error: {abs_avg_error:.3f}"
                batch_summary_avg_brier = f"Average Brier score: {avg_brier:.3f}"
                if median_superforecast_briers:
                    avg_median_sf_brier = float(np.mean(median_superforecast_briers))
                    batch_summary_avg_brier += f"\nAverage median superforecaster brier: {avg_median_sf_brier:.3f}"
                batch_summary = f"{batch_summary_avg_error}\n{batch_summary_avg_brier}"
            else:
                batch_summary = "No valid Delphi predictions."
            all_perf_summaries.append(batch_summary)
            all_traces.append("\n\n".join(prompt_traces[-3:]))

            # Build per-prompt metric summary
            metrics_dict = {
                "num_questions": valid_predictions,
                "mean_abs_error": float(np.mean(abs_errors)) if abs_errors else None,
                "mean_brier": float(np.mean(briers)) if briers else None,
                "sf_mean_brier": float(np.mean(median_superforecast_briers))
                if median_superforecast_briers
                else None,
                "public_mean_brier": float(np.mean(median_public_briers))
                if median_public_briers
                else None,
            }

            # Compute smooth improvement metrics from Delphi logs
            try:
                outcomes = {}
                for qid, qm in all_question_metrics[temp_prompt_id].items():
                    if qm.get("actual_outcome") is not None:
                        outcomes[qid] = qm["actual_outcome"]
                if delphi_logs and outcomes:
                    delphi_metrics = calculate_delphi_fitness(
                        delphi_logs,
                        outcomes,
                        variance_weight=self.variance_weight,
                        smoothness_weight=self.smoothness_weight,
                        improvement_weight=self.improvement_weight,
                    )
                    metrics_dict.update(
                        {
                            "delphi_total_improvement": float(
                                delphi_metrics.total_improvement
                            ),
                            "delphi_improvement_variance": float(
                                delphi_metrics.improvement_variance
                            ),
                            "delphi_smoothness": float(delphi_metrics.smoothness_score),
                            "delphi_fitness": float(delphi_metrics.fitness),
                        }
                    )
            except Exception:
                pass

            per_prompt_metrics.append(metrics_dict)

        # give final summary if is_final
        if is_final:
            print(f"\nFinal performance summary for best prompt:")
            best_prompt_id = list(all_abs_errors.keys())[0]
            abs_errors = all_abs_errors[best_prompt_id]
            briers = all_briers[best_prompt_id]
            median_sf_briers = all_median_superforecast_briers[best_prompt_id]
            delphi_logs = all_delphi_logs[best_prompt_id]
            median_public_briers = all_median_public_briers[best_prompt_id]

            if abs_errors:
                abs_avg_error = float(np.mean(abs_errors))
                avg_brier = float(np.mean(briers))
                summary = f"Prompt {best_prompt_id}:\n  Number of predictions: {len(abs_errors)}\n  Average absolute error: {abs_avg_error:.3f}\n  Average Brier score: {avg_brier:.3f}"
                if median_sf_briers:
                    avg_median_sf_brier = float(np.mean(median_sf_briers))
                    summary += f"\n  Average median superforecaster brier: {avg_median_sf_brier:.3f}"
                    median_median_sf_brier = float(np.median(median_sf_briers))
                    summary += f"\n  Median Brier of median superforecaster scores: {median_median_sf_brier:.3f}"
                if median_public_briers:
                    avg_median_public_brier = float(np.mean(median_public_briers))
                    summary += f"\n  Average median public forecaster brier: {avg_median_public_brier:.3f}"
                    median_median_public_brier = float(np.median(median_public_briers))
                    summary += f"\n  Median Brier of median public forecaster scores: {median_median_public_brier:.3f}"
                print(summary)
            else:
                print(f"Prompt {best_prompt_id}: No valid predictions.")

        # Attach traces to population
        if attach_traces:
            try:
                self.optimizer.population.attach_reasoning_traces(all_traces)
                self.optimizer.population.attach_performance_summaries(
                    all_perf_summaries
                )
            except Exception:
                pass
        return fitness_scores, per_prompt_metrics

    async def evaluate_prompt_fitness(
        self,
        prompts: List[str],
        validation_batch: List[Question],
        attach_traces: bool = True,
    ):
        """
        Evaluate fitness of prompts by testing on validation questions.
        This is the core fitness function for the genetic algorithm.
        """
        resolution_date = self.config["data"]["resolution_date"]
        fitness_scores = []
        all_traces: List[str] = []
        all_perf_summaries: List[str] = []

        print(
            f"Evaluating {len(prompts)} prompts on {len(validation_batch)} validation questions..."
        )

        per_prompt_metrics: List[Dict] = []
        for i, prompt in enumerate(prompts):
            print(f"  Evaluating prompt {i + 1}/{len(prompts)}: {prompt[:50]}...")

            total_score = 0.0
            valid_predictions = 0
            abs_errors: List[float] = []
            dir_errors: List[float] = []
            prompt_traces: List[str] = []

            # Test prompt on validation batch
            for question in validation_batch:
                try:
                    # Get prediction with this prompt
                    pred_prob, reasoning = await self.run_expert_with_prompt(
                        question, prompt, resolution_date
                    )

                    # Get ground truth
                    resolution = self.loader.get_resolution(
                        question.id, resolution_date
                    )
                    if resolution and resolution.resolved:
                        actual_outcome = resolution.resolved_to

                        # Calculate Brier score (lower is better, so we negate it for fitness)
                        brier_score = (pred_prob - actual_outcome) ** 2
                        fitness_score = (
                            1.0 - brier_score
                        )  # Convert to fitness (higher is better)

                        total_score += fitness_score
                        valid_predictions += 1
                        abs_errors.append(abs(pred_prob - actual_outcome))
                        dir_errors.append(pred_prob - actual_outcome)
                        prompt_traces.append(
                            f"Question: {question.question}\nPredicted: {pred_prob:.2f}\nActual: {actual_outcome:.2f}\nReasoning:\n{reasoning}"
                        )

                except Exception as e:
                    print(f"    Error evaluating question {question.id}: {e}")
                    continue

            # Calculate average fitness
            if valid_predictions > 0:
                avg_fitness = total_score / valid_predictions
            else:
                avg_fitness = 0.0

            fitness_scores.append(avg_fitness)
            print(
                f"    Fitness: {avg_fitness:.3f} (from {valid_predictions} valid predictions)"
            )
            if abs_errors:
                abs_avg_error = float(np.mean(abs_errors))
                avg_dir_error = float(np.mean(dir_errors))
                batch_summary = f"Number of predictions: {valid_predictions}\nAverage absolute error: {abs_avg_error:.3f}\nAverage directional error: {avg_dir_error:+.3f}"
            else:
                batch_summary = "No valid predictions."
            all_perf_summaries.append(batch_summary)
            all_traces.append("\n\n".join(prompt_traces[-3:]))

            # Per-prompt metrics
            metrics_dict = {
                "num_questions": valid_predictions,
                "mean_abs_error": float(np.mean(abs_errors)) if abs_errors else None,
                "mean_brier": float(np.mean([(de) ** 2 for de in dir_errors]))
                if dir_errors
                else None,
            }
            per_prompt_metrics.append(metrics_dict)

        if attach_traces:
            try:
                self.optimizer.population.attach_reasoning_traces(all_traces)
                self.optimizer.population.attach_performance_summaries(
                    all_perf_summaries
                )
            except Exception:
                pass
        return fitness_scores, per_prompt_metrics

    async def _dry_run_evaluate(
        self, prompts: List[str], validation_batch: Optional[List[Question]] = None
    ):
        """Deterministic, no-LLM fitness for smoke testing pipeline wiring."""
        target = self.target_length if hasattr(self, "target_length") else 100

        def score(p: str) -> float:
            n = len((p or "").split())
            return max(0.0, 1.0 - abs(n - target) / max(target, 1))

        scores = [score(p) for p in prompts]
        return scores, {}

    async def run_expert_with_prompt(
        self, question: Question, prompt: str, resolution_date: str
    ) -> Tuple[float, str]:
        """Run expert with a specific prompt."""

        # Get base question prompt
        base_prompt = load_prompt(
            "expert_forecast",
            "v1",
            question=question.question,
            background=question.background,
            resolution_criteria=question.resolution_criteria,
            url=question.url,
            freeze_datetime_value=question.freeze_datetime_value,
            prior_forecast_info="",
        )

        # Combine evolved prompt with question
        if prompt.strip():
            full_prompt = f"""{prompt}

---

Now apply these strategies to the following question:

{base_prompt}"""
        else:
            full_prompt = base_prompt

        # Get prediction
        self.expert.conversation_manager.messages.clear()
        temperature = self.config["model"].get("expert", {}).get("temperature", 0.3)
        response = await self.expert.conversation_manager.generate_response(
            full_prompt,
            max_tokens=self.config["model"].get("expert", {}).get("max_tokens", 6000),
            temperature=temperature,
        )

        prob = await extract_final_probability_with_retry(
            response, self.expert.conversation_manager, max_retries=3
        )

        return prob, response

    async def evaluate_final_performance(
        self, test_questions: List[Question], split_label: str = "test"
    ):
        """Evaluate the best evolved prompt on the full test set."""
        resolution_date = self.config["data"]["resolution_date"]

        predictions = []
        errors = []

        print(
            f"Testing best prompt on {len(test_questions)} {split_label} questions..."
        )

        if not self.use_delphi_evaluation:
            for i, question in enumerate(test_questions):
                try:
                    # Get prediction with best evolved prompt
                    pred_prob, reasoning = await self.run_expert_with_prompt(
                        question, self.best_prompt, resolution_date
                    )

                    superforecaster_forecasts = self.loader.get_super_forecasts(
                        question_id=question.id, resolution_date=resolution_date
                    )
                    sf_values = [sf.forecast for sf in superforecaster_forecasts]
                    sf_median = np.median(sf_values)

                    public_forecasts = self.loader.get_public_forecasts(
                        question_id=question.id, resolution_date=resolution_date
                    )
                    public_values = [pf.forecast for pf in public_forecasts]
                    public_median = np.median(public_values) if public_values else None

                    # Get ground truth
                    resolution = self.loader.get_resolution(
                        question.id, resolution_date
                    )
                    if resolution and resolution.resolved:
                        actual_outcome = resolution.resolved_to
                        error = abs(pred_prob - actual_outcome)
                        errors.append(error)

                        predictions.append(
                            {
                                "question_id": question.id,
                                "question": question.question,
                                "topic": question.topic,
                                "predicted_prob": pred_prob,
                                "superforecaster_median": sf_median,
                                "public_median": public_median,
                                "actual_outcome": actual_outcome,
                                "error": error,
                                "reasoning": reasoning,
                            }
                        )

                        print(
                            f"  [{i + 1:3d}/{len(test_questions)}] {question.question[:40]}... "
                            f"Pred: {pred_prob:.2f},  SF Median: {sf_median:.2f}, Actual: {actual_outcome:.2f}, Error: {error:.3f}, Brier: {(pred_prob - actual_outcome) ** 2:.3f}, SF Brier: {(sf_median - actual_outcome) ** 2:.3f}"
                        )

                except Exception as e:
                    print(f"  [{i + 1:3d}/{len(test_questions)}] Error: {e}")
                    continue

            if errors:
                mean_error = np.mean(errors)
                std_error = np.std(errors)
                brier_score = np.mean(
                    [
                        (p["predicted_prob"] - p["actual_outcome"]) ** 2
                        for p in predictions
                    ]
                )
                median_brier_score = np.median(
                    [
                        (p["predicted_prob"] - p["actual_outcome"]) ** 2
                        for p in predictions
                    ]
                )
                sf_brier_score = np.mean(
                    [
                        (p["superforecaster_median"] - p["actual_outcome"]) ** 2
                        for p in predictions
                    ]
                )
                median_sf_brier_score = np.median(
                    [
                        (p["superforecaster_median"] - p["actual_outcome"]) ** 2
                        for p in predictions
                    ]
                )
                public_brier_score = np.mean(
                    [
                        (p["public_median"] - p["actual_outcome"]) ** 2
                        for p in predictions
                    ]
                )
                median_public_brier_score = np.median(
                    [
                        (p["public_median"] - p["actual_outcome"]) ** 2
                        for p in predictions
                    ]
                )

                print(f"\nFinal Validation Results:")
                print(f"  Questions evaluated: {len(predictions)}")
                print(f"  Mean absolute error: {mean_error:.3f} ± {std_error:.3f}")
                print(f"  Brier score: {brier_score:.3f}")
                print(f"  Superforecaster median Brier score: {sf_brier_score:.3f}")

                # Store results
                self.evolution_results["final_validation"] = {
                    "mean_absolute_error": mean_error,
                    "std_error": std_error,
                    "brier_score": brier_score,
                    "superforecaster_brier_score": sf_brier_score,
                    "predictions": predictions,
                    "questions_evaluated": len(predictions),
                }
            else:
                print("  No valid predictions made!")

        else:
            # Evaluate using full Delphi process with best prompt
            print(
                f"Evaluating best prompt using full Delphi process ({split_label})..."
            )
            (
                delphi_fitness_scores,
                metrics_list,
            ) = await self.evaluate_prompt_fitness_delphi(
                [self.best_prompt],
                test_questions,
                is_final=True,
                split_label=split_label,
            )
            if delphi_fitness_scores:
                avg_fitness = delphi_fitness_scores[0]
                metrics = metrics_list[0] if metrics_list else {}
                print(f"  Average Delphi fitness on test set: {avg_fitness:.3f}")
                self.evolution_results["final_validation"] = {
                    "delphi_fitness": avg_fitness,
                    "num_questions": len(test_questions),
                    "questions_evaluated": [q.id for q in test_questions],
                    **metrics,
                }
            else:
                print("  No valid Delphi fitness scores obtained!")

    def save_results(self):
        """Save evolution results and best prompt."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use the log_dir (timestamped, and includes dry_run when applicable)
        out_dir = self.log_dir
        suffix = "_DRYRUN" if self.dry_run else ""

        # Defensive: ensure best_prompt is a string
        best_prompt_text = self.best_prompt or ""

        # Save best prompt
        best_prompt_path = out_dir / f"best_evolved_prompt_{timestamp}{suffix}.md"
        with open(best_prompt_path, "w") as f:
            f.write("# Best Evolved Prompt\n\n")
            if self.dry_run:
                f.write("Mode: DRY RUN (no external calls)\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Fitness: {self.evolution_results.get('best_fitness', 0.0):.3f}\n")
            f.write(
                f"Generations: {self.evolution_results.get('total_generations', 0)}\n\n"
            )
            f.write("## Prompt\n\n")
            f.write(best_prompt_text)

        # Save complete results
        results_path = out_dir / f"genetic_evolution_results_{timestamp}{suffix}.json"
        complete_results = {
            "config": self.config,
            "timestamp": timestamp,
            "run_timestamp": getattr(self, "run_timestamp", timestamp),
            "run_dir": str(self.log_dir),
            "run_id": getattr(self, "run_id", None),
            "evolution_results": self.evolution_results,
            "best_prompt": best_prompt_text,
            "dry_run": self.dry_run,
        }
        # Only include heavy stats when not in dry run (optional, but cleaner)
        if not self.dry_run:
            complete_results["superforecaster_stats"] = (
                self.superforecaster_manager.get_statistics()
                if self.superforecaster_manager
                else {}
            )

        with open(results_path, "w") as f:
            json.dump(complete_results, f, indent=2)

        print("\nResults saved:")
        print(f"  Best prompt: {best_prompt_path}")
        print(f"  Complete results: {results_path}")
        # Log directory reflects dry-run routing
        print(f"  Evolution logs: {self.log_dir}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Genetic Prompt Evolution for Delphi Forecasting"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without external LLM/network calls"
    )
    args = parser.parse_args()

    pipeline = GeneticEvolutionPipeline(args.config, dry_run=args.dry_run)
    await pipeline.run_evolution()


if __name__ == "__main__":
    asyncio.run(main())
