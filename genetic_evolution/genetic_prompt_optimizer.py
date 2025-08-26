"""
Genetic Algorithm Prompt Optimizer

Main class for optimizing prompts using genetic algorithms with fitness evaluation
and optional length penalties.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json
from pathlib import Path

from utils.models import BaseLLM
from utils.superforecaster_utils import SuperforecasterManager
from genetic_evolution.prompt_population import PromptPopulation, PromptCandidate
from genetic_evolution.operators import PromptCandidate


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation."""
    length_penalty_weight: float = 0.0  # Weight for length penalty (0 = no penalty)
    target_length: Optional[int] = None  # Target prompt length in tokens
    superforecaster_manager: Optional[SuperforecasterManager] = None  # Superforecaster manager
    include_reasoning: bool = True  # Include superforecaster reasoning
    include_examples: bool = True  # Include ICL examples
    n_examples: int = 3  # Number of examples to include


class GeneticPromptOptimizer:
    """
    Genetic algorithm optimizer for evolving prompts to maximize forecasting performance.
    
    Features:
    - Population-based evolution with elitism
    - Tournament selection, crossover, and mutation
    - Adaptive mutation rate based on fitness stagnation
    - Optional length penalties for prompt optimization
    - Support for superforecaster reasoning and ICL examples
    - Comprehensive logging and statistics tracking
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        population_config: Optional[Dict[str, Any]] = None,
        fitness_config: Optional[FitnessConfig] = None,
        log_dir: Optional[str] = None,
        max_concurrent_mutations: int = 5,
        component_type: str = 'expert'
    ):
        """
        Initialize the genetic prompt optimizer.
        
        Args:
            llm: Language model for evaluation
            population_config: Configuration for population management
            fitness_config: Configuration for fitness evaluation
            log_dir: Directory for logging results
            max_concurrent_mutations: Maximum concurrent mutations per generation
            component_type: Type of component being optimized ('expert' or 'mediator')
        """
        self.llm = llm
        self.fitness_config = fitness_config or FitnessConfig()
        self.max_concurrent_mutations = max_concurrent_mutations
        self.component_type = component_type
        
        # Initialize population with default config
        pop_config = population_config or {}
        self.population = PromptPopulation(**pop_config)
        # Pass concurrency setting and component type to population
        self.population.max_concurrent_mutations = max_concurrent_mutations
        self.population.component_type = component_type
        
        # Logging setup
        self.log_dir = Path(log_dir) if log_dir else Path("genetic_evolution_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"genetic_optimizer_{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / f"evolution_{self.population.generation}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Evaluation function (to be set by user)
        self.evaluation_function: Optional[Callable] = None
        
    def set_evaluation_function(self, eval_func: Callable[[List[str]], List[float]]) -> None:
        """
        Set the function used to evaluate prompt fitness.
        
        Args:
            eval_func: Function that takes list of prompts and returns list of fitness scores
        """
        self.evaluation_function = eval_func
        
    def calculate_fitness_with_penalty(self, base_fitness: float, prompt: str) -> float:
        """
        Calculate fitness with optional length penalty.
        
        Args:
            base_fitness: Base fitness score from task performance
            prompt: Prompt text for length calculation
            
        Returns:
            Final fitness score with length penalty applied
        """
        if self.fitness_config.length_penalty_weight == 0.0:
            return base_fitness
            
        # Simple token count (split on whitespace)
        prompt_length = len(prompt.split())
        
        if self.fitness_config.target_length is None:
            # Penalty for overly long prompts (generic penalty)
            length_penalty = max(0, (prompt_length - 100) * 0.01)  # Penalty after 100 tokens
        else:
            # Penalty based on deviation from target length
            target = self.fitness_config.target_length
            length_penalty = abs(prompt_length - target) * 0.01
            
        final_fitness = base_fitness - (self.fitness_config.length_penalty_weight * length_penalty)
        return final_fitness
        
    def prepare_prompts_for_evaluation(self, prompts: List[str]) -> List[str]:
        """
        Prepare prompts for evaluation by adding superforecaster context if configured.
        
        Args:
            prompts: Raw prompt texts
            
        Returns:
            Enhanced prompts with superforecaster context (only for expert prompts)
        """
        # Don't enhance mediator prompts with superforecaster context
        if self.component_type == 'mediator':
            return prompts
            
        if not self.fitness_config.superforecaster_manager:
            return prompts
        
        enhanced_prompts = []
        manager = self.fitness_config.superforecaster_manager
        
        for prompt in prompts:
            enhanced_prompt = manager.enhance_prompt(
                prompt,
                include_reasoning=self.fitness_config.include_reasoning,
                include_examples=self.fitness_config.include_examples,
                n_examples=self.fitness_config.n_examples
            )
            enhanced_prompts.append(enhanced_prompt)
            
        return enhanced_prompts
        
    async def evolve(
        self,
        seed_prompts: List[str],
        validation_batch: Any,
        max_generations: int = 40,
        save_every: int = 5
    ) -> Dict[str, Any]:
        """
        Run the genetic algorithm to evolve prompts.
        
        Args:
            seed_prompts: Initial prompt population
            validation_batch: Validation data for fitness evaluation
            max_generations: Maximum number of generations to run
            save_every: Save progress every N generations
            
        Returns:
            Dictionary with evolution results and statistics
        """
        if not self.evaluation_function:
            raise ValueError("Evaluation function not set. Use set_evaluation_function() first.")
            
        self.logger.info(f"Starting genetic evolution with {len(seed_prompts)} seed prompts")
        self.logger.info(f"Population size: {self.population.population_size}")
        self.logger.info(f"Max generations: {max_generations}")
        
        # Initialize population
        self.population.initialize_population(seed_prompts)
        # Evolution loop
        
        while not self.population.should_terminate(max_generations):
            generation_start_time = asyncio.get_event_loop().time()
            
            self.logger.info(f"Generation {self.population.generation}")
            
            # Get current prompts
            current_prompts = self.population.get_current_prompts()
            
            # Prepare prompts with superforecaster context
            enhanced_prompts = self.prepare_prompts_for_evaluation(current_prompts)
            
            # Evaluate fitness
            base_fitness_scores = await self.evaluation_function(enhanced_prompts, validation_batch)
       
            # Apply length penalties
            final_fitness_scores = []
            for base_fitness, prompt in zip(base_fitness_scores, current_prompts):
                final_fitness = self.calculate_fitness_with_penalty(base_fitness, prompt)
                final_fitness_scores.append(final_fitness)
            
            # Update population fitness
            self.population.evaluate_fitness(final_fitness_scores)
            
            # Log generation statistics
            best_candidate = self.population.get_best_candidate()
            if best_candidate:
                self.logger.info(f"Best fitness: {best_candidate.fitness:.4f}")
                self.logger.info(f"Best prompt: {best_candidate.text[:100]}...")
                
            self.logger.info(f"Mutation rate: {self.population.mutation_rate:.3f}")
            self.logger.info(f"Generations without improvement: {self.population.generations_without_improvement}")
            
            # Save progress periodically
            if self.population.generation % save_every == 0:
                await self.save_progress()
                
            # Evolve to next generation
            await self.population.evolve_generation(self.llm)
            
            generation_time = asyncio.get_event_loop().time() - generation_start_time
            self.logger.info(f"Generation {self.population.generation-1} completed in {generation_time:.2f}s")
            
        # Final logging
        self.logger.info("Evolution completed!")
        final_summary = self.population.get_evolution_summary()
        best_candidate = self.population.get_best_candidate()
        actual_best_fitness = best_candidate.fitness if best_candidate else final_summary['best_fitness']
        
        self.logger.info(f"Total generations: {final_summary['total_generations']}")
        self.logger.info(f"Best final fitness: {actual_best_fitness:.4f}")
        
        # Save final results
        await self.save_final_results()
        
        return final_summary
        
    async def save_progress(self) -> None:
        """Save current evolution progress to files."""
        progress_file = self.log_dir / f"progress_gen_{self.population.generation}.json"
        
        summary = self.population.get_evolution_summary()
        summary['current_population'] = [
            {
                'text': candidate.text,
                'fitness': candidate.fitness,
                'generation': candidate.generation
            }
            for candidate in self.population.population
        ]
        
        with open(progress_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Progress saved to {progress_file}")
        
    async def save_final_results(self) -> None:
        """Save final evolution results."""
        results_file = self.log_dir / "final_results.json"
        
        summary = self.population.get_evolution_summary()
        summary['final_population'] = [
            {
                'text': candidate.text,
                'fitness': candidate.fitness,
                'generation': candidate.generation,
                'parent_ids': candidate.parent_ids
            }
            for candidate in self.population.population
        ]
        
        # Add configuration info
        summary['config'] = {
            'population_size': self.population.population_size,
            'elitism_size': self.population.elitism_size,
            'tournament_size': self.population.tournament_size,
            'initial_mutation_rate': self.population.initial_mutation_rate,
            'fitness_config': {
                'length_penalty_weight': self.fitness_config.length_penalty_weight,
                'target_length': self.fitness_config.target_length,
                'has_superforecaster_manager': self.fitness_config.superforecaster_manager is not None,
                'include_reasoning': self.fitness_config.include_reasoning,
                'include_examples': self.fitness_config.include_examples,
                'n_examples': self.fitness_config.n_examples
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Final results saved to {results_file}")
        
    def get_best_prompt(self) -> Optional[str]:
        """Get the best prompt from the final population."""
        best_candidate = self.population.get_best_candidate()
        return best_candidate.text if best_candidate else None