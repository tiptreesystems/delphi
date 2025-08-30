"""
Prompt Population Management for Genetic Algorithm

This module manages the population of prompt candidates during evolution.
"""

import asyncio
import random
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import json

from genetic_evolution.operators import PromptCandidate, tournament_selection, crossover, mutate
from utils.prompt_loader import load_prompt


@dataclass
class GenerationStats:
    """Statistics for a generation of prompt evolution."""
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    fitness_std: float
    best_prompt: str
    mutation_rate: float
    # Extended: track train/validation curves separately
    train_best_fitness: float = 0.0
    train_mean_fitness: float = 0.0
    val_best_fitness: float = 0.0
    val_mean_fitness: float = 0.0
    # Optional aggregated metric summaries for this generation
    train_metrics: Optional[Dict[str, float]] = None
    val_metrics: Optional[Dict[str, float]] = None


class PromptPopulation:
    """
    Manages a population of prompt candidates for genetic evolution.

    Implements elitism, tournament selection, crossover, and mutation
    with adaptive mutation rate based on fitness stagnation.
    """

    def __init__(
        self,
        population_size: int = 8,
        elitism_size: int = 2,
        tournament_size: int = 3,
        initial_mutation_rate: float = 0.5,
        mutation_rate_increase: float = 0.1,
        max_mutation_rate: float = 0.9,
        stagnation_threshold: int = 4,
        max_stagnation: int = 8
    ):
        """
        Initialize the prompt population.

        Args:
            population_size: Number of prompts in population (default: 8)
            elitism_size: Number of top prompts to carry forward (default: 2)
            tournament_size: Size of tournament selection (default: 3)
            initial_mutation_rate: Starting mutation rate (default: 0.5)
            mutation_rate_increase: Amount to increase mutation rate (default: 0.1)
            max_mutation_rate: Maximum mutation rate (default: 0.9)
            stagnation_threshold: Generations without improvement to increase mutation (default: 4)
            max_stagnation: Generations without improvement to terminate (default: 8)
        """
        self.population_size = population_size
        self.elitism_size = elitism_size
        self.tournament_size = tournament_size
        self.initial_mutation_rate = initial_mutation_rate
        self.mutation_rate_increase = mutation_rate_increase
        self.max_mutation_rate = max_mutation_rate
        self.stagnation_threshold = stagnation_threshold
        self.max_stagnation = max_stagnation

        # State
        self.population: List[PromptCandidate] = []
        self.generation = 0
        self.mutation_rate = initial_mutation_rate
        self.best_fitness_history: List[float] = []
        self.train_best_fitness_history: List[float] = []
        self.val_best_fitness_history: List[float] = []
        self.generations_without_improvement = 0
        self.generation_stats: List[GenerationStats] = []
        # Pending metrics injected by optimizer before stats capture
        self._pending_train_metrics: Optional[Dict[str, float]] = None
        self._pending_val_metrics: Optional[Dict[str, float]] = None

        # Concurrency control
        self.max_concurrent_mutations = 5  # Default, can be overridden

    def initialize_population(self, seed_prompts: List[str]) -> None:
        """
        Initialize the population with seed prompts.

        Args:
            seed_prompts: List of initial prompt strings
        """
        self.population = []

        # Use provided seed prompts
        for i, prompt in enumerate(seed_prompts[:self.population_size]):
            candidate = PromptCandidate(
                text=prompt,
                fitness=0.0,
                generation=0
            )
            self.population.append(candidate)

        # If we need more prompts, create variations of existing ones
        while len(self.population) < self.population_size:
            base_prompt = random.choice(seed_prompts)
            # Create simple variations
            variations = [
                f"Carefully {base_prompt.lower()}",
                f"{base_prompt} Consider all factors.",
                f"Systematically {base_prompt.lower()}",
                f"{base_prompt} Think step by step.",
                base_prompt  # Include original as fallback
            ]
            variation = random.choice(variations)

            candidate = PromptCandidate(
                text=variation,
                fitness=0.0,
                generation=0
            )
            self.population.append(candidate)

        # Ensure we have exactly the right size
        self.population = self.population[:self.population_size]

    def evaluate_fitness(self, fitness_scores: List[float]) -> None:
        """
        Assign fitness scores to the current population.

        Args:
            fitness_scores: List of fitness scores for each candidate
        """
        if len(fitness_scores) != len(self.population):
            raise ValueError(f"Expected {len(self.population)} fitness scores, got {len(fitness_scores)}")

        for candidate, fitness in zip(self.population, fitness_scores):
            candidate.fitness = fitness
            candidate.val_fitness = fitness

    def attach_aux_fitness(self, train_fitness_scores: List[float], val_fitness_scores: List[float]) -> None:
        """Attach auxiliary train/val fitness scores to candidates without affecting selection."""
        if not self.population:
            return
        for cand, tr, va in zip(self.population, train_fitness_scores, val_fitness_scores):
            cand.train_fitness = tr
            cand.val_fitness = va

    def get_current_prompts(self) -> List[str]:
        """Get the current population prompts as strings."""
        return [candidate.text for candidate in self.population]

    def attach_reasoning_traces(self, traces: List[str]) -> None:
        if not self.population or not traces:
            return
        for candidate, trace in zip(self.population, traces):
            candidate.reasoning_traces = [trace] if trace else None

    def attach_performance_summaries(self, summaries: List[str]) -> None:
        if not self.population or not summaries:
            return
        for candidate, summary in zip(self.population, summaries):
            candidate.performance_summary = summary

    def attach_candidate_metrics(self, train_metrics_list: List[Dict[str, Any]], val_metrics_list: List[Dict[str, Any]]) -> None:
        """Attach per-candidate metric summaries for train/val."""
        if not self.population:
            return
        for cand, trm, vam in zip(self.population, train_metrics_list, val_metrics_list):
            cand.train_metrics = trm
            cand.val_metrics = vam

    def set_pending_generation_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Set aggregated train/val metrics for inclusion in GenerationStats."""
        self._pending_train_metrics = train_metrics
        self._pending_val_metrics = val_metrics

    async def evolve_generation(self, llm=None) -> None:
        """
        Evolve the population by one generation using genetic operators.

        Args:
            llm: Language model for LLM-based mutations (optional)
        """
        if not self.population:
            raise ValueError("Population not initialized")

        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Track fitness progress (selection uses validation fitness stored in candidate.fitness)
        best_val_fitness = self.population[0].fitness
        self.best_fitness_history.append(best_val_fitness)
        # Also track train/val best curves
        train_list = [getattr(c, 'train_fitness', 0.0) for c in self.population]
        val_list = [getattr(c, 'val_fitness', c.fitness) for c in self.population]
        self.train_best_fitness_history.append(max(train_list) if train_list else 0.0)
        self.val_best_fitness_history.append(max(val_list) if val_list else 0.0)

        # Check for improvement
        if (len(self.best_fitness_history) > 1 and
            best_val_fitness <= self.best_fitness_history[-2]):
            self.generations_without_improvement += 1
        else:
            self.generations_without_improvement = 0

        # Adaptive mutation rate
        if (self.generations_without_improvement >= self.stagnation_threshold and
            self.mutation_rate < self.max_mutation_rate):
            self.mutation_rate = min(
                self.max_mutation_rate,
                self.mutation_rate + self.mutation_rate_increase
            )

        # Record generation statistics
        fitnesses = [c.fitness for c in self.population]
        # Compute means for train/val
        train_mean = sum(train_list)/len(train_list) if train_list else 0.0
        val_mean = sum(val_list)/len(val_list) if val_list else 0.0
        stats = GenerationStats(
            generation=self.generation,
            best_fitness=max(fitnesses),
            mean_fitness=sum(fitnesses) / len(fitnesses),
            worst_fitness=min(fitnesses),
            fitness_std=(sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses))**0.5,
            best_prompt=self.population[0].text,
            mutation_rate=self.mutation_rate,
            train_best_fitness=max(train_list) if train_list else 0.0,
            train_mean_fitness=train_mean,
            val_best_fitness=max(val_list) if val_list else 0.0,
            val_mean_fitness=val_mean,
            # Pass through any aggregated metrics (optimizer fills these)
            **({'train_metrics': self._pending_train_metrics} if self._pending_train_metrics else {}),
            **({'val_metrics': self._pending_val_metrics} if self._pending_val_metrics else {})
        )
        self.generation_stats.append(stats)
        # Clear pending metrics after recording
        self._pending_train_metrics = None
        self._pending_val_metrics = None

        # Create next generation
        new_population = []

        # Elitism: copy top performers
        for i in range(self.elitism_size):
            elite = PromptCandidate(
                text=self.population[i].text,
                fitness=0.0,  # Will be re-evaluated
                generation=self.generation + 1,
                parent_ids=(id(self.population[i]),),
                reasoning_traces=self.population[i].reasoning_traces,
                performance_summary=self.population[i].performance_summary
            )
            new_population.append(elite)

        # Fill remaining slots with offspring
        # Determine operations needed for remaining slots
        remaining_slots = self.population_size - len(new_population)
        operations = []

        for _ in range(remaining_slots):
            if random.random() < self.mutation_rate:
                # Mutation operation
                parent = tournament_selection(self.population, self.tournament_size)
                operations.append(('mutate', parent))
            else:
                # Crossover operation
                parent1 = tournament_selection(self.population, self.tournament_size)
                parent2 = tournament_selection(self.population, self.tournament_size)
                operations.append(('crossover', parent1, parent2))

        # Execute operations in parallel
        if llm and operations:
            mutation_count = sum(1 for op in operations if op[0] == 'mutate')
            if mutation_count > 1:
                print(f"  Running {mutation_count} mutations in parallel (max_concurrent: {self.max_concurrent_mutations})")

            start_time = time.time()
            offspring = await self._execute_operations_parallel(operations, llm)

            if mutation_count > 1:
                elapsed = time.time() - start_time
                print(f"  Parallel mutations completed in {elapsed:.2f}s")
        else:
            # Fallback to sequential if no LLM
            offspring = []
            for op in operations:
                if op[0] == 'mutate':
                    offspring.append(await mutate(op[1], llm=None, component_type=getattr(self, 'component_type', 'expert')))
                else:
                    offspring.append(crossover(op[1], op[2]))

        new_population.extend(offspring)

        self.population = new_population
        self.generation += 1

    async def _execute_operations_parallel(self, operations: List, llm) -> List[PromptCandidate]:
        """
        Execute genetic operations (mutations and crossovers) in parallel.

        Args:
            operations: List of operations to execute
            llm: Language model for mutations

        Returns:
            List of offspring candidates
        """
        # Separate mutations and crossovers
        mutation_tasks = []
        crossover_results = []

        for op in operations:
            if op[0] == 'mutate':
                # Create mutation task
                parent = op[1]
                task = mutate(parent, llm=llm, component_type=getattr(self, 'component_type', 'expert'))
                mutation_tasks.append(task)
            else:
                # Execute crossover immediately (no LLM needed)
                parent1, parent2 = op[1], op[2]
                offspring = crossover(parent1, parent2)
                crossover_results.append(offspring)

        # Execute all mutations in parallel
        if mutation_tasks:
            try:
                # Use asyncio.gather with concurrency limit to avoid overwhelming the API
                max_concurrent = self.max_concurrent_mutations

                if len(mutation_tasks) <= max_concurrent:
                    # Run all tasks at once if within limit
                    mutation_results = await asyncio.gather(*mutation_tasks, return_exceptions=True)
                else:
                    # Run in batches if too many tasks
                    mutation_results = []
                    for i in range(0, len(mutation_tasks), max_concurrent):
                        batch = mutation_tasks[i:i + max_concurrent]
                        batch_results = await asyncio.gather(*batch, return_exceptions=True)
                        mutation_results.extend(batch_results)

                # Handle exceptions and failed mutations
                successful_mutations = []
                for result in mutation_results:
                    if isinstance(result, Exception):
                        print(f"Mutation failed: {result}")
                        # Use a fallback mutation or skip
                        continue
                    else:
                        successful_mutations.append(result)

            except Exception as e:
                print(f"Parallel mutation failed: {e}")
                # Fallback to sequential mutations
                successful_mutations = []
                for task in mutation_tasks:
                    try:
                        result = await task
                        successful_mutations.append(result)
                    except Exception as task_e:
                        print(f"Sequential fallback mutation failed: {task_e}")
                        continue
        else:
            successful_mutations = []

        # Combine all results
        all_offspring = crossover_results + successful_mutations

        return all_offspring

    def should_terminate(self, max_generations: int = 40) -> bool:
        """
        Check if evolution should terminate.

        Args:
            max_generations: Maximum number of generations

        Returns:
            True if evolution should terminate
        """
        return (self.generation >= max_generations or
                self.generations_without_improvement >= self.max_stagnation)

    def get_best_candidate(self) -> Optional[PromptCandidate]:
        """Get the best candidate from the current population."""
        if not self.population:
            return None
        return max(self.population, key=lambda x: x.fitness)

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of the evolution process."""
        best = self.get_best_candidate()
        return {
            'total_generations': self.generation,
            'final_mutation_rate': self.mutation_rate,
            'generations_without_improvement': self.generations_without_improvement,
            'best_fitness': best.fitness if best else 0.0,
            'best_prompt': best.text if best else "",
            'fitness_history': self.best_fitness_history,
            'train_best_fitness_history': self.train_best_fitness_history,
            'val_best_fitness_history': self.val_best_fitness_history,
            'generation_stats': [asdict(stats) for stats in self.generation_stats]
        }
