"""
Genetic Algorithm Operators for Prompt Evolution

This module implements crossover, mutation, and selection operators for evolving prompts.
"""

import random
import re
from typing import Optional, List, Tuple
from dataclasses import dataclass
from utils.models import BaseLLM
from utils.prompt_loader import load_prompt


@dataclass
class PromptCandidate:
    """A candidate prompt with its fitness score."""
    text: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: Tuple[int, ...] = ()
    reasoning_traces: Optional[List[str]] = None
    performance_summary: Optional[str] = None
    # Auxiliary fitness tracking for train/validation
    train_fitness: float = 0.0
    val_fitness: float = 0.0


def tokenize_prompt(prompt: str) -> List[str]:
    """Simple tokenization that preserves meaningful units."""
    # Split on whitespace but preserve punctuation as separate tokens
    tokens = []
    for word in prompt.split():
        # Split punctuation from words
        word_tokens = re.findall(r'\w+|[^\w\s]', word)
        tokens.extend(word_tokens)
    return [token for token in tokens if token.strip()]


def detokenize_prompt(tokens: List[str]) -> str:
    """Reconstruct prompt from tokens."""
    result = []
    for i, token in enumerate(tokens):
        if i > 0 and token not in '.,!?;:' and tokens[i-1] not in '([{':
            result.append(' ')
        result.append(token)
    return ''.join(result)


def single_point_crossover(parent1: PromptCandidate, parent2: PromptCandidate) -> PromptCandidate:
    """
    Perform single-point crossover between two parent prompts.
    
    Args:
        parent1: First parent prompt candidate
        parent2: Second parent prompt candidate
        
    Returns:
        New child prompt candidate
    """
    tokens1 = tokenize_prompt(parent1.text)
    tokens2 = tokenize_prompt(parent2.text)
    
    if not tokens1 or not tokens2:
        # If either parent is empty, return the non-empty one
        return parent1 if tokens1 else parent2
    
    # Choose crossover point
    min_len = min(len(tokens1), len(tokens2))
    if min_len <= 1:
        # For very short prompts, just randomly choose one parent
        return random.choice([parent1, parent2])
    
    crossover_point = random.randint(1, min_len - 1)
    
    # Create child by combining parts
    child_tokens = tokens1[:crossover_point] + tokens2[crossover_point:]
    
    child = PromptCandidate(
        text=detokenize_prompt(child_tokens),
        fitness=0.0,
        generation=max(parent1.generation, parent2.generation) + 1,
        parent_ids=(id(parent1), id(parent2))
    )
    
    return child


async def mutate(candidate: PromptCandidate, llm: BaseLLM, component_type: str = 'expert') -> PromptCandidate:
    """
    Mutate a prompt candidate using LLM-based mutations.
    
    Args:
        candidate: The prompt candidate to mutate
        llm: Language model to use for mutations
        component_type: Type of component being optimized ('expert' or 'mediator')
        
    Returns:
        Mutated prompt candidate
    """
    # Select appropriate mutation prompt based on component type
    if component_type == 'mediator':
        prompt_prefix = 'mutation_mediator'
    else:
        prompt_prefix = 'mutation'
    
    if candidate.reasoning_traces or candidate.performance_summary:
        traces_text = "\n\n".join((candidate.reasoning_traces or [])[:3])
        perf_text = candidate.performance_summary or ""
        mutation_prompt = load_prompt(
            'genetic_evolution',
            f'{prompt_prefix}_with_context',
            current_prompt=candidate.text,
            performance_summary=perf_text,
            reasoning_traces=traces_text
        )
    else:
        mutation_type = random.choice(['rephrase', 'add_detail', 'simplify', 'change_focus'])
        mutation_prompt = load_prompt(
            'genetic_evolution',
            f'{prompt_prefix}_basic',
            current_prompt=candidate.text,
            mutation_type=mutation_type
        )
    
    # Use the LLM to generate the mutation
    response = await llm.generate(
        mutation_prompt,
        max_tokens=6000,
        temperature=0.2
    )
    print(response)

    # Extract the mutated prompt (simple extraction)
    mutated_text = response.strip()
    
    mutated = PromptCandidate(
        text=mutated_text,
        fitness=0.0,
        generation=candidate.generation + 1,
        parent_ids=(id(candidate),),
        reasoning_traces=candidate.reasoning_traces,
        performance_summary=candidate.performance_summary
    )
    
    return mutated


def tournament_selection(population: List[PromptCandidate], tournament_size: int = 3) -> PromptCandidate:
    """
    Select a candidate using tournament selection.
    
    Args:
        population: List of prompt candidates
        tournament_size: Number of candidates to compete in tournament
        
    Returns:
        Selected prompt candidate
    """
    if len(population) < tournament_size:
        tournament_size = len(population)
    
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)


def crossover(parent1: PromptCandidate, parent2: PromptCandidate) -> PromptCandidate:
    """Alias for single_point_crossover for consistency."""
    return single_point_crossover(parent1, parent2)
