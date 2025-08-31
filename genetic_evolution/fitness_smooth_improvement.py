"""
Fitness function for optimizing smooth improvement across Delphi rounds.

This module implements a fitness function that:
1. Optimizes for median expert improvement in Brier score across rounds
2. Encourages smooth, consistent improvement (low variance)
3. Penalizes erratic behavior across rounds
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RoundMetrics:
    """Metrics for a single Delphi round."""
    round_num: int
    expert_briers: List[float]  # Individual expert Brier scores
    median_brier: float
    mean_brier: float
    std_brier: float
    improvement_from_prev: Optional[float] = None  # Improvement from previous round


@dataclass
class DelphiMetrics:
    """Aggregated metrics across all Delphi rounds."""
    rounds: List[RoundMetrics]
    median_improvements: List[float]  # Median improvement at each round
    improvement_variance: float  # Variance of improvements
    total_improvement: float  # Total improvement from first to last
    smoothness_score: float  # How smooth the improvement curve is
    fitness: float  # Final fitness score


def calculate_round_metrics(predictions: List[float], actual_outcome: float, round_num: int, 
                           prev_median: Optional[float] = None) -> RoundMetrics:
    """
    Calculate metrics for a single round.
    
    Args:
        predictions: List of expert predictions
        actual_outcome: Ground truth (0 or 1)
        round_num: Round number
        prev_median: Median Brier from previous round (for improvement calculation)
    
    Returns:
        RoundMetrics object with calculated values
    """
    # Calculate Brier scores for each expert
    brier_scores = [(pred - actual_outcome) ** 2 for pred in predictions]
    
    # Calculate statistics
    median_brier = np.median(brier_scores)
    mean_brier = np.mean(brier_scores)
    std_brier = np.std(brier_scores)
    
    # Calculate improvement from previous round
    improvement = None
    if prev_median is not None:
        # Improvement is positive when Brier score decreases (gets better)
        improvement = prev_median - median_brier
    
    return RoundMetrics(
        round_num=round_num,
        expert_briers=brier_scores,
        median_brier=median_brier,
        mean_brier=mean_brier,
        std_brier=std_brier,
        improvement_from_prev=improvement
    )


def calculate_smoothness(improvements: List[float]) -> float:
    """
    Calculate smoothness score for improvement trajectory.
    
    A smooth improvement has:
    - Consistent positive improvements
    - Low variance between consecutive improvements
    - No large jumps or reversals
    
    Args:
        improvements: List of round-to-round improvements
    
    Returns:
        Smoothness score (0-1, higher is smoother)
    """
    if len(improvements) < 2:
        return 1.0  # Perfect smoothness for single improvement
    
    # Calculate consecutive differences
    diffs = np.diff(improvements)
    
    # Penalize variance in improvement rates
    variance_penalty = np.var(diffs)
    
    # Penalize negative improvements (getting worse)
    negative_penalty = sum(1 for imp in improvements if imp < 0) / len(improvements)
    
    # Penalize large jumps
    max_jump = np.max(np.abs(diffs)) if len(diffs) > 0 else 0
    jump_penalty = max_jump / (np.mean(np.abs(improvements)) + 1e-6)
    
    # Combine penalties into smoothness score
    smoothness = 1.0 / (1.0 + variance_penalty + negative_penalty + jump_penalty)
    
    return smoothness


def calculate_delphi_fitness(delphi_results: List[Dict], actual_outcomes: Dict[str, float],
                            variance_weight: float = 0.3,
                            smoothness_weight: float = 0.2,
                            improvement_weight: float = 0.5) -> DelphiMetrics:
    """
    Calculate fitness for Delphi results focusing on smooth improvement.
    
    Args:
        delphi_results: List of Delphi result dictionaries (one per question)
        actual_outcomes: Dictionary mapping question_id to actual outcome (0 or 1)
        variance_weight: Weight for variance penalty (0-1)
        smoothness_weight: Weight for smoothness reward (0-1)
        improvement_weight: Weight for total improvement reward (0-1)
    
    Returns:
        DelphiMetrics object with calculated fitness
    """
    all_round_metrics = []
    all_improvements_by_round = {1: [], 2: [], 3: []}  # Collect improvements by round number
    
    for result in delphi_results:
        question_id = result.get('question_id')
        if question_id not in actual_outcomes:
            continue
            
        actual = actual_outcomes[question_id]
        rounds = result.get('rounds', [])
        
        if not rounds:
            continue
        
        question_metrics = []
        prev_median = None
        
        for round_data in rounds:
            round_num = round_data['round']
            experts = round_data.get('experts', {})
            
            if not experts:
                continue
            
            # Extract predictions
            predictions = [expert['prob'] for expert in experts.values()]
            
            # Calculate metrics for this round
            metrics = calculate_round_metrics(predictions, actual, round_num, prev_median)
            question_metrics.append(metrics)
            
            # Track improvement by round number
            if metrics.improvement_from_prev is not None:
                all_improvements_by_round[round_num].append(metrics.improvement_from_prev)
            
            prev_median = metrics.median_brier
        
        all_round_metrics.extend(question_metrics)
    
    if not all_round_metrics:
        return DelphiMetrics(
            rounds=[],
            median_improvements=[],
            improvement_variance=0,
            total_improvement=0,
            smoothness_score=0,
            fitness=0
        )
    
    # Calculate median improvements across questions for each round
    median_improvements = []
    for round_num in sorted(all_improvements_by_round.keys()):
        improvements = all_improvements_by_round[round_num]
        if improvements:
            median_improvements.append(np.median(improvements))
    
    # Calculate overall metrics
    improvement_variance = np.var(median_improvements) if median_improvements else 0
    total_improvement = sum(median_improvements) if median_improvements else 0
    smoothness_score = calculate_smoothness(median_improvements) if median_improvements else 0
    
    # Calculate fitness score
    # Higher fitness for:
    # - More total improvement (lower final Brier)
    # - Lower variance (more consistent improvement)
    # - Higher smoothness (steady progression)
    
    # Normalize components
    improvement_component = 1.0 / (1.0 + np.exp(-total_improvement * 10))  # Sigmoid to 0-1
    variance_component = 1.0 / (1.0 + improvement_variance * 10)  # Lower variance is better
    smoothness_component = smoothness_score  # Already 0-1
    
    # Weighted combination
    fitness = (improvement_weight * improvement_component +
              variance_weight * variance_component +
              smoothness_weight * smoothness_component)
    
    return DelphiMetrics(
        rounds=all_round_metrics,
        median_improvements=median_improvements,
        improvement_variance=improvement_variance,
        total_improvement=total_improvement,
        smoothness_score=smoothness_score,
        fitness=fitness
    )


def evaluate_prompt_smooth_improvement(delphi_log: Dict, actual_outcome: float) -> Tuple[float, Dict]:
    """
    Evaluate a single Delphi result for smooth improvement.
    
    Args:
        delphi_log: Delphi result dictionary for a single question
        actual_outcome: Ground truth (0 or 1)
    
    Returns:
        Tuple of (fitness_score, detailed_metrics)
    """
    rounds = delphi_log.get('rounds', [])
    
    if not rounds:
        return 0.0, {'error': 'No rounds found'}
    
    round_metrics = []
    improvements = []
    prev_median = None
    
    for round_data in rounds:
        round_num = round_data['round']
        experts = round_data.get('experts', {})
        
        if not experts:
            continue
        
        predictions = [expert['prob'] for expert in experts.values()]
        metrics = calculate_round_metrics(predictions, actual_outcome, round_num, prev_median)
        round_metrics.append(metrics)
        
        if metrics.improvement_from_prev is not None:
            improvements.append(metrics.improvement_from_prev)
        
        prev_median = metrics.median_brier
    
    # Calculate fitness components
    total_improvement = sum(improvements) if improvements else 0
    improvement_variance = np.var(improvements) if improvements else 0
    smoothness = calculate_smoothness(improvements) if improvements else 0
    
    # Calculate final fitness
    fitness = (0.5 * (1.0 / (1.0 + np.exp(-total_improvement * 10))) +  # Improvement
              0.3 * (1.0 / (1.0 + improvement_variance * 10)) +  # Low variance
              0.2 * smoothness)  # Smoothness
    
    detailed_metrics = {
        'total_improvement': total_improvement,
        'improvement_variance': improvement_variance,
        'smoothness': smoothness,
        'improvements_by_round': improvements,
        'median_briers_by_round': [m.median_brier for m in round_metrics],
        'fitness': fitness
    }
    
    return fitness, detailed_metrics


def evaluate_prompt_smooth_improvement_v2(delphi_log: Dict, actual_outcome: float) -> Tuple[float, Dict]:
    """
    Calibrated evaluator with wider dynamic range and richer diagnostics.

    Improvements over the baseline evaluator:
    - Uses a Brier skill score vs a naive 0.5 baseline to reward absolute
      forecast quality (not just improvement).
    - Rewards earlier improvements via normalized area-under-improvement.
    - Includes monotonicity and smoothness for consistency.
    - Returns a richer metrics dict for analysis and dashboards.

    Args:
        delphi_log: Delphi result dictionary for a single question
        actual_outcome: Ground truth (0 or 1)

    Returns:
        Tuple of (fitness_score in [0,1], detailed_metrics)
    """
    rounds = delphi_log.get('rounds', [])
    if not rounds:
        return 0.0, {'error': 'No rounds found'}

    round_metrics: List[RoundMetrics] = []
    improvements: List[float] = []
    median_briers: List[float] = []
    prev_median = None

    for round_data in rounds:
        experts = round_data.get('experts', {})
        if not experts:
            continue
        predictions = [expert['prob'] for expert in experts.values()]
        metrics = calculate_round_metrics(predictions, actual_outcome, round_data['round'], prev_median)
        round_metrics.append(metrics)
        median_briers.append(metrics.median_brier)
        if metrics.improvement_from_prev is not None:
            improvements.append(metrics.improvement_from_prev)
        prev_median = metrics.median_brier

    if not median_briers:
        return 0.0, {'error': 'No usable rounds'}

    # Core quantities
    initial_brier = float(median_briers[0])
    final_brier = float(median_briers[-1])
    total_improvement = float(max(0.0, initial_brier - final_brier))
    rel_improvement = float(total_improvement / (initial_brier + 1e-6))  # 0..1 typically
    rel_improvement = float(np.clip(rel_improvement, 0.0, 1.0))

    # Absolute quality via Brier skill score vs naive 0.5 forecast (baseline brier=0.25)
    baseline_brier = 0.25
    skill = (baseline_brier - final_brier) / baseline_brier  # 1.0 best, 0.0 baseline, <0 worse than baseline
    skill_component = float(np.clip(skill, 0.0, 1.0))

    # Area-under-improvement: reward earlier gains (normalize by max possible area)
    # Area = sum over rounds of (initial_brier - brier_r), r>=2
    if len(median_briers) > 1:
        area = sum(max(0.0, initial_brier - b) for b in median_briers[1:])
        max_area = initial_brier * (len(median_briers) - 1)
        area_norm = float(area / (max_area + 1e-6))
        area_norm = float(np.clip(area_norm, 0.0, 1.0))
    else:
        area_norm = 0.0

    # Consistency and smoothness
    if len(improvements) > 0:
        monotonicity = float(sum(1 for imp in improvements if imp >= 0) / len(improvements))
        smoothness = float(calculate_smoothness(improvements))
        improvement_variance = float(np.var(improvements))
    else:
        monotonicity = 1.0
        smoothness = 1.0
        improvement_variance = 0.0

    # Combine components with weights chosen to widen dynamic range
    # - skill_component captures absolute forecast quality
    # - rel_improvement + area_norm capture progress and timing
    # - monotonicity + smoothness reward consistency
    fitness_raw = (
        0.35 * skill_component +
        0.25 * rel_improvement +
        0.20 * area_norm +
        0.10 * monotonicity +
        0.10 * smoothness
    )
    fitness = float(np.clip(fitness_raw, 0.0, 1.0))

    detailed_metrics = {
        'initial_median_brier': initial_brier,
        'final_median_brier': final_brier,
        'total_improvement': total_improvement,
        'relative_improvement': rel_improvement,
        'area_under_improvement': area_norm,
        'improvement_variance': improvement_variance,
        'monotonicity': monotonicity,
        'smoothness': smoothness,
        'skill_component': skill_component,
        'median_briers_by_round': median_briers,
        'improvements_by_round': [float(i) for i in improvements],
        'fitness': fitness,
    }

    return fitness, detailed_metrics
