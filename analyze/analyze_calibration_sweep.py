#!/usr/bin/env python3
"""
Analyze Expected Calibration Error (ECE) across sweep results.

ECE measures how well calibrated probability forecasts are. 
For perfect calibration, when forecasts predict X% probability, 
the event should occur X% of the time.
"""

import json
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple

def calculate_ece(probabilities: List[float], outcomes: List[int], n_bins: int = 10) -> Tuple[float, Dict]:
    """
    Calculate Expected Calibration Error.
    
    Args:
        probabilities: List of predicted probabilities [0, 1]
        outcomes: List of binary outcomes (0 or 1)
        n_bins: Number of bins for calibration
    
    Returns:
        (ece_value, bin_stats) where bin_stats contains detailed info per bin
    """
    if len(probabilities) != len(outcomes):
        raise ValueError("Probabilities and outcomes must have same length")
    
    if len(probabilities) == 0:
        return 0.0, {}
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_stats = {}
    
    ece = 0.0
    total_samples = len(probabilities)
    
    for i in range(n_bins):
        # Get samples in this bin
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:  # Last bin includes 1.0
            in_bin = [(p, o) for p, o in zip(probabilities, outcomes) 
                      if bin_lower <= p <= bin_upper]
        else:
            in_bin = [(p, o) for p, o in zip(probabilities, outcomes) 
                      if bin_lower <= p < bin_upper]
        
        if len(in_bin) > 0:
            bin_probs = [p for p, _ in in_bin]
            bin_outcomes = [o for _, o in in_bin]
            
            # Average predicted probability in bin
            avg_pred_prob = np.mean(bin_probs)
            # Actual frequency in bin  
            actual_freq = np.mean(bin_outcomes)
            # Number of samples in bin
            bin_count = len(in_bin)
            
            # Calibration error for this bin
            calibration_error = abs(avg_pred_prob - actual_freq)
            
            # Weight by proportion of samples
            bin_weight = bin_count / total_samples
            ece += bin_weight * calibration_error
            
            bin_stats[f"bin_{i}"] = {
                "range": (bin_lower, bin_upper),
                "avg_pred_prob": avg_pred_prob,
                "actual_freq": actual_freq,
                "count": bin_count,
                "calibration_error": calibration_error,
                "weight": bin_weight
            }
    
    return ece, bin_stats

def extract_forecasts_from_json(json_path: Path) -> Dict:
    """Extract expert forecasts and outcomes from a Delphi JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = {
        'question_id': data.get('question_id', 'unknown'),
        'resolution': data.get('resolution'),
        'rounds': {}
    }
    
    # Get outcome (assuming binary - resolved to 1 or 0)
    if results['resolution'] and hasattr(results['resolution'], 'resolved_to'):
        outcome = results['resolution'].resolved_to
    else:
        # Try to infer from question outcome if available
        outcome = None
        if 'outcome' in data:
            outcome = data['outcome']
    
    results['outcome'] = outcome
    
    # Extract forecasts by round
    for round_data in data.get('rounds', []):
        round_num = round_data.get('round', 0)
        experts_data = round_data.get('experts', {})
        
        expert_probs = {}
        for expert_id, expert_response in experts_data.items():
            if 'prob' in expert_response:
                expert_probs[expert_id] = expert_response['prob']
        
        results['rounds'][round_num] = expert_probs
    
    return results

def load_pickle_for_resolution(pickle_path: Path) -> Tuple[float, any]:
    """Load pickle file to get resolution data."""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        # Extract outcome from pickle structure
        if isinstance(data, dict):
            # Look for resolution in various places
            if 'resolution' in data:
                resolution = data['resolution']
                if hasattr(resolution, 'resolved_to'):
                    return resolution.resolved_to
                elif isinstance(resolution, dict) and 'resolved_to' in resolution:
                    return resolution['resolved_to']
            
            # Try outcome field
            if 'outcome' in data:
                return data['outcome']
            
            # Try in question data
            if 'question' in data:
                question = data['question']
                if hasattr(question, 'resolution'):
                    if hasattr(question.resolution, 'resolved_to'):
                        return question.resolution.resolved_to
                elif isinstance(question, dict) and 'resolution' in question:
                    return question['resolution'].get('resolved_to')
    except Exception as e:
        pass
    
    return None

def load_resolution_data(resolution_file: Path = None) -> Dict[str, int]:
    """Load resolution data from JSON file."""
    if resolution_file is None:
        # Default location
        resolution_file = Path("../data/2024-07-21_resolution_set.json")
    
    if not resolution_file.exists():
        print(f"Warning: Resolution file not found at {resolution_file}")
        return {}
    
    print(f"Loading resolutions from {resolution_file}")
    
    with open(resolution_file, 'r') as f:
        resolution_data = json.load(f)
    
    # Extract outcomes for each question
    outcomes = {}
    
    # Handle the actual format: {"resolutions": [{"id": ..., "resolved_to": ...}]}
    if isinstance(resolution_data, dict) and 'resolutions' in resolution_data:
        for res_entry in resolution_data['resolutions']:
            if isinstance(res_entry, dict) and 'id' in res_entry and 'resolved_to' in res_entry:
                question_id = res_entry['id']
                resolved_to = res_entry['resolved_to']
                # Convert to binary (0 or 1)
                if resolved_to in [0, 0.0, 1, 1.0]:
                    outcomes[str(question_id)] = int(resolved_to)
    
    print(f"Loaded {len(outcomes)} question resolutions")
    return outcomes

def analyze_calibration_for_sweep(sweep_dir: Path) -> Dict:
    """Analyze calibration for all experiments in a sweep."""
    
    # Load resolution data
    resolution_outcomes = load_resolution_data()
    
    results = {
        'by_n_experts': defaultdict(lambda: defaultdict(lambda: {'probs': [], 'outcomes': []})),
        'by_round': defaultdict(lambda: {'probs': [], 'outcomes': []}),
        'by_config': {}
    }
    
    # Find all output directories
    output_dirs = [d for d in sweep_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('results_n_experts_')]
    
    print(f"Found {len(output_dirs)} output directories")
    
    questions_with_outcomes = 0
    questions_without_outcomes = 0
    
    for output_dir in output_dirs:
        # Parse n_experts and seed from directory name
        parts = output_dir.name.replace('results_n_experts_', '').split('_')
        if len(parts) >= 3 and parts[1] == 'seed':
            n_experts = int(parts[0])
            seed = int(parts[2])
        else:
            continue
        
        config_key = f"{n_experts}_experts_seed_{seed}"
        results['by_config'][config_key] = defaultdict(lambda: {'probs': [], 'outcomes': []})
        
        # Process JSON files in output directory
        json_files = list(output_dir.glob("*.json"))
        print(f"Processing {output_dir.name}: {len(json_files)} files")
        
        for json_file in json_files:
            # Extract question ID from filename
            # Format: experts_comparison_MODEL_QUESTIONID_DATE.json
            parts = json_file.stem.split('_')
            if len(parts) >= 4:
                # Question ID is everything after the model name and before the date
                question_id = '_'.join(parts[3:-1]) if parts[-1].startswith('202') else '_'.join(parts[3:])
            else:
                continue
            
            # Get resolution from loaded data
            outcome = resolution_outcomes.get(question_id)
            
            if outcome is not None and outcome in [0, 1]:
                questions_with_outcomes += 1
                # We have a valid outcome, extract forecasts
                forecast_data = extract_forecasts_from_json(json_file)
                
                for round_num, expert_probs in forecast_data['rounds'].items():
                    for expert_id, prob in expert_probs.items():
                        # Add to aggregated data
                        results['by_n_experts'][n_experts][round_num]['probs'].append(prob)
                        results['by_n_experts'][n_experts][round_num]['outcomes'].append(outcome)
                        
                        results['by_round'][round_num]['probs'].append(prob)
                        results['by_round'][round_num]['outcomes'].append(outcome)
                        
                        results['by_config'][config_key][round_num]['probs'].append(prob)
                        results['by_config'][config_key][round_num]['outcomes'].append(outcome)
            else:
                questions_without_outcomes += 1
    
    print(f"Questions with outcomes: {questions_with_outcomes}")
    print(f"Questions without outcomes: {questions_without_outcomes}")
    
    return results

def plot_ece_by_n_experts(results: Dict, output_path: Path):
    """Plot ECE by number of experts across rounds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: ECE by round for each n_experts
    colors = plt.cm.viridis(np.linspace(0, 0.9, 6))
    
    for i, n_experts in enumerate(sorted(results['by_n_experts'].keys())):
        rounds = []
        eces = []
        counts = []
        
        for round_num in sorted(results['by_n_experts'][n_experts].keys()):
            probs = results['by_n_experts'][n_experts][round_num]['probs']
            outcomes = results['by_n_experts'][n_experts][round_num]['outcomes']
            
            if len(probs) > 0:
                ece, _ = calculate_ece(probs, outcomes)
                rounds.append(round_num)
                eces.append(ece)
                counts.append(len(probs))
        
        if rounds:
            ax1.plot(rounds, eces, 'o-', label=f'{n_experts} experts', 
                    color=colors[i], linewidth=2, markersize=8)
            
            # Add sample size annotations
            for r, e, c in zip(rounds, eces, counts):
                if r == rounds[-1]:  # Only annotate last point
                    ax1.annotate(f'n={c}', (r, e), 
                               xytext=(5, 0), textcoords='offset points', 
                               fontsize=7, alpha=0.7)
    
    ax1.set_xlabel('Delphi Round', fontsize=12)
    ax1.set_ylabel('Expected Calibration Error', fontsize=12)
    ax1.set_title('Calibration Error Evolution by Expert Group Size', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Right plot: Average ECE across all rounds
    n_experts_list = []
    avg_eces = []
    std_eces = []
    
    for n_experts in sorted(results['by_n_experts'].keys()):
        all_eces = []
        for round_num in results['by_n_experts'][n_experts].keys():
            probs = results['by_n_experts'][n_experts][round_num]['probs']
            outcomes = results['by_n_experts'][n_experts][round_num]['outcomes']
            if len(probs) > 0:
                ece, _ = calculate_ece(probs, outcomes)
                all_eces.append(ece)
        
        if all_eces:
            n_experts_list.append(n_experts)
            avg_eces.append(np.mean(all_eces))
            std_eces.append(np.std(all_eces))
    
    ax2.errorbar(n_experts_list, avg_eces, yerr=std_eces, 
                fmt='o-', linewidth=2.5, markersize=10,
                capsize=5, capthick=2, elinewidth=2, color='darkblue')
    
    ax2.set_xlabel('Number of Experts', fontsize=12)
    ax2.set_ylabel('Average ECE Across Rounds', fontsize=12)
    ax2.set_title('Average Calibration Error by Group Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(n_experts_list)
    
    plt.tight_layout()
    plt.savefig(output_path / 'ece_by_n_experts.png', dpi=150, bbox_inches='tight')
    return fig

def plot_calibration_curves(results: Dict, output_path: Path):
    """Plot calibration curves showing predicted vs actual frequencies."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    n_experts_values = sorted(results['by_n_experts'].keys())[:6]  # Max 6 subplots
    
    for idx, n_experts in enumerate(n_experts_values):
        ax = axes[idx]
        
        # Combine all rounds for this n_experts
        all_probs = []
        all_outcomes = []
        
        for round_num in results['by_n_experts'][n_experts].keys():
            all_probs.extend(results['by_n_experts'][n_experts][round_num]['probs'])
            all_outcomes.extend(results['by_n_experts'][n_experts][round_num]['outcomes'])
        
        if len(all_probs) > 0:
            # Create calibration plot
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            
            predicted_probs = []
            actual_freqs = []
            bin_counts = []
            
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                if i == n_bins - 1:
                    in_bin = [(p, o) for p, o in zip(all_probs, all_outcomes) 
                              if bin_lower <= p <= bin_upper]
                else:
                    in_bin = [(p, o) for p, o in zip(all_probs, all_outcomes) 
                              if bin_lower <= p < bin_upper]
                
                if len(in_bin) > 0:
                    bin_probs = [p for p, _ in in_bin]
                    bin_outcomes = [o for _, o in in_bin]
                    
                    predicted_probs.append(np.mean(bin_probs))
                    actual_freqs.append(np.mean(bin_outcomes))
                    bin_counts.append(len(in_bin))
            
            # Plot calibration curve
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
            
            # Plot with point sizes proportional to bin counts
            if predicted_probs:
                sizes = [min(200, 20 + c/2) for c in bin_counts]
                scatter = ax.scatter(predicted_probs, actual_freqs, s=sizes, 
                                   alpha=0.7, color='blue', edgecolor='darkblue', linewidth=1)
                ax.plot(predicted_probs, actual_freqs, 'b-', alpha=0.5, linewidth=2)
                
                # Calculate and display ECE
                ece, _ = calculate_ece(all_probs, all_outcomes)
                ax.text(0.05, 0.95, f'ECE = {ece:.3f}\nn = {len(all_probs)}', 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Mean Predicted Probability', fontsize=10)
            ax.set_ylabel('Actual Frequency', fontsize=10)
            ax.set_title(f'{n_experts} Experts', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_aspect('equal')
    
    # Remove unused subplots
    for idx in range(len(n_experts_values), 6):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Calibration Curves by Number of Experts', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'calibration_curves.png', dpi=150, bbox_inches='tight')
    return fig

def plot_ece_heatmap(results: Dict, output_path: Path):
    """Create heatmap of ECE values across rounds and n_experts."""
    # Prepare data for heatmap
    n_experts_values = sorted(results['by_n_experts'].keys())
    rounds = sorted(set(r for n in results['by_n_experts'].values() for r in n.keys()))
    
    # Create matrix
    ece_matrix = np.full((len(n_experts_values), len(rounds)), np.nan)
    
    for i, n_experts in enumerate(n_experts_values):
        for j, round_num in enumerate(rounds):
            if round_num in results['by_n_experts'][n_experts]:
                probs = results['by_n_experts'][n_experts][round_num]['probs']
                outcomes = results['by_n_experts'][n_experts][round_num]['outcomes']
                if len(probs) > 0:
                    ece, _ = calculate_ece(probs, outcomes)
                    ece_matrix[i, j] = ece
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(ece_matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest', vmin=0, vmax=0.3)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(rounds)))
    ax.set_yticks(np.arange(len(n_experts_values)))
    ax.set_xticklabels([f'Round {r}' for r in rounds])
    ax.set_yticklabels([f'{n} Experts' for n in n_experts_values])
    
    # Add text annotations
    for i in range(len(n_experts_values)):
        for j in range(len(rounds)):
            if not np.isnan(ece_matrix[i, j]):
                text = ax.text(j, i, f'{ece_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Expected Calibration Error (lower is better)', rotation=270, labelpad=20)
    
    ax.set_xlabel('Delphi Round', fontsize=12)
    ax.set_ylabel('Number of Experts', fontsize=12)
    ax.set_title('Calibration Error Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'ece_heatmap.png', dpi=150, bbox_inches='tight')
    return fig

def create_summary_statistics(results: Dict):
    """Print summary statistics for calibration analysis."""
    print("\n" + "="*80)
    print("CALIBRATION ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    total_forecasts = sum(len(results['by_round'][r]['probs']) for r in results['by_round'])
    print(f"\nTotal forecasts analyzed: {total_forecasts}")
    
    # By number of experts
    print("\n" + "-"*40)
    print("ECE by Number of Experts (averaged across rounds):")
    print("-"*40)
    print(f"{'N Experts':<12} {'Avg ECE':<12} {'Std ECE':<12} {'N Forecasts'}")
    
    for n_experts in sorted(results['by_n_experts'].keys()):
        all_eces = []
        total_n = 0
        for round_num in results['by_n_experts'][n_experts].keys():
            probs = results['by_n_experts'][n_experts][round_num]['probs']
            outcomes = results['by_n_experts'][n_experts][round_num]['outcomes']
            if len(probs) > 0:
                ece, _ = calculate_ece(probs, outcomes)
                all_eces.append(ece)
                total_n += len(probs)
        
        if all_eces:
            avg_ece = np.mean(all_eces)
            std_ece = np.std(all_eces)
            print(f"{n_experts:<12} {avg_ece:<12.4f} {std_ece:<12.4f} {total_n}")
    
    # By round
    print("\n" + "-"*40)
    print("ECE by Round (averaged across all experts):")
    print("-"*40)
    print(f"{'Round':<12} {'ECE':<12} {'N Forecasts'}")
    
    for round_num in sorted(results['by_round'].keys()):
        probs = results['by_round'][round_num]['probs']
        outcomes = results['by_round'][round_num]['outcomes']
        if len(probs) > 0:
            ece, _ = calculate_ece(probs, outcomes)
            print(f"{round_num:<12} {ece:<12.4f} {len(probs)}")
    
    # Find best configuration
    print("\n" + "-"*40)
    print("Best Calibrated Configurations:")
    print("-"*40)
    
    config_eces = []
    for config_key in results['by_config']:
        all_probs = []
        all_outcomes = []
        for round_num in results['by_config'][config_key].keys():
            all_probs.extend(results['by_config'][config_key][round_num]['probs'])
            all_outcomes.extend(results['by_config'][config_key][round_num]['outcomes'])
        
        if len(all_probs) > 0:
            ece, _ = calculate_ece(all_probs, all_outcomes)
            config_eces.append((config_key, ece, len(all_probs)))
    
    # Sort by ECE
    config_eces.sort(key=lambda x: x[1])
    
    # Show top 5 and bottom 5
    print("\nTop 5 Best Calibrated:")
    for config, ece, n in config_eces[:5]:
        print(f"  {config}: ECE = {ece:.4f} (n={n})")
    
    if len(config_eces) > 5:
        print("\nTop 5 Worst Calibrated:")
        for config, ece, n in config_eces[-5:]:
            print(f"  {config}: ECE = {ece:.4f} (n={n})")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Expected Calibration Error across sweep results"
    )
    parser.add_argument(
        "sweep_dir",
        help="Path to sweep results directory"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots"
    )
    
    args = parser.parse_args()
    
    sweep_path = Path(args.sweep_dir)
    if not sweep_path.exists():
        print(f"Error: {sweep_path} does not exist")
        sys.exit(1)
    
    print(f"Analyzing calibration for sweep: {sweep_path}")
    
    # Analyze calibration
    results = analyze_calibration_for_sweep(sweep_path)
    
    # Create output directory for plots
    output_path = sweep_path
    
    # Generate plots
    print("\nGenerating calibration plots...")
    
    fig1 = plot_ece_by_n_experts(results, output_path)
    print(f"Saved: {output_path / 'ece_by_n_experts.png'}")
    
    fig2 = plot_calibration_curves(results, output_path)
    print(f"Saved: {output_path / 'calibration_curves.png'}")
    
    fig3 = plot_ece_heatmap(results, output_path)
    print(f"Saved: {output_path / 'ece_heatmap.png'}")
    
    # Print summary statistics
    create_summary_statistics(results)
    
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()