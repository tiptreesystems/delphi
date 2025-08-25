#!/usr/bin/env python
"""
Script to analyze expert agreement and movement between rounds for sweep results.
Modified from analyze_agreement_and_movement.py to work with sweep directory structure.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_model_name(folder_name):
    """Extract a clean model name from the folder name."""
    # For sweep results, we'll extract number of experts only (ignore seed for grouping)
    if folder_name.startswith('results_n_experts_'):
        parts = folder_name.replace('results_n_experts_', '').split('_')
        if len(parts) >= 3 and parts[1] == 'seed':
            n_experts = parts[0]
            return f"{n_experts} Experts"
    
    # Fallback to original naming scheme if needed
    name = folder_name.replace('results_experts_comparison_', '')
    name = folder_name.replace('results_prompt_comparison_', '')
    
    name_map = {
        'claude37_sonnet': 'Claude 3.7 Sonnet',
        'deepseek_r1': 'DeepSeek R1',
        'gpt_oss_120b': 'GPT OSS 120B',
        'gpt_oss_20b': 'GPT OSS 20B',
        'gpt5': 'GPT-5',
        'llama_maverick': 'Llama Maverick',
        'o1': 'O1',
        'o3': 'O3',
        'qwen3_32b': 'Qwen3 32B',
        'baseline': 'Baseline',
        'baseline_with_examples': 'Baseline w/ Examples',
        'base_rate': 'Base Rate',
        'deep_analytical': 'Deep Analytical',
        'frequency_based': 'Frequency Based',
        'high_variance': 'High Variance',
        'opinionated': 'Opinionated',
        'short_focused': 'Short Focused'
    }
    
    return name_map.get(name, name.replace('_', ' ').title())

def aggregate_results_by_n_experts(all_results):
    """Aggregate results by number of experts, averaging across seeds."""
    aggregated = {}
    
    # Group by number of experts
    by_n_experts = defaultdict(list)
    for model_name, (agreement_stats, movement_stats) in all_results.items():
        if "Experts" in model_name and "(seed" in model_name:
            # Extract number from "N Experts (seed X)" format
            n_experts = model_name.split()[0]
            by_n_experts[n_experts].append((agreement_stats, movement_stats))
    
    # Average across seeds for each number of experts
    for n_experts, results_list in by_n_experts.items():
        # Collect all agreement and movement data across seeds
        all_agreement_by_round = defaultdict(list)
        all_movement_by_transition = defaultdict(list)
        
        for agreement_stats, movement_stats in results_list:
            # Collect agreement data
            for round_num, stats in agreement_stats.items():
                all_agreement_by_round[round_num].append(stats['mean'])
            
            # Collect movement data
            for transition, stats in movement_stats.items():
                all_movement_by_transition[transition].append(stats['mean'])
        
        # Calculate averages and std devs
        avg_agreement_stats = {}
        for round_num, means_list in all_agreement_by_round.items():
            if len(means_list) > 0:
                avg_agreement_stats[round_num] = {
                    'mean': np.mean(means_list),
                    'std': np.std(means_list, ddof=1) if len(means_list) > 1 else 0.0,
                    'n': len(means_list)
                }
        
        avg_movement_stats = {}
        for transition, means_list in all_movement_by_transition.items():
            if len(means_list) > 0:
                avg_movement_stats[transition] = {
                    'mean': np.mean(means_list),
                    'std': np.std(means_list, ddof=1) if len(means_list) > 1 else 0.0,
                    'n': len(means_list)
                }
        
        aggregated[f"{n_experts} Experts"] = (avg_agreement_stats, avg_movement_stats)
    
    return aggregated

def compute_brier_score(prob, outcome):
    """Compute Brier score for a binary outcome."""
    return (prob - outcome) ** 2

def analyze_agreement_and_movement(output_dir):
    """Analyze expert agreement (std dev) and movement between rounds."""
    output_dir = Path(output_dir)
    json_files = sorted([f for f in output_dir.glob("*.json") if f.is_file()])
    
    # Store agreement (std dev) and movement data
    agreement_by_round = defaultdict(list)  # round -> list of std devs across questions
    movement_between_rounds = defaultdict(list)  # (round, round+1) -> list of movements
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            rounds = data.get('rounds', [])
            
            # Track expert predictions across rounds for movement calculation
            expert_predictions_by_round = {}
            
            for round_data in rounds:
                round_num = round_data.get('round', 0)
                experts = round_data.get('experts', {})
                
                # Get all expert probabilities for this round
                probs = []
                expert_ids = []
                for expert_id, expert_data in experts.items():
                    if 'prob' in expert_data:
                        probs.append(expert_data['prob'])
                        expert_ids.append(expert_id)
                
                if probs:
                    # Calculate agreement (using std dev - lower means more agreement)
                    std_dev = np.std(probs)
                    agreement_by_round[round_num].append(std_dev)
                    
                    # Store for movement calculation
                    expert_predictions_by_round[round_num] = {
                        expert_id: prob for expert_id, prob in zip(expert_ids, probs)
                    }
            
            # Calculate movement between consecutive rounds
            rounds_sorted = sorted(expert_predictions_by_round.keys())
            for i in range(len(rounds_sorted) - 1):
                curr_round = rounds_sorted[i]
                next_round = rounds_sorted[i + 1]
                
                curr_preds = expert_predictions_by_round[curr_round]
                next_preds = expert_predictions_by_round[next_round]
                
                # Find common experts between rounds
                common_experts = set(curr_preds.keys()) & set(next_preds.keys())
                
                if common_experts:
                    # Calculate average absolute movement
                    movements = []
                    for expert_id in common_experts:
                        movement = abs(next_preds[expert_id] - curr_preds[expert_id])
                        movements.append(movement)
                    
                    avg_movement = np.mean(movements)
                    movement_between_rounds[(curr_round, next_round)].append(avg_movement)
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}", file=sys.stderr)
    
    # Calculate statistics
    agreement_stats = {}
    for round_num in sorted(agreement_by_round.keys()):
        std_devs = agreement_by_round[round_num]
        if std_devs:
            agreement_stats[round_num] = {
                'mean': np.mean(std_devs),
                'median': np.median(std_devs),
                'std': np.std(std_devs),
                'n': len(std_devs)
            }
    
    movement_stats = {}
    for (curr_round, next_round) in sorted(movement_between_rounds.keys()):
        movements = movement_between_rounds[(curr_round, next_round)]
        if movements:
            movement_stats[(curr_round, next_round)] = {
                'mean': np.mean(movements),
                'median': np.median(movements),
                'std': np.std(movements),
                'n': len(movements)
            }
    
    return agreement_stats, movement_stats

def plot_agreement(all_results, title="Expert Agreement (Lower = More Consensus)", selected_models=None):
    """Plot expert agreement (std dev of predictions) across rounds."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Filter models if specified
    if selected_models:
        filtered_results = {k: v for k, v in all_results.items() if k in selected_models}
    else:
        filtered_results = all_results
    
    # Color map - specific colors for different numbers of experts
    expert_colors = {
        '1 Experts': '#1f77b4',  # blue
        '2 Experts': '#ff7f0e',  # orange  
        '3 Experts': '#2ca02c',  # green
        '4 Experts': '#d62728',  # red
        '5 Experts': '#9467bd',  # purple
    }
    
    # Fallback to dynamic colors if needed
    if not all(model_name in expert_colors for model_name in filtered_results.keys()):
        colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_results)))
        color_map = {model_name: color for model_name, color in zip(filtered_results.keys(), colors)}
    else:
        color_map = expert_colors
    
    for model_name, (agreement_stats, _) in filtered_results.items():
        if not agreement_stats:
            continue
        
        rounds = sorted(agreement_stats.keys())
        mean_agreements = [agreement_stats[r]['mean'] for r in rounds]
        std_agreements = [agreement_stats[r]['std'] for r in rounds]
        
        color = color_map.get(model_name, None)
        ax.errorbar(rounds, mean_agreements, yerr=std_agreements, 
                   fmt='o-', label=model_name, color=color, 
                   linewidth=2, markersize=8, alpha=0.8, capsize=5)
    
    ax.set_xlabel('Delphi Round', fontsize=12)
    ax.set_ylabel('Average Std Dev of Expert Predictions', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Set x-axis to show integer rounds
    if rounds:
        ax.set_xticks(rounds)
        ax.set_xticklabels([str(r) for r in rounds])
    
    plt.tight_layout()
    return fig

def plot_movement(all_results, title="Average Movement Between Rounds", selected_models=None):
    """Plot average movement of predictions between consecutive rounds."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Filter models if specified
    if selected_models:
        filtered_results = {k: v for k, v in all_results.items() if k in selected_models}
    else:
        filtered_results = all_results
    
    # Color map - specific colors for different numbers of experts
    expert_colors = {
        '1 Experts': '#1f77b4',  # blue
        '2 Experts': '#ff7f0e',  # orange  
        '3 Experts': '#2ca02c',  # green
        '4 Experts': '#d62728',  # red
        '5 Experts': '#9467bd',  # purple
    }
    
    # Fallback to dynamic colors if needed
    if not all(model_name in expert_colors for model_name in filtered_results.keys()):
        colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_results)))
        color_map = {model_name: color for model_name, color in zip(filtered_results.keys(), colors)}
    else:
        color_map = expert_colors
    
    bar_width = 0.8 / len(filtered_results) if filtered_results else 0.8
    x_offset = 0
    
    for model_name, (_, movement_stats) in filtered_results.items():
        if not movement_stats:
            continue
        
        # Extract movement data
        transitions = []
        movements = []
        movement_stds = []
        for (curr_round, next_round), stats in sorted(movement_stats.items()):
            transitions.append(f"R{curr_round}→R{next_round}")
            movements.append(stats['mean'])
            movement_stds.append(stats['std'])
        
        if transitions:
            x_pos = np.arange(len(transitions)) + x_offset
            color = color_map.get(model_name, None)
            ax.bar(x_pos, movements, bar_width, yerr=movement_stds,
                   label=model_name, color=color, alpha=0.8, 
                   capsize=3, error_kw={'linewidth': 1})
            x_offset += bar_width
    
    ax.set_xlabel('Round Transition', fontsize=12)
    ax.set_ylabel('Average Absolute Change in Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=10)
    
    # Set x-axis labels
    if transitions:
        ax.set_xticks(np.arange(len(transitions)) + bar_width * (len(filtered_results) - 1) / 2)
        ax.set_xticklabels(transitions)
    
    plt.tight_layout()
    return fig

def main():
    if len(sys.argv) > 1:
        # Use directory provided as argument
        sweep_dir = Path(sys.argv[1])
        if not sweep_dir.exists():
            print(f"Directory {sweep_dir} does not exist")
            sys.exit(1)
    else:
        # Default to current directory
        sweep_dir = Path('..')
    
    # Find all output directories in the sweep results
    output_dirs = sorted([d for d in sweep_dir.glob('results_n_experts_*')
                         if d.is_dir() and '_initial' not in d.name])
    
    if not output_dirs:
        print(f"No expert output directories found in {sweep_dir}")
        print("Looking for directories matching pattern: results_n_experts_*")
        sys.exit(1)
    
    print(f"Found {len(output_dirs)} expert output directories")
    print("=" * 60)
    
    # Process each directory
    all_results = {}
    for output_dir in output_dirs:
        print(f"Processing {output_dir.name}...")
        agreement_stats, movement_stats = analyze_agreement_and_movement(output_dir)
        
        if agreement_stats:
            # Use full name including seed for individual results
            full_name = f"{extract_model_name(output_dir.name)} (seed {output_dir.name.split('_seed_')[1]})"
            all_results[full_name] = (agreement_stats, movement_stats)
            
            # Print summary
            print(f"  Agreement (avg std dev by round):")
            for r in sorted(agreement_stats.keys()):
                print(f"    R{r}: {agreement_stats[r]['mean']:.3f}")
            
            print(f"  Movement between rounds:")
            for (r1, r2) in sorted(movement_stats.keys()):
                print(f"    R{r1}→R{r2}: {movement_stats[(r1, r2)]['mean']:.3f}")
        else:
            print(f"  No data found in {output_dir.name}")
        print()
    
    if not all_results:
        print("No valid results found")
        sys.exit(1)
    
    # Aggregate results by number of experts (averaging across seeds)
    print("=" * 60)
    print("Aggregating results by number of experts (averaging across seeds)...")
    aggregated_results = aggregate_results_by_n_experts(all_results)
    
    # Print aggregated summary
    for model_name, (agreement_stats, movement_stats) in sorted(aggregated_results.items()):
        print(f"\n{model_name} (averaged across seeds):")
        print(f"  Agreement (avg std dev by round):")
        for r in sorted(agreement_stats.keys()):
            print(f"    R{r}: {agreement_stats[r]['mean']:.3f} ± {agreement_stats[r]['std']:.3f}")
        
        print(f"  Movement between rounds:")
        for (r1, r2) in sorted(movement_stats.keys()):
            print(f"    R{r1}→R{r2}: {movement_stats[(r1, r2)]['mean']:.3f} ± {movement_stats[(r1, r2)]['std']:.3f}")
    
    # Create plots using aggregated results
    print("=" * 60)
    
    # Plot agreement
    print("Creating aggregated agreement plot...")
    fig_agreement = plot_agreement(aggregated_results, 
                                  title="Expert Consensus Evolution by Group Size (Averaged Across Seeds)")
    fig_agreement.savefig('sweep_expert_agreement_aggregated.png', dpi=150, bbox_inches='tight')
    print("Aggregated agreement plot saved to sweep_expert_agreement_aggregated.png")
    
    # Plot movement
    print("Creating aggregated movement plot...")
    fig_movement = plot_movement(aggregated_results,
                                title="Opinion Change Between Rounds by Group Size (Averaged Across Seeds)")
    fig_movement.savefig('sweep_expert_movement_aggregated.png', dpi=150, bbox_inches='tight')
    print("Aggregated movement plot saved to sweep_expert_movement_aggregated.png")
    
    if '--no-show' not in sys.argv:
        plt.show()

if __name__ == "__main__":
    main()