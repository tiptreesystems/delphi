#!/usr/bin/env python
"""
Script to analyze expert agreement and movement between rounds.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_model_name(folder_name):
    """Extract a clean model name from the folder name."""
    name = folder_name.replace('outputs_experts_comparison_', '')
    name = name.replace('outputs_prompt_comparison_', '')
    
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
    
    # Color map for selected models
    color_map = {
        'O3': 'blue',
        'Claude 3.7 Sonnet': 'green',
        'GPT OSS 120B': 'red',
        'DeepSeek R1': 'purple',
        'Llama Maverick': 'orange'
    }
    
    for model_name, (agreement_stats, _) in filtered_results.items():
        if not agreement_stats:
            continue
        
        rounds = sorted(agreement_stats.keys())
        mean_agreements = [agreement_stats[r]['mean'] for r in rounds]
        
        color = color_map.get(model_name, None)
        ax.plot(rounds, mean_agreements, 'o-', label=model_name, 
                color=color, linewidth=2, markersize=8, alpha=0.8)
    
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
    
    # Color map for selected models
    color_map = {
        'O3': 'blue',
        'Claude 3.7 Sonnet': 'green',
        'GPT OSS 120B': 'red',
        'DeepSeek R1': 'purple',
        'Llama Maverick': 'orange'
    }
    
    bar_width = 0.15
    x_offset = 0
    
    for model_name, (_, movement_stats) in filtered_results.items():
        if not movement_stats:
            continue
        
        # Extract movement data
        transitions = []
        movements = []
        for (curr_round, next_round), stats in sorted(movement_stats.items()):
            transitions.append(f"R{curr_round}→R{next_round}")
            movements.append(stats['mean'])
        
        if transitions:
            x_pos = np.arange(len(transitions)) + x_offset
            color = color_map.get(model_name, None)
            ax.bar(x_pos, movements, bar_width, label=model_name, 
                   color=color, alpha=0.8)
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
    # Find all output directories
    expert_dirs = sorted([d for d in Path('.').glob('outputs_experts_comparison_*') 
                         if d.is_dir() and '_initial' not in d.name])
    
    if not expert_dirs:
        print("No expert comparison directories found")
        sys.exit(1)
    
    print(f"Found {len(expert_dirs)} expert comparison directories")
    print("=" * 60)
    
    # Process each directory
    all_results = {}
    for output_dir in expert_dirs:
        print(f"Processing {output_dir.name}...")
        agreement_stats, movement_stats = analyze_agreement_and_movement(output_dir)
        
        if agreement_stats:
            model_name = extract_model_name(output_dir.name)
            all_results[model_name] = (agreement_stats, movement_stats)
            
            # Print summary
            print(f"  Agreement (avg std dev by round):")
            for r in sorted(agreement_stats.keys()):
                print(f"    R{r}: {agreement_stats[r]['mean']:.3f}")
            
            print(f"  Movement between rounds:")
            for (r1, r2) in sorted(movement_stats.keys()):
                print(f"    R{r1}→R{r2}: {movement_stats[(r1, r2)]['mean']:.3f}")
        print()
    
    # Create plots
    print("=" * 60)
    
    # Select models to display
    selected_models = ['O3', 'Claude 3.7 Sonnet', 'GPT OSS 120B']
    
    # Plot agreement
    print("Creating agreement plot...")
    fig_agreement = plot_agreement(all_results, 
                                  title="Expert Consensus Evolution (Lower = More Agreement)",
                                  selected_models=selected_models)
    fig_agreement.savefig('expert_agreement.png', dpi=150, bbox_inches='tight')
    print("Agreement plot saved to expert_agreement.png")
    
    # Plot movement
    print("Creating movement plot...")
    fig_movement = plot_movement(all_results,
                                title="Opinion Change Between Rounds",
                                selected_models=selected_models)
    fig_movement.savefig('expert_movement.png', dpi=150, bbox_inches='tight')
    print("Movement plot saved to expert_movement.png")
    
    if '--no-show' not in sys.argv:
        plt.show()

if __name__ == "__main__":
    main()