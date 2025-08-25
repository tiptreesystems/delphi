#!/usr/bin/env python
"""
Script to run icl_delphi_results.py on all expert comparison folders and plot the results.
"""

import subprocess
import sys
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_model_name(folder_name):
    """Extract a clean model name from the folder name."""
    # Remove prefix and suffix
    name = folder_name.replace('results_experts_comparison_', '')
    name = name.replace('results_prompt_comparison_', '')
    
    # Clean up common patterns
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

def find_config_for_output_dir(output_dir):
    """Find the corresponding config file for an output directory."""
    # Map output directory to config pattern
    dir_name = Path(output_dir).name
    
    # Try to find matching config
    if 'experts_comparison' in dir_name:
        config_pattern = dir_name.replace('results_', '') + '.yml'
    elif 'prompt_comparison' in dir_name:
        config_pattern = dir_name.replace('results_', '') + '.yml'
    else:
        return None
    
    config_path = Path('../configs') / config_pattern
    if config_path.exists():
        return str(config_path)
    
    # Try without 'results_' prefix
    config_pattern2 = dir_name.replace('results_', '').replace('_comparison', '_comparison') + '.yml'
    config_path2 = Path('../configs') / config_pattern2
    if config_path2.exists():
        return str(config_path2)
    
    return None

def run_icl_delphi_results(config_path):
    """Run icl_delphi_results.py and capture the output."""
    try:
        result = subprocess.run(
            ['python', 'icl_delphi_results.py', config_path],
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout
    except Exception as e:
        print(f"Error running icl_delphi_results.py with {config_path}: {e}", file=sys.stderr)
        return None

def parse_results(output_text):
    """Parse the output from icl_delphi_results.py."""
    results = {
        'sf_brier': None,
        'public_brier': None,
        'llm_baseline': None,
        'llm_rounds': {},
        'n_questions': None
    }
    
    if not output_text:
        return results
    
    # Parse SF Brier
    sf_match = re.search(r'\[SF\].*?median SF forecast.*?: ([\d.]+|nan).*?\(n=(\d+)', output_text)
    if sf_match and sf_match.group(1) != 'nan':
        results['sf_brier'] = float(sf_match.group(1))
        if sf_match.group(2):
            results['n_questions'] = int(sf_match.group(2))
    
    # Parse Public Brier
    public_match = re.search(r'\[Public\].*?median Public forecast.*?: ([\d.]+|nan)', output_text)
    if public_match and public_match.group(1) != 'nan':
        results['public_brier'] = float(public_match.group(1))
    
    # Parse LLM baseline (no-example)
    baseline_match = re.search(r'\[LLM\].*?no-example.*?median LLM forecast.*?: ([\d.]+|nan)', output_text)
    if baseline_match and baseline_match.group(1) != 'nan':
        results['llm_baseline'] = float(baseline_match.group(1))
    
    # Parse LLM rounds with sample sizes
    round_matches = re.findall(r'\[LLM\].*?median LLM forecast.*?at round (\d+): ([\d.]+|nan).*?\(n=(\d+)', output_text)
    for round_num, brier, n in round_matches:
        if brier != 'nan':
            results['llm_rounds'][int(round_num)] = float(brier)
            if results['n_questions'] is None and n:
                results['n_questions'] = int(n)
    
    return results

def plot_results(all_results, exclude_models=['GPT-5', 'Gpt5']):
    """Create a plot with all the results, with error bars."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Filter out excluded models
    filtered_results = {k: v for k, v in all_results.items() 
                       if k not in exclude_models}
    
    # Color palette - using distinct colors for each model
    colors = plt.cm.tab20(np.linspace(0, 1, len(filtered_results)))
    
    # Track SF and Public baselines
    sf_values = []
    public_values = []
    
    for idx, (model_name, results) in enumerate(filtered_results.items()):
        if not results['llm_rounds']:
            print(f"Skipping {model_name}: no LLM round data", file=sys.stderr)
            continue
        
        # Get rounds and Brier scores
        rounds = sorted(results['llm_rounds'].keys())
        briers = [results['llm_rounds'][r] for r in rounds]
        
        # Calculate error bars using standard error formula for Brier scores
        # Standard error = sqrt(brier * (1-brier) / n)
        n_questions = results.get('n_questions', 20)  # Default to 20 if not found
        errors = [np.sqrt(b * (1 - b) / n_questions) for b in briers]
        
        # Plot LLM performance across rounds with error bars
        ax.errorbar(rounds, briers, yerr=errors, fmt='o-', label=model_name, 
                   color=colors[idx], linewidth=2, markersize=8, alpha=0.8,
                   capsize=5, capthick=1.5)
        
        # Collect SF and Public values
        if results['sf_brier'] is not None:
            sf_values.append(results['sf_brier'])
        if results['public_brier'] is not None:
            public_values.append(results['public_brier'])
    
    # Add horizontal lines for SF and Public baselines with shaded regions
    if sf_values:
        avg_sf = np.mean(sf_values)
        std_sf = np.std(sf_values) if len(sf_values) > 1 else 0
        ax.axhline(y=avg_sf, color='green', linestyle='--', linewidth=2, 
                   label=f'SF Median (avg: {avg_sf:.3f})', alpha=0.7)
        if std_sf > 0:
            ax.fill_between(ax.get_xlim(), avg_sf - std_sf, avg_sf + std_sf, 
                          color='green', alpha=0.1)
    
    if public_values:
        avg_public = np.mean(public_values)
        std_public = np.std(public_values) if len(public_values) > 1 else 0
        ax.axhline(y=avg_public, color='orange', linestyle='--', linewidth=2,
                   label=f'Public Median (avg: {avg_public:.3f})', alpha=0.7)
        if std_public > 0:
            ax.fill_between(ax.get_xlim(), avg_public - std_public, avg_public + std_public,
                          color='orange', alpha=0.1)
    
    # Formatting
    ax.set_xlabel('Delphi Round', fontsize=12)
    ax.set_ylabel('Brier Score (lower is better)', fontsize=12)
    ax.set_title('Brier Scores Across Delphi Rounds', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, ncol=2)
    
    # Set x-axis to show integer rounds
    if rounds:
        ax.set_xticks(rounds)
        ax.set_xticklabels([str(r) for r in rounds])
    
    # Set y-axis limits for better visibility
    ax.set_ylim(bottom=0, top=max(0.25, ax.get_ylim()[1]))
    
    plt.tight_layout()
    return fig

def main():
    # Find all output directories
    expert_dirs = sorted([d for d in Path('..').glob('results_experts_comparison_*')
                         if d.is_dir() and '_initial' not in d.name])
    prompt_dirs = sorted([d for d in Path('..').glob('results_prompt_comparison_*')
                         if d.is_dir() and '_initial' not in d.name])
    
    if not expert_dirs and not prompt_dirs:
        print("No output directories found")
        sys.exit(1)
    
    print(f"Found {len(expert_dirs)} expert comparison directories")
    print(f"Found {len(prompt_dirs)} prompt comparison directories")
    print("=" * 60)
    
    # Process expert comparisons
    expert_results = {}
    if expert_dirs:
        print("\nProcessing Expert Comparisons:")
        print("-" * 40)
        for output_dir in expert_dirs:
            config_path = find_config_for_output_dir(output_dir)
            
            if not config_path:
                print(f"Warning: No config found for {output_dir}, skipping", file=sys.stderr)
                continue
            
            print(f"Processing {output_dir.name}...")
            print(f"  Using config: {config_path}")
            
            output = run_icl_delphi_results(config_path)
            
            if output:
                results = parse_results(output)
                model_name = extract_model_name(output_dir.name)
                expert_results[model_name] = results
                
                if results['llm_rounds']:
                    rounds_str = ', '.join([f"R{r}: {b:.3f}" for r, b in sorted(results['llm_rounds'].items())])
                    print(f"  Results: {rounds_str}")
                else:
                    print(f"  No results found")
            else:
                print(f"  Failed to run analysis")
            print()
    
    # Process prompt comparisons
    prompt_results = {}
    if prompt_dirs:
        print("\nProcessing Prompt Comparisons:")
        print("-" * 40)
        for output_dir in prompt_dirs:
            config_path = find_config_for_output_dir(output_dir)
            
            if not config_path:
                print(f"Warning: No config found for {output_dir}, skipping", file=sys.stderr)
                continue
            
            print(f"Processing {output_dir.name}...")
            print(f"  Using config: {config_path}")
            
            output = run_icl_delphi_results(config_path)
            
            if output:
                results = parse_results(output)
                model_name = extract_model_name(output_dir.name)
                prompt_results[model_name] = results
                
                if results['llm_rounds']:
                    rounds_str = ', '.join([f"R{r}: {b:.3f}" for r, b in sorted(results['llm_rounds'].items())])
                    print(f"  Results: {rounds_str}")
                else:
                    print(f"  No results found")
            else:
                print(f"  Failed to run analysis")
            print()
    
    # Create plots
    print("=" * 60)
    
    # Plot expert comparisons
    if expert_results:
        print("Creating expert comparison plot...")
        fig = plot_results(expert_results)
        fig.suptitle('Expert Model Comparison: Brier Scores Across Delphi Rounds', 
                     fontsize=14, fontweight='bold', y=1.02)
        output_file = 'expert_comparison_brier_scores.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Expert comparison plot saved to {output_file}")
        
        if '--no-show' not in sys.argv:
            plt.show()
    
    # Plot prompt comparisons
    if prompt_results:
        print("Creating prompt comparison plot...")
        fig = plot_results(prompt_results)
        fig.suptitle('Prompt Strategy Comparison: Brier Scores Across Delphi Rounds', 
                     fontsize=14, fontweight='bold', y=1.02)
        output_file = 'prompt_comparison_brier_scores.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Prompt comparison plot saved to {output_file}")
        
        if '--no-show' not in sys.argv:
            plt.show()
    
    if not expert_results and not prompt_results:
        print("No results to plot")

if __name__ == "__main__":
    main()