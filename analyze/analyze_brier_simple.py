#!/usr/bin/env python
"""
Simple Brier score analysis without variance/error bars.
"""

import subprocess
import sys
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_model_name(folder_name):
    """Extract a clean model name from the folder name."""
    name = folder_name.replace('results_experts_comparison_', '')
    name = name.replace('results_prompt_comparison_', '')
    
    name_map = {
        'claude37_sonnet': 'Claude 3.7 Sonnet',
        'deepseek_r1': 'DeepSeek R1',
        'gpt_oss_120b': 'GPT OSS 120B',
        'gpt_oss_20b': 'GPT OSS 20B',
        'gpt5': 'GPT-5',
        'llama_maverick': 'Llama Maverick',
        'o1': 'O1',
        'o3': 'O3',
        'qwen3_32b': 'Qwen3 32B'
    }
    
    return name_map.get(name, name.replace('_', ' ').title())

def find_config_for_output_dir(output_dir):
    """Find the corresponding config file for an output directory."""
    dir_name = Path(output_dir).name
    
    if 'experts_comparison' in dir_name:
        config_pattern = dir_name.replace('results_', '') + '.yml'
    else:
        return None
    
    config_path = Path('../configs') / config_pattern
    if config_path.exists():
        return str(config_path)
    
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
        'llm_rounds': {}
    }
    
    if not output_text:
        return results
    
    # Parse SF Brier
    sf_match = re.search(r'\[SF\].*?median SF forecast.*?: ([\d.]+|nan)', output_text)
    if sf_match and sf_match.group(1) != 'nan':
        results['sf_brier'] = float(sf_match.group(1))
    
    # Parse Public Brier
    public_match = re.search(r'\[Public\].*?median Public forecast.*?: ([\d.]+|nan)', output_text)
    if public_match and public_match.group(1) != 'nan':
        results['public_brier'] = float(public_match.group(1))
    
    # Parse LLM rounds
    round_matches = re.findall(r'\[LLM\].*?median LLM forecast.*?at round (\d+): ([\d.]+|nan)', output_text)
    for round_num, brier in round_matches:
        if brier != 'nan':
            results['llm_rounds'][int(round_num)] = float(brier)
    
    return results

def plot_brier_simple(all_results, selected_models=None):
    """Create a simple plot without error bars."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Filter models
    if selected_models:
        filtered_results = {k: v for k, v in all_results.items() if k in selected_models}
    else:
        filtered_results = all_results
    
    # Color map for selected models
    color_map = {
        'O3': 'blue',
        'Claude 3.7 Sonnet': 'green',
        'GPT OSS 120B': 'red'
    }
    
    # Track baselines
    sf_values = []
    public_values = []
    
    for model_name, results in filtered_results.items():
        if not results['llm_rounds']:
            continue
        
        rounds = sorted(results['llm_rounds'].keys())
        briers = [results['llm_rounds'][r] for r in rounds]
        
        color = color_map.get(model_name, None)
        ax.plot(rounds, briers, 'o-', label=model_name, color=color,
                linewidth=2.5, markersize=10, alpha=0.9)
        
        if results['sf_brier'] is not None:
            sf_values.append(results['sf_brier'])
        if results['public_brier'] is not None:
            public_values.append(results['public_brier'])
    
    # Add baselines
    if sf_values:
        avg_sf = np.mean(sf_values)
        ax.axhline(y=avg_sf, color='darkgreen', linestyle='--', linewidth=2,
                   label=f'SF Median: {avg_sf:.3f}', alpha=0.7)
    
    if public_values:
        avg_public = np.mean(public_values)
        ax.axhline(y=avg_public, color='darkorange', linestyle='--', linewidth=2,
                   label=f'Public Median: {avg_public:.3f}', alpha=0.7)
    
    ax.set_xlabel('Delphi Round', fontsize=14)
    ax.set_ylabel('Brier Score (lower is better)', fontsize=14)
    ax.set_title('Model Performance: O3 vs Claude 3.7 vs GPT OSS 120B', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    # Set x-axis
    if rounds:
        ax.set_xticks(rounds)
        ax.set_xticklabels([str(r) for r in rounds])
    
    ax.set_ylim(bottom=0, top=0.25)
    
    plt.tight_layout()
    return fig

def main():
    # Find expert directories
    expert_dirs = sorted([d for d in Path('..').glob('results_experts_comparison_*')
                         if d.is_dir() and '_initial' not in d.name])
    
    print(f"Found {len(expert_dirs)} expert comparison directories")
    print("=" * 60)
    
    # Process directories
    all_results = {}
    for output_dir in expert_dirs:
        config_path = find_config_for_output_dir(output_dir)
        
        if not config_path:
            continue
        
        print(f"Processing {output_dir.name}...")
        output = run_icl_delphi_results(config_path)
        
        if output:
            results = parse_results(output)
            model_name = extract_model_name(output_dir.name)
            all_results[model_name] = results
            
            if results['llm_rounds']:
                rounds_str = ', '.join([f"R{r}: {b:.3f}" for r, b in sorted(results['llm_rounds'].items())])
                print(f"  {rounds_str}")
    
    print("=" * 60)
    
    # Create plot
    selected_models = ['O3', 'Claude 3.7 Sonnet', 'GPT OSS 120B']
    
    print("Creating Brier score plot...")
    fig = plot_brier_simple(all_results, selected_models=selected_models)
    fig.savefig('brier_scores_simple.png', dpi=150, bbox_inches='tight')
    print("Plot saved to brier_scores_simple.png")
    
    if '--no-show' not in sys.argv:
        plt.show()

if __name__ == "__main__":
    main()