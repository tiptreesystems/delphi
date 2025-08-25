#!/usr/bin/env python3
"""
Analyze and visualize parameter sweep results.

This script processes sweep results and creates plots showing how Brier scores
vary with different parameter values.
"""

import json
import sys
import subprocess
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict

def run_icl_delphi_results(config_path):
    """Run icl_delphi_results.py and capture the output."""
    try:
        result = subprocess.run(
            ['python3', 'icl_delphi_results.py', config_path],
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout
    except Exception as e:
        print(f"Error running icl_delphi_results.py with {config_path}: {e}", file=sys.stderr)
        return None

def parse_brier_scores(output_text):
    """Parse Brier scores from icl_delphi_results.py output."""
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

def load_results(sweep_dir):
    """Load and process sweep results from a directory."""
    sweep_dir = Path(sweep_dir)
    
    # Load sweep configuration
    sweep_config_path = sweep_dir / "sweep_config.json"
    if not sweep_config_path.exists():
        print(f"Error: No sweep_config.json found in {sweep_dir}")
        return None
    
    with open(sweep_config_path, 'r') as f:
        sweep_config = json.load(f)
    
    # Load results summary
    results_summary_path = sweep_dir / "results_summary.json"
    if not results_summary_path.exists():
        print(f"Warning: No results_summary.json found in {sweep_dir}")
        results_summary = []
    else:
        with open(results_summary_path, 'r') as f:
            results_summary = json.load(f)
    
    # Process each experiment result
    processed_results = []
    
    for result in results_summary:
        if result['success']:
            config_path = result['config_file']
            
            # Run icl_delphi_results to get Brier scores
            print(f"Processing {Path(config_path).name}...")
            output = run_icl_delphi_results(config_path)
            
            if output:
                brier_scores = parse_brier_scores(output)
                
                # Parse parameter values from the result
                param_values = {}
                if sweep_config.get('mode') == 'combinatorial':
                    # For combinatorial mode, parse multiple parameters
                    param_names = result['parameter'].split(' + ')
                    param_vals = result['value'].split(' + ')
                    for name, val in zip(param_names, param_vals):
                        # Try to convert to float if possible
                        try:
                            param_values[name] = float(val)
                        except ValueError:
                            param_values[name] = val
                else:
                    # For sequential mode
                    param_values[result['parameter']] = result['value']
                
                processed_results.append({
                    'config_file': result['config_file'],
                    'parameters': param_values,
                    'brier_scores': brier_scores,
                    'duration': result['duration']
                })
    
    return {
        'sweep_config': sweep_config,
        'results': processed_results
    }

def plot_sequential_sweep(data, param_name=None):
    """Create plot for sequential parameter sweep with error bars."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Extract parameter name if not provided
    if param_name is None:
        param_name = data['sweep_config']['parameters'][0]['name']
    
    # Group results by parameter value (handle multiple runs per value)
    param_to_briers = defaultdict(list)
    all_rounds_data = defaultdict(lambda: defaultdict(list))
    
    for result in data['results']:
        if param_name in result['parameters']:
            param_val = result['parameters'][param_name]
            
            # Get final round Brier score
            if result['brier_scores']['llm_rounds']:
                final_round = max(result['brier_scores']['llm_rounds'].keys())
                final_brier = result['brier_scores']['llm_rounds'][final_round]
                param_to_briers[param_val].append(final_brier)
                
                # Store all rounds for detailed plot
                for round_num, brier in result['brier_scores']['llm_rounds'].items():
                    all_rounds_data[round_num][param_val].append(brier)
    
    # Calculate means and standard errors
    param_values = []
    mean_briers = []
    std_errors = []
    
    for param_val in sorted(param_to_briers.keys(), key=lambda x: (isinstance(x, str), x)):
        briers = param_to_briers[param_val]
        param_values.append(param_val)
        mean_briers.append(np.mean(briers))
        
        # Calculate standard error (std / sqrt(n))
        if len(briers) > 1:
            std_error = np.std(briers, ddof=1) / np.sqrt(len(briers))
        else:
            std_error = 0
        std_errors.append(std_error)
    
    # Convert to numpy arrays for easier manipulation
    param_values = np.array(param_values)
    mean_briers = np.array(mean_briers)
    std_errors = np.array(std_errors)
    
    # Plot with error bars
    ax.errorbar(param_values, mean_briers, yerr=std_errors, 
                fmt='o-', linewidth=2.5, markersize=10,
                capsize=5, capthick=2, elinewidth=2,
                label=f'Mean ± SE (Final Round)', color='blue',
                alpha=0.8)
    
    # Add individual points as scatter (if multiple runs)
    for param_val, briers in param_to_briers.items():
        if len(briers) > 1:
            # Add some jitter to x-axis for visibility
            jitter = np.random.normal(0, 0.01, len(briers))
            x_positions = [param_val + j for j in jitter]
            ax.scatter(x_positions, briers, alpha=0.3, s=30, color='gray')
    
    # Add grid and labels
    ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel('Brier Score (lower is better)', fontsize=14)
    ax.set_title(f'Parameter Sweep: {param_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    # Set y-axis limits
    if len(mean_briers) > 0:
        y_min = (mean_briers - std_errors).min() * 0.95
        y_max = (mean_briers + std_errors).max() * 1.05
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig

def plot_combinatorial_sweep(data):
    """Create heatmap for combinatorial parameter sweep."""
    sweep_config = data['sweep_config']
    
    if len(sweep_config['parameters']) != 2:
        print("Heatmap visualization only supports 2-parameter combinatorial sweeps")
        return None
    
    param1 = sweep_config['parameters'][0]
    param2 = sweep_config['parameters'][1]
    
    # Get unique values for each parameter
    param1_values = sorted(set(param1['values']), key=lambda x: (isinstance(x, str), x))
    param2_values = sorted(set(param2['values']), key=lambda x: (isinstance(x, str), x))
    
    # Create matrix for heatmap
    brier_matrix = np.full((len(param2_values), len(param1_values)), np.nan)
    
    for result in data['results']:
        if result['brier_scores']['llm_rounds']:
            final_round = max(result['brier_scores']['llm_rounds'].keys())
            final_brier = result['brier_scores']['llm_rounds'][final_round]
            
            val1 = result['parameters'].get(param1['name'])
            val2 = result['parameters'].get(param2['name'])
            
            if val1 in param1_values and val2 in param2_values:
                i = param2_values.index(val2)
                j = param1_values.index(val1)
                brier_matrix[i, j] = final_brier
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    im = ax.imshow(brier_matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(param1_values)))
    ax.set_yticks(np.arange(len(param2_values)))
    ax.set_xticklabels([str(v) for v in param1_values])
    ax.set_yticklabels([str(v) for v in param2_values])
    
    # Add labels
    ax.set_xlabel(param1['name'].replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel(param2['name'].replace('_', ' ').title(), fontsize=14)
    ax.set_title('Combinatorial Parameter Sweep: Brier Scores (Final Round)', fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Brier Score (lower is better)', rotation=270, labelpad=20, fontsize=12)
    
    # Add text annotations
    for i in range(len(param2_values)):
        for j in range(len(param1_values)):
            if not np.isnan(brier_matrix[i, j]):
                text = ax.text(j, i, f'{brier_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_round_progression(data):
    """Plot how Brier scores evolve across rounds for different parameter values."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Group results by parameter configuration
    param_config_to_results = defaultdict(list)
    
    for result in data['results']:
        if result['brier_scores']['llm_rounds']:
            # Create a hashable key from parameters
            param_key = tuple(sorted(result['parameters'].items()))
            param_config_to_results[param_key].append(result)
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(param_config_to_results)))
    
    for idx, (param_key, results) in enumerate(param_config_to_results.items()):
        # Aggregate rounds data across multiple runs
        rounds_data = defaultdict(list)
        
        for result in results:
            for round_num, brier in result['brier_scores']['llm_rounds'].items():
                rounds_data[round_num].append(brier)
        
        # Calculate means and standard errors
        rounds = sorted(rounds_data.keys())
        mean_briers = []
        std_errors = []
        
        for round_num in rounds:
            briers = rounds_data[round_num]
            mean_briers.append(np.mean(briers))
            if len(briers) > 1:
                std_errors.append(np.std(briers, ddof=1) / np.sqrt(len(briers)))
            else:
                std_errors.append(0)
        
        # Create label from parameters
        param_str = ', '.join([f"{k}={v}" for k, v in dict(param_key).items()])
        
        # Plot with error bars
        ax.errorbar(rounds, mean_briers, yerr=std_errors,
                   fmt='o-', label=param_str, color=colors[idx],
                   linewidth=2, markersize=8, alpha=0.8,
                   capsize=3, capthick=1.5, elinewidth=1.5)
    
    # Add SF and Public baselines if available
    sf_values = [r['brier_scores']['sf_brier'] for r in data['results'] 
                 if r['brier_scores']['sf_brier'] is not None]
    public_values = [r['brier_scores']['public_brier'] for r in data['results']
                     if r['brier_scores']['public_brier'] is not None]
    
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
    ax.set_title('Brier Score Progression Across Rounds', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Position legend outside plot area if too many items
    if len(data['results']) > 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    return fig

def create_summary_table(data):
    """Create a summary table of sweep results."""
    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)
    
    # Determine sweep mode
    mode = data['sweep_config'].get('mode', 'sequential')
    print(f"Sweep Mode: {mode}")
    
    # Print parameter information
    print("\nParameters Swept:")
    for param in data['sweep_config']['parameters']:
        print(f"  - {param['name']}: {param['values']}")
    
    print("\n" + "-"*80)
    print("Results (sorted by final round Brier score):")
    print("-"*80)
    
    # Prepare results for sorting
    summary_data = []
    for result in data['results']:
        if result['brier_scores']['llm_rounds']:
            final_round = max(result['brier_scores']['llm_rounds'].keys())
            final_brier = result['brier_scores']['llm_rounds'][final_round]
            
            param_str = ', '.join([f"{k}={v}" for k, v in result['parameters'].items()])
            
            summary_data.append({
                'params': param_str,
                'final_brier': final_brier,
                'final_round': final_round,
                'all_rounds': result['brier_scores']['llm_rounds']
            })
    
    # Sort by final Brier score
    summary_data.sort(key=lambda x: x['final_brier'])
    
    # Print table
    print(f"{'Parameters':<40} {'Final Brier':<12} {'All Rounds'}")
    print("-"*80)
    
    for item in summary_data:
        rounds_str = ', '.join([f"R{r}:{b:.3f}" for r, b in sorted(item['all_rounds'].items())])
        print(f"{item['params']:<40} {item['final_brier']:<12.3f} {rounds_str}")
    
    if summary_data:
        print("\n" + "-"*80)
        best = summary_data[0]
        worst = summary_data[-1]
        print(f"Best configuration: {best['params']} (Brier: {best['final_brier']:.3f})")
        print(f"Worst configuration: {worst['params']} (Brier: {worst['final_brier']:.3f})")
        print(f"Improvement: {((worst['final_brier'] - best['final_brier']) / worst['final_brier'] * 100):.1f}%")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize parameter sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py results/20241219_150000
  python analyze_results.py results/20241219_150000 --no-show
  python analyze_results.py results/20241219_150000 --output sweep_analysis
        """
    )
    
    parser.add_argument(
        "sweep_dir",
        help="Path to sweep results directory"
    )
    
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots, only save them"
    )
    
    parser.add_argument(
        "--output",
        default="sweep_plots",
        help="Output prefix for plot files (default: sweep_plots)"
    )
    
    args = parser.parse_args()
    
    # Load sweep results
    print(f"Loading sweep results from {args.sweep_dir}...")
    data = load_results(args.sweep_dir)
    
    if not data:
        print("Failed to load sweep results")
        sys.exit(1)
    
    print(f"Loaded {len(data['results'])} successful experiment results")
    
    # Create summary table
    create_summary_table(data)
    
    # Determine output directory and ensure it exists
    sweep_path = Path(args.sweep_dir)
    output_prefix = sweep_path / args.output
    
    # Determine sweep mode and create appropriate plots
    mode = data['sweep_config'].get('mode', 'sequential')
    
    # Check if this is a combinatorial sweep with seeds (treat as sequential with error bars)
    param_names = [p['name'] for p in data['sweep_config']['parameters']]
    is_seed_sweep = 'seed' in param_names and len(param_names) == 2

    if mode == 'combinatorial' and len(data['sweep_config']['parameters']) == 2 and not is_seed_sweep:
        # Create heatmap for 2-parameter combinatorial sweep (but not seed sweeps)
        print("\nCreating combinatorial heatmap...")
        fig = plot_combinatorial_sweep(data)
        if fig:
            plot_path = f'{output_prefix}_heatmap.png'
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Heatmap saved to {plot_path}")
    elif mode == 'sequential' or is_seed_sweep:
        # Create line plot for sequential sweep or seed sweep with error bars
        print("\nCreating sequential sweep plot with error bars...")
        
        # For seed sweeps, determine the primary parameter (not seed)
        if is_seed_sweep:
            primary_param = [name for name in param_names if name != 'seed'][0]
            fig = plot_sequential_sweep(data, param_name=primary_param)
        else:
            fig = plot_sequential_sweep(data)
            
        if fig:
            plot_path = f'{output_prefix}_sequential.png'
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
    
    # Always create round progression plot
    print("Creating round progression plot...")
    fig = plot_round_progression(data)
    if fig:
        plot_path = f'{output_prefix}_progression.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
    
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()