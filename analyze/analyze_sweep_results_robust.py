#!/usr/bin/env python3
"""
Robust version of analyze_results.py that works with partial/incomplete sweeps.

This script can handle:
- Incomplete sweeps with missing experiments
- Incorrect success flags in results_summary.json  
- Missing results_summary.json entirely
- Timeouts and partial results

It scans for actual output directories and processes whatever data is available.
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

def extract_params_from_output_dir(output_dir_name):
    """Extract parameter values from output directory name."""
    params = {}
    
    # Handle n_experts_X_seed_Y pattern
    if 'results_n_experts_' in output_dir_name:
        parts = output_dir_name.replace('results_n_experts_', '').split('_')
        
        if len(parts) >= 3 and parts[1] == 'seed':
            # Standard format: results_n_experts_X_seed_Y
            params['n_experts'] = int(parts[0])
            params['seed'] = int(parts[2])
        elif len(parts) == 1 and parts[0].isdigit():
            # Simple format: results_n_experts_X (assume default seed)
            params['n_experts'] = int(parts[0])
            params['seed'] = 42  # Default seed assumption
    
    return params

def scan_for_actual_results(sweep_dir):
    """Scan directory for actual output directories and build results list."""
    sweep_dir = Path(sweep_dir)
    
    # Look for output directories
    output_dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith('results_')]
    
    # Filter out _initial directories
    output_dirs = [d for d in output_dirs if not d.name.endswith('_initial')]
    
    results = []
    
    for output_dir in output_dirs:
        # Extract parameters from directory name
        params = extract_params_from_output_dir(output_dir.name)
        
        if not params:
            print(f"Could not parse parameters from {output_dir.name}, skipping...")
            continue
        
        # Check if directory has JSON files (indicating completed experiments)
        json_files = list(output_dir.glob("*.json"))
        
        if json_files:
            # Find corresponding config file
            if 'n_experts' in params:
                if 'seed' in params and params['seed'] != 42:
                    # Standard format with explicit seed
                    config_name = f"config_n_experts_{params['n_experts']}_seed_{params['seed']}.yml"
                else:
                    # Simple format - try both with and without seed
                    config_name = f"config_n_experts_{params['n_experts']}.yml"
                    config_path_simple = sweep_dir / config_name
                    if not config_path_simple.exists():
                        # Fallback to seed format with default seed
                        config_name = f"config_n_experts_{params['n_experts']}_seed_{params['seed']}.yml"
            else:
                print(f"Could not determine config file for {output_dir.name}, skipping...")
                continue
                
            config_path = sweep_dir / config_name
            
            if config_path.exists():
                # Create parameter string
                param_names = [name for name in sorted(params.keys()) if name != 'seed' or params[name] != 42]
                if len(param_names) == 2:  # n_experts + seed (non-default)
                    param_str = " + ".join(param_names)
                    value_str = " + ".join(str(params[name]) for name in param_names)
                elif len(param_names) == 1:  # Just n_experts
                    param_str = param_names[0]
                    value_str = str(params[param_names[0]])
                else:
                    param_str = "unknown"
                    value_str = "unknown"
                
                result = {
                    "parameter": param_str,
                    "value": value_str,
                    "config_file": str(config_path),
                    "output_dir": str(output_dir),
                    "success": True,  # We found JSON files, so consider it successful
                    "n_json_files": len(json_files),
                    "parameters": params
                }
                results.append(result)
                print(f"Found {len(json_files)} results in {output_dir.name}")
            else:
                print(f"Config file not found for {output_dir.name}: {config_path}")
        else:
            print(f"No JSON files found in {output_dir.name}, skipping...")
    
    return results

def load_results_robust(sweep_dir):
    """Load and process sweep results, handling partial/incomplete sweeps."""
    sweep_dir = Path(sweep_dir)
    
    # Load sweep configuration
    sweep_config_path = sweep_dir / "sweep_config.json"
    if sweep_config_path.exists():
        with open(sweep_config_path, 'r') as f:
            sweep_config = json.load(f)
    else:
        # Create a minimal sweep config if not found
        print(f"Warning: No sweep_config.json found in {sweep_dir}")
        sweep_config = {
            "mode": "combinatorial",
            "parameters": [
                {"name": "n_experts", "values": [1, 2, 3, 4, 5]},
                {"name": "seed", "values": [42, 123, 456]}
            ]
        }
    
    # Try to load results summary, but don't fail if it's missing or incorrect
    results_summary_path = sweep_dir / "results_summary.json"
    if results_summary_path.exists():
        try:
            with open(results_summary_path, 'r') as f:
                results_summary = json.load(f)
            print(f"Loaded {len(results_summary)} entries from results_summary.json")
        except Exception as e:
            print(f"Warning: Could not load results_summary.json: {e}")
            results_summary = []
    else:
        print(f"Warning: No results_summary.json found in {sweep_dir}")
        results_summary = []
    
    # Scan for actual results (more reliable than results_summary.json)
    print("Scanning for actual output directories...")
    actual_results = scan_for_actual_results(sweep_dir)
    
    # Process each actual result
    processed_results = []
    
    for result in actual_results:
        config_path = result['config_file']
        
        # Run icl_delphi_results to get Brier scores
        print(f"Processing {Path(config_path).name}...")
        output = run_icl_delphi_results(config_path)
        
        if output:
            brier_scores = parse_brier_scores(output)
            
            # Check if we got any meaningful results
            has_results = (brier_scores['llm_rounds'] or 
                          brier_scores['sf_brier'] is not None or 
                          brier_scores['public_brier'] is not None)
            
            if has_results:
                processed_results.append({
                    'config_file': result['config_file'],
                    'parameters': result['parameters'],
                    'brier_scores': brier_scores,
                    'n_json_files': result['n_json_files'],
                    'duration': None  # We don't have duration info from scanning
                })
                print(f"  ✓ Successfully extracted Brier scores")
            else:
                print(f"  ✗ No Brier scores found (resolution data may be missing)")
        else:
            print(f"  ✗ Failed to run icl_delphi_results.py")
    
    return {
        'sweep_config': sweep_config,
        'results': processed_results,
        'total_found': len(actual_results),
        'total_processed': len(processed_results)
    }

def plot_sequential_sweep_robust(data, param_name=None):
    """Create plot for sequential parameter sweep with error bars, handling sparse data."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Extract parameter name if not provided
    if param_name is None:
        if data['sweep_config']['parameters']:
            param_name = data['sweep_config']['parameters'][0]['name']
        else:
            param_name = 'parameter'
    
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
    
    if not param_to_briers:
        ax.text(0.5, 0.5, 'No data available\n(resolution data may be missing)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Parameter Sweep: {param_name.replace("_", " ").title()} (No Data)', 
                    fontsize=16, fontweight='bold')
        return fig
    
    # Calculate means and standard errors
    param_values = []
    mean_briers = []
    std_errors = []
    sample_sizes = []
    
    for param_val in sorted(param_to_briers.keys(), key=lambda x: (isinstance(x, str), x)):
        briers = param_to_briers[param_val]
        param_values.append(param_val)
        mean_briers.append(np.mean(briers))
        sample_sizes.append(len(briers))
        
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
    for i, (param_val, briers) in enumerate(zip(param_values, [param_to_briers[pv] for pv in param_values])):
        if len(briers) > 1:
            # Add some jitter to x-axis for visibility
            jitter = np.random.normal(0, 0.01, len(briers))
            x_positions = [param_val + j for j in jitter]
            ax.scatter(x_positions, briers, alpha=0.3, s=30, color='gray')
    
    # Add sample size annotations
    for param_val, n in zip(param_values, sample_sizes):
        ax.annotate(f'n={n}', (param_val, mean_briers[list(param_values).index(param_val)]), 
                   xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8, alpha=0.7)
    
    # Add grid and labels
    ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel('Brier Score (lower is better)', fontsize=14)
    ax.set_title(f'Parameter Sweep: {param_name.replace("_", " ").title()} (Partial Results)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    # Set y-axis limits
    if len(mean_briers) > 0:
        y_min = (mean_briers - std_errors).min() * 0.95
        y_max = (mean_briers + std_errors).max() * 1.05
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig

def plot_round_progression_robust(data):
    """Plot how Brier scores evolve across rounds, handling sparse data."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Check if this is a sweep with seeds
    has_seeds = any('seed' in result['parameters'] for result in data['results'] if result['brier_scores']['llm_rounds'])
    
    if has_seeds:
        # Group results by n_experts (averaging over seeds)
        n_experts_to_results = defaultdict(list)
        
        for result in data['results']:
            if result['brier_scores']['llm_rounds'] and 'n_experts' in result['parameters']:
                n_experts = result['parameters']['n_experts']
                n_experts_to_results[n_experts].append(result)
        
        if not n_experts_to_results:
            ax.text(0.5, 0.5, 'No round progression data available\n(resolution data may be missing)', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Brier Score Progression Across Rounds (No Data)', fontsize=16, fontweight='bold')
            return fig
        
        # Color palette
        n_experts_values = sorted(n_experts_to_results.keys())
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_experts_values)))
        
        for idx, n_experts in enumerate(n_experts_values):
            # Aggregate rounds data across all seeds for this n_experts value
            rounds_data = defaultdict(list)
            
            for result in n_experts_to_results[n_experts]:
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
            
            # Plot with error bars
            ax.errorbar(rounds, mean_briers, yerr=std_errors,
                       fmt='o-', label=f'n_experts={n_experts}', color=colors[idx],
                       linewidth=2, markersize=8, alpha=0.8,
                       capsize=3, capthick=1.5, elinewidth=1.5)
    else:
        # Original behavior for non-seed sweeps
        # Group results by parameter configuration
        param_config_to_results = defaultdict(list)
        
        for result in data['results']:
            if result['brier_scores']['llm_rounds']:
                # Create a hashable key from parameters
                param_key = tuple(sorted(result['parameters'].items()))
                param_config_to_results[param_key].append(result)
        
        if not param_config_to_results:
            ax.text(0.5, 0.5, 'No round progression data available\n(resolution data may be missing)', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Brier Score Progression Across Rounds (No Data)', fontsize=16, fontweight='bold')
            return fig
        
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
            param_dict = dict(param_key)
            param_str = ', '.join([f"{k}={v}" for k, v in param_dict.items()])
            
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
    ax.set_title('Brier Score Progression Across Rounds (Partial Results)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Position legend outside plot area if too many items
    if has_seeds:
        # For seed sweeps, check number of n_experts values
        if len(n_experts_to_results) > 5:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        else:
            ax.legend(loc='best', fontsize=11)
    else:
        # For non-seed sweeps, check number of parameter configurations
        if len(param_config_to_results) > 5:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        else:
            ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    return fig

def create_summary_table_robust(data):
    """Create a summary table of sweep results, handling sparse data."""
    print("\n" + "="*80)
    print("ROBUST SWEEP RESULTS SUMMARY")
    print("="*80)
    
    print(f"Scan Summary:")
    print(f"  - Output directories found: {data['total_found']}")
    print(f"  - Successfully processed: {data['total_processed']}")
    if data['total_processed'] < data['total_found']:
        print(f"  - Failed to process: {data['total_found'] - data['total_processed']} (likely missing resolution data)")
    
    # Determine sweep mode
    mode = data['sweep_config'].get('mode', 'combinatorial')
    print(f"\nSweep Mode: {mode}")
    
    # Print parameter information
    print("\nParameters Swept:")
    for param in data['sweep_config']['parameters']:
        print(f"  - {param['name']}: {param['values']}")
    
    if not data['results']:
        print("\n" + "-"*80)
        print("No processable results found!")
        print("This usually means resolution data is missing for Brier score calculation.")
        print("-"*80)
        print("="*80)
        return
    
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
        description="Robustly analyze partial/incomplete parameter sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results_robust.py results/20250820_184056
  python analyze_results_robust.py results/20250820_184056 --no-show
  python analyze_results_robust.py results/20250820_184056 --output partial_analysis

This script can handle:
- Incomplete sweeps with missing experiments  
- Incorrect success flags in results_summary.json
- Missing results_summary.json entirely
- Timeouts and partial results
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
        default="robust_sweep_plots",
        help="Output prefix for plot files (default: robust_sweep_plots)"
    )
    
    args = parser.parse_args()
    
    # Load sweep results robustly
    print(f"Robustly loading sweep results from {args.sweep_dir}...")
    data = load_results_robust(args.sweep_dir)
    
    print(f"Found {data['total_found']} output directories, successfully processed {data['total_processed']}")
    
    # Create summary table
    create_summary_table_robust(data)
    
    # Determine output directory and ensure it exists
    sweep_path = Path(args.sweep_dir)
    output_prefix = sweep_path / args.output
    
    # Determine sweep mode and create appropriate plots
    mode = data['sweep_config'].get('mode', 'sequential')
    
    # Check if this is a combinatorial sweep with seeds (treat as sequential with error bars)
    param_names = [p['name'] for p in data['sweep_config']['parameters']]
    is_seed_sweep = 'seed' in param_names and len(param_names) == 2
    
    if mode == 'combinatorial' and len(data['sweep_config']['parameters']) == 2 and not is_seed_sweep:
        print("\nNote: Combinatorial heatmaps not yet implemented in robust version")
        print("Creating sequential plot instead...")
        
    # Always create sequential plot
    print("Creating robust sequential sweep plot...")
    
    # For seed sweeps, determine the primary parameter (not seed)
    if is_seed_sweep:
        primary_param = [name for name in param_names if name != 'seed'][0]
        fig = plot_sequential_sweep_robust(data, param_name=primary_param)
    else:
        fig = plot_sequential_sweep_robust(data)
        
    if fig:
        plot_path = f'{output_prefix}_sequential.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Sequential plot saved to {plot_path}")
    
    # Always create round progression plot
    print("Creating robust round progression plot...")
    fig = plot_round_progression_robust(data)
    if fig:
        plot_path = f'{output_prefix}_progression.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Progression plot saved to {plot_path}")
    
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()