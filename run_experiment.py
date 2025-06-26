#!/usr/bin/env python3
"""
Run Delphi evaluation experiments with specified configuration files.

Usage:
    python run_experiment.py                    # Uses default configs/config.yml
    python run_experiment.py config_openai.yml  # Looks in configs/ folder
    python run_experiment.py path/to/config.yml # Uses full path
"""

import sys
import os
from eval import run_comprehensive_evaluation, load_config

def main():
    # Get config file from command line or use default
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        # If no path specified, assume it's in configs folder
        if not os.path.dirname(config_file):
            config_file = os.path.join("configs", config_file)
    else:
        config_file = "configs/config.yml"
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    
    print(f"Loading configuration from: {config_file}")
    config = load_config(config_file)
    
    print(f"\nExperiment: {config['experiment']['name']}")
    print(f"Model: {config['model']['provider']} - {config['model']['name']}")
    print(f"Questions: {config['evaluation']['n_questions']}")
    print(f"Expert range: {config['evaluation']['n_experts_range']}")
    print(f"Max workers: {config['evaluation']['max_workers']}")
    if 'random_seed' in config['evaluation']:
        print(f"Random seed: {config['evaluation']['random_seed']}")
    print("-" * 60)

    # Run the evaluation
    results = run_comprehensive_evaluation(config=config)
    
    results_dir = config['experiment']['results_dir'].format(
        timestamp=results['metadata']['timestamp']
    )
    print(f"\nExperiment completed!")
    print(f"Results saved to: {results_dir}/")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for n_experts in sorted(results['aggregate_metrics'].keys()):
        metrics = results['aggregate_metrics'][n_experts]
        print(f"\nN={n_experts} experts:")
        print(f"  Brier: {metrics['mean_brier']:.4f} (±{metrics['std_brier']:.4f})")
        print(f"  MAE: {metrics['mean_mae']:.4f} (±{metrics['std_mae']:.4f})")

if __name__ == "__main__":
    main() 