#!/usr/bin/env python3
"""
Parameter Sweep Script for Delphi Experiments

This script runs experiments while sweeping over one or more parameters.
The sweep configuration can be defined in the YAML config file or hardcoded.

Example sweep parameters:
- temperature: Model temperature for experts or mediator
- n_experts: Number of experts in the Delphi panel
- n_rounds: Number of Delphi rounds
- max_tokens: Maximum tokens for generation
"""

import subprocess
import sys
import os
import time
import argparse
import yaml
import json
import asyncio
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, path):
    """Save configuration to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def update_nested_dict(d, path, value):
    """Update a nested dictionary value using dot notation path."""
    keys = path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_nested_value(d, path):
    """Get a nested dictionary value using dot notation path."""
    keys = path.split('.')
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def create_sweep_output_dir(results_dir, param_name, param_value):
    """Create a unique output directory for this sweep iteration within the sweep results directory."""
    # Clean parameter value for directory name
    param_value_str = str(param_value).replace('.', '_').replace('/', '_')
    
    # Create directory name with parameter info inside the sweep results directory
    dir_name = results_dir / f"results_{param_name}_{param_value_str}"
    
    return str(dir_name)


def run_single_experiment(config_path, experiment_script="icl_delphi_tests.py", timeout=3600):
    """Run a single experiment with the given configuration."""
    
    try:
        start_time = time.time()
        
        # Set environment for unbuffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        # Run with direct output to terminal
        result = subprocess.run(
            [sys.executable, '-u', experiment_script, config_path],
            env=env,
            timeout=timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Experiment completed successfully in {duration:.1f}s")
            return True, duration
        else:
            print(f"❌ Experiment failed with return code {result.returncode}")
            return False, duration
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment timed out after {timeout}s")
        return False, timeout
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False, 0


def run_parameter_sweep(
    base_config_path,
    sweep_config=None,
    experiment_script="icl_delphi_tests.py",
    dry_run=False,
    timeout=3600,
    max_parallel=1,
    reuse_initial_dir=None
):
    """
    Run a parameter sweep over specified parameters.
    
    Args:
        base_config_path: Path to base configuration file
        sweep_config: Dictionary with sweep configuration or None to use hardcoded
        experiment_script: Script to run for each experiment
        dry_run: If True, only show what would be run
        timeout: Timeout for each experiment in seconds
        max_parallel: Maximum number of parallel experiments (default: 1 for sequential)
    """
    
    # Load base configuration
    base_config = load_config(base_config_path)

    # Generate sweep ID based on timestamp
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create sweep results directory
    results_dir = Path(f"results/{sweep_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sweep configuration
    sweep_config_path = results_dir / "sweep_config.json"
    with open(sweep_config_path, 'w') as f:
        json.dump(sweep_config, f, indent=2)
    
    print(f"🎯 Parameter Sweep Experiment")
    print(f"📁 Base config: {base_config_path}")
    print(f"🆔 Sweep ID: {sweep_id}")
    print(f"📊 Results directory: {results_dir}")
    print(f"🔧 Max parallel experiments: {max_parallel}")
    
    if dry_run:
        print("🔍 DRY RUN MODE - No experiments will actually be executed")
    
    # Results tracking
    all_results = []
    
    # Prepare all experiment configurations
    experiment_configs = []
    
    # Determine sweep mode (default to sequential if not specified)
    sweep_mode = sweep_config.get('mode', 'sequential')
    
    print(f"\n📊 Sweep mode: {sweep_mode}")
    
    if sweep_mode == 'combinatorial':
        # Combinatorial mode - test all combinations of parameters
        param_names = []
        param_paths = []
        param_value_lists = []
        
        for param_config in sweep_config['parameters']:
            param_names.append(param_config['name'])
            param_paths.append(param_config['path'])
            param_value_lists.append(param_config['values'])
            
            print(f"\n📈 Parameter: {param_config['name']}")
            print(f"📍 Path: {param_config['path']}")
            print(f"📊 Values: {param_config['values']}")
        
        # Generate all combinations
        all_combinations = list(product(*param_value_lists))
        print(f"\n🔢 Total combinations: {len(all_combinations)}")
        
        for combo_idx, value_combo in enumerate(all_combinations, 1):
            # Create modified configuration
            modified_config = deepcopy(base_config)
            
            # Create combo description for naming
            combo_parts = []
            for param_name, param_path, value in zip(param_names, param_paths, value_combo):
                update_nested_dict(modified_config, param_path, value)
                combo_parts.append(f"{param_name}_{str(value).replace('.', '_').replace('/', '_')}")
            
            combo_str = "_".join(combo_parts)
            
            # Update output directories
            output_dir = str(results_dir / f"results_{combo_str}")
            modified_config['experiment']['output_dir'] = output_dir
            
            # Also update initial forecasts dir if it exists
            if 'initial_forecasts_dir' in modified_config.get('experiment', {}):
                modified_config['experiment']['initial_forecasts_dir'] = f"{output_dir}_initial"
            
            # Add reuse initial forecasts configuration if specified
            if reuse_initial_dir:
                if 'experiment' not in modified_config:
                    modified_config['experiment'] = {}
                modified_config['experiment']['reuse_initial_forecasts'] = {
                    'enabled': True,
                    'source_dir': reuse_initial_dir if reuse_initial_dir != '--reuse-initial' else 'auto'
                }
                # Also update sampling method to use existing forecasts
                if 'data' in modified_config and 'sampling' in modified_config['data']:
                    modified_config['data']['sampling']['method'] = 'from_initial_forecasts'
            
            # Save modified configuration
            config_filename = f"config_{combo_str}.yml"
            temp_config_path = results_dir / config_filename
            save_config(modified_config, temp_config_path)
            
            experiment_configs.append({
                'param_name': " + ".join(param_names),
                'param_value': " + ".join(str(v) for v in value_combo),
                'config_path': temp_config_path,
                'output_dir': output_dir,
                'index': combo_idx,
                'total': len(all_combinations)
            })
    else:
        # Sequential mode - sweep each parameter independently
        for param_config in sweep_config['parameters']:
            param_name = param_config['name']
            param_path = param_config['path']
            param_values = param_config['values']
            
            print(f"\n{'='*60}")
            print(f"📈 Sweeping parameter: {param_name}")
            print(f"📍 Path: {param_path}")
            print(f"📊 Values: {param_values}")
            print(f"{'='*60}")
            
            for i, value in enumerate(param_values, 1):
                # Create modified configuration
                modified_config = deepcopy(base_config)
                
                # Update parameter value
                update_nested_dict(modified_config, param_path, value)
                
                # Update output directories
                output_dir = create_sweep_output_dir(results_dir, param_name, value)
                modified_config['experiment']['output_dir'] = output_dir
                
                # Also update initial forecasts dir if it exists
                if 'initial_forecasts_dir' in modified_config.get('experiment', {}):
                    modified_config['experiment']['initial_forecasts_dir'] = f"{output_dir}_initial"
                
                # Save modified configuration
                config_filename = f"config_{param_name}_{str(value).replace('.', '_')}.yml"
                temp_config_path = results_dir / config_filename
                save_config(modified_config, temp_config_path)
                
                experiment_configs.append({
                    'param_name': param_name,
                    'param_value': value,
                    'config_path': temp_config_path,
                    'output_dir': output_dir,
                    'index': i,
                    'total': len(param_values)
                })
    
    print(f"\n📋 Total experiments to run: {len(experiment_configs)}")
    
    if dry_run:
        # Dry run - just show what would be executed
        for exp_config in experiment_configs:
            print(f"\n🔍 DRY RUN: Would execute:")
            print(f"   Parameter: {exp_config['param_name']} = {exp_config['param_value']}")
            print(f"   Command: python {experiment_script} {exp_config['config_path']}")
            print(f"   Output: {exp_config['output_dir']}")
            
            result = {
                'parameter': exp_config['param_name'],
                'value': exp_config['param_value'],
                'config_file': str(exp_config['config_path']),
                'output_dir': exp_config['output_dir'],
                'success': None,
                'duration': None
            }
            all_results.append(result)
    else:
        # Run experiments (either parallel or sequential)
        if max_parallel > 1:
            print(f"\n🚀 Running experiments in parallel (max {max_parallel} concurrent)...")
            
            with ProcessPoolExecutor(max_workers=max_parallel) as executor:
                # Submit all experiments
                future_to_config = {}
                for exp_config in experiment_configs:
                    future = executor.submit(
                        run_single_experiment,
                        exp_config['config_path'],
                        experiment_script,
                        timeout
                    )
                    future_to_config[future] = exp_config
                
                # Process completed experiments
                completed = 0
                for future in as_completed(future_to_config):
                    exp_config = future_to_config[future]
                    completed += 1
                    
                    try:
                        success, duration = future.result()
                        status = "✅" if success else "❌"
                        print(f"\n{status} [{completed}/{len(experiment_configs)}] {exp_config['param_name']}={exp_config['param_value']} - {duration:.1f}s")
                    except Exception as e:
                        success, duration = False, 0
                        print(f"\n❌ [{completed}/{len(experiment_configs)}] {exp_config['param_name']}={exp_config['param_value']} - Error: {e}")
                    
                    result = {
                        'parameter': exp_config['param_name'],
                        'value': exp_config['param_value'],
                        'config_file': str(exp_config['config_path']),
                        'output_dir': exp_config['output_dir'],
                        'success': success,
                        'duration': duration
                    }
                    all_results.append(result)
        else:
            print(f"\n🚀 Running experiments sequentially...")
            
            for i, exp_config in enumerate(experiment_configs, 1):
                print(f"\n🔄 Experiment {i}/{len(experiment_configs)}: {exp_config['param_name']} = {exp_config['param_value']}")
                print(f"📄 Config: {exp_config['config_path']}")
                print(f"📂 Output: {exp_config['output_dir']}")
                
                success, duration = run_single_experiment(
                    exp_config['config_path'],
                    experiment_script,
                    timeout
                )
                
                result = {
                    'parameter': exp_config['param_name'],
                    'value': exp_config['param_value'],
                    'config_file': str(exp_config['config_path']),
                    'output_dir': exp_config['output_dir'],
                    'success': success,
                    'duration': duration
                }
                all_results.append(result)
                
                # Add delay between experiments in sequential mode
                if i < len(experiment_configs):
                    print("⏳ Waiting 2 seconds before next experiment...")
                    time.sleep(2)
    
    # Save final results summary
    results_summary_path = results_dir / "results_summary.json"
    with open(results_summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 PARAMETER SWEEP COMPLETE")
    print(f"{'='*60}")
    
    if not dry_run:
        successful = sum(1 for r in all_results if r['success'])
        failed = sum(1 for r in all_results if r['success'] is False)
        
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        # Print details for each parameter
        for param_config in sweep_config['parameters']:
            param_name = param_config['name']
            param_results = [r for r in all_results if r['parameter'] == param_name]
            
            print(f"\n📈 {param_name} sweep results:")
            for r in param_results:
                status = "✅" if r['success'] else "❌" if r['success'] is False else "⏭️"
                duration_str = f"{r['duration']:.1f}s" if r['duration'] else "N/A"
                print(f"  {status} {param_name}={r['value']}: {duration_str}")
    
    print(f"\n📁 All results saved to: {results_summary_path}")
    
    return all_results


def load_sweep_from_yaml(yaml_path):
    """Load sweep configuration from a YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep experiments for Delphi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default hardcoded sweep (sequential)
  python run_parameter_sweep.py configs/experts_comparison_gpt_oss_120b.yml
  
  # Run experiments in parallel (4 concurrent)
  python run_parameter_sweep.py configs/base_config.yml --parallel 4
  
  # Use sweep configuration from YAML file with parallelism
  python run_parameter_sweep.py configs/base_config.yml --sweep sweep_config.yml --parallel 8
  
  # Dry run to see what would be executed
  python run_parameter_sweep.py configs/base_config.yml --dry-run
  
  # Use different experiment script
  python run_parameter_sweep.py configs/base_config.yml --script my_script.py

Sweep Configuration YAML Format:
  parameters:
    - name: temperature
      path: model.expert.temperature
      values: [0.1, 0.3, 0.5, 0.7, 0.9]
    - name: n_experts
      path: delphi.n_experts  
      values: [3, 5, 7]
        """
    )
    
    parser.add_argument(
        "base_config",
        help="Path to base configuration file"
    )
    
    parser.add_argument(
        "--sweep",
        help="Path to YAML file with sweep configuration"
    )
    
    parser.add_argument(
        "--script",
        default="icl_delphi_tests.py",
        help="Experiment script to use (default: icl_delphi_tests.py)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing experiments"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout for each experiment in seconds (default: 3600)"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum number of parallel experiments (default: 1 for sequential)"
    )
    
    parser.add_argument(
        "--reuse-initial",
        help="Path to directory with initial forecasts to reuse (e.g., results/20250819_200714/results_n_experts_4_seed_42_initial)"
    )
    
    args = parser.parse_args()
    
    # Verify base config exists
    if not Path(args.base_config).exists():
        print(f"❌ Base configuration file not found: {args.base_config}")
        sys.exit(1)
    
    # Verify experiment script exists
    if not Path(args.script).exists():
        print(f"❌ Experiment script not found: {args.script}")
        sys.exit(1)
    
    # Load sweep configuration if provided
    sweep_config = None
    if args.sweep:
        if not Path(args.sweep).exists():
            print(f"❌ Sweep configuration file not found: {args.sweep}")
            sys.exit(1)
        sweep_config = load_sweep_from_yaml(args.sweep)
        print(f"📄 Loaded sweep configuration from: {args.sweep}")
    
    # Run parameter sweep
    results = run_parameter_sweep(
        base_config_path=args.base_config,
        sweep_config=sweep_config,
        experiment_script=args.script,
        dry_run=args.dry_run,
        timeout=args.timeout,
        max_parallel=args.parallel,
        reuse_initial_dir=getattr(args, 'reuse_initial', None)
    )
    
    # Exit with success if all experiments succeeded (or dry run)
    if args.dry_run:
        sys.exit(0)
    else:
        all_success = all(r['success'] for r in results if r['success'] is not None)
        sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()