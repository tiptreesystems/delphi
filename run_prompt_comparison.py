#!/usr/bin/env python3
"""
Prompt Technique Comparison Script

This script runs a batch of Delphi experiments comparing different prompt techniques.
All experiments use the same model (DeepSeek R1) for fair comparison.

Prompt techniques being tested:
1. Baseline (v1) - Original prompt
2. Frequency-Based - "Out of 100 similar cases..." approach
3. Short Focused - Concise, direct analysis
4. Deep Analytical - Comprehensive multi-angle analysis  
5. High Variance - Contrarian, distinctive positions
6. Opinionated - Strong, definitive stances
7. Base Rate Focused - Historical frequencies + adjustments
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration for all prompt techniques in the comparison
PROMPT_COMPARISON_CONFIGS = [
    {
        "name": "Baseline (v1)",
        "config_file": "configs/prompt_comparison_baseline.yml",
        "description": "Original prompt technique"
    },
    {
        "name": "Baseline (v1) with Examples",
        "config_file": "configs/prompt_comparison_baseline_with_examples.yml",
        "description": "Original prompt technique with examples"
    },
    {
        "name": "Frequency-Based",
        "config_file": "configs/prompt_comparison_frequency_based.yml",
        "description": "Out of 100 similar cases approach"
    },
    {
        "name": "Short Focused",
        "config_file": "configs/prompt_comparison_short_focused.yml",
        "description": "Concise, direct analysis"
    },
    {
        "name": "Deep Analytical",
        "config_file": "configs/prompt_comparison_deep_analytical.yml", 
        "description": "Comprehensive multi-angle analysis"
    },
    {
        "name": "High Variance",
        "config_file": "configs/prompt_comparison_high_variance.yml",
        "description": "Contrarian, distinctive positions"
    },
    {
        "name": "Opinionated",
        "config_file": "configs/prompt_comparison_opinionated.yml",
        "description": "Strong, definitive stances"
    },
    {
        "name": "Base Rate Focused",
        "config_file": "configs/prompt_comparison_base_rate.yml",
        "description": "Historical frequencies + adjustments"
    }
]

def check_api_keys():
    """Check that required API keys are set."""
    required_keys = {
        "GROQ_API_KEY": "Groq API"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{key} (for {description})")
    
    if missing_keys:
        print("❌ Missing required API keys:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\\nPlease set these environment variables before running the experiments.")
        return False
    
    print("✅ All required API keys are set.")
    return True

def check_config_files():
    """Check that all config files exist."""
    missing_files = []
    for config in PROMPT_COMPARISON_CONFIGS:
        config_path = Path(config["config_file"])
        if not config_path.exists():
            missing_files.append(config["config_file"])
    
    if missing_files:
        print("❌ Missing configuration files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✅ All configuration files found.")
    return True

def run_experiment(config, experiment_script="icl_delphi_tests.py", dry_run=False):
    """Run a single experiment with the given configuration."""
    config_file = config["config_file"]
    prompt_name = config["name"]
    description = config["description"]
    
    print(f"\\n{'='*60}")
    print(f"🚀 Running experiment: {prompt_name}")
    print(f"📄 Config: {config_file}")
    print(f"📝 Description: {description}")
    print(f"{'='*60}")
    
    if dry_run:
        print("🔍 DRY RUN: Would execute:")
        print(f"   python {experiment_script} {config_file}")
        return True
    
    try:
        start_time = time.time()
        
        # Run the experiment with direct output (no capture)
        print("🔄 Starting experiment...")
        print("=" * 60)
        
        # Set environment for unbuffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        # Run with direct output to terminal
        result = subprocess.run(
            [sys.executable, '-u', experiment_script, config_file],
            env=env,
            timeout=3600  # 1 hour timeout
        )
        
        print("=" * 60)
        return_code = result.returncode
        
        end_time = time.time()
        duration = end_time - start_time
        
        if return_code == 0:
            print(f"✅ Experiment completed successfully in {duration:.1f}s")
            return True
        else:
            print(f"❌ Experiment failed with return code {return_code}")
            return False
            
    except subprocess.TimeoutExpired:
        print("=" * 60)
        print(f"⏰ Experiment timed out after 1 hour")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False

def run_all_experiments(experiment_script="icl_delphi_tests.py", dry_run=False, selected_prompts=None):
    """Run all prompt comparison experiments."""
    
    # Filter configs if specific prompts are selected
    configs_to_run = PROMPT_COMPARISON_CONFIGS
    if selected_prompts:
        configs_to_run = [
            config for config in PROMPT_COMPARISON_CONFIGS 
            if any(prompt.lower() in config["name"].lower() for prompt in selected_prompts)
        ]
    
    print(f"\\n🎯 Prompt Technique Comparison Study")
    print(f"📊 Running {len(configs_to_run)} experiments")
    if dry_run:
        print("🔍 DRY RUN MODE - No experiments will actually be executed")
    print()
    
    # Pre-flight checks
    if not dry_run:
        if not check_api_keys():
            return False
        
    if not check_config_files():
        return False
    
    # Track results
    successful_experiments = []
    failed_experiments = []
    start_time = time.time()
    
    # Run experiments
    for i, config in enumerate(configs_to_run, 1):
        print(f"\\n📍 Progress: {i}/{len(configs_to_run)}")
        
        success = run_experiment(config, experiment_script, dry_run)
        
        if success:
            successful_experiments.append(config["name"])
        else:
            failed_experiments.append(config["name"])
                
    # Final report
    total_time = time.time() - start_time
    print(f"\\n{'='*60}")
    print(f"📊 PROMPT COMPARISON STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"✅ Successful: {len(successful_experiments)}")
    print(f"❌ Failed: {len(failed_experiments)}")
    
    if successful_experiments:
        print(f"\\n✅ Successful experiments:")
        for name in successful_experiments:
            print(f"  - {name}")
    
    if failed_experiments:
        print(f"\\n❌ Failed experiments:")
        for name in failed_experiments:
            print(f"  - {name}")
    
    success_rate = len(successful_experiments) / len(configs_to_run) * 100
    print(f"\\n📈 Success rate: {success_rate:.1f}%")
    
    if successful_experiments:
        print(f"\\n🎉 Ready for analysis! Run:")
        print(f"   python analyze_prompt_comparison.py")
    
    return len(failed_experiments) == 0

def main():
    parser = argparse.ArgumentParser(
        description="Run prompt technique comparison experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_prompt_comparison.py                    # Run all experiments
  python run_prompt_comparison.py --dry-run          # Show what would be run
  python run_prompt_comparison.py --prompts frequency baseline  # Run specific prompts
  python run_prompt_comparison.py --script my_script.py  # Use different experiment script
        """
    )
    
    parser.add_argument(
        "--script", 
        default="icl_delphi_tests.py",
        help="Experiment script to use (default: icl_delphi_tests.py)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Show what would be run without actually executing experiments"
    )
    
    parser.add_argument(
        "--prompts",
        nargs="*",
        help="Run only prompts containing these keywords (e.g., --prompts frequency baseline)"
    )
    
    args = parser.parse_args()
    
    # Verify experiment script exists
    if not Path(args.script).exists():
        print(f"❌ Experiment script not found: {args.script}")
        sys.exit(1)
    
    # Run experiments
    success = run_all_experiments(
        experiment_script=args.script,
        dry_run=args.dry_run,
        selected_prompts=args.prompts
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()