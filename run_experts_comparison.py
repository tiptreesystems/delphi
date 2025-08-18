#!/usr/bin/env python3
"""
Expert Model Comparison Script

This script runs a batch of Delphi experiments comparing different models for experts.
It launches experiments for all 9 models in the comparison study with consistent parameters.

Models being compared:
1. meta-llama/llama-4-maverick-17b-128e-instruct (Groq)
2. openai/gpt-oss-120b (Groq) 
3. openai/gpt-oss-20b (Groq)
4. qwen/qwen3-32b (Groq)
5. deepseek-r1-distill-llama-70b (Groq)
6. o3-2025-04-16 (OpenAI)
7. gpt-5-2025-08-07 (OpenAI)
8. o1-2024-12-17 (OpenAI)
9. claude-3-7-sonnet-20250219 (Anthropic)
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Configuration for all models in the comparison
EXPERT_COMPARISON_CONFIGS = [
    {
        "name": "Llama 4 Maverick 17B",
        "config_file": "configs/experts_comparison_llama_maverick.yml",
        "provider": "groq"
    },
    {
        "name": "GPT-OSS 120B",
        "config_file": "configs/experts_comparison_gpt_oss_120b.yml", 
        "provider": "groq"
    },
    {
        "name": "GPT-OSS 20B",
        "config_file": "configs/experts_comparison_gpt_oss_20b.yml",
        "provider": "groq"
    },
    {
        "name": "Qwen3 32B",
        "config_file": "configs/experts_comparison_qwen3_32b.yml",
        "provider": "groq"
    },
    {
        "name": "DeepSeek R1 Distill 70B",
        "config_file": "configs/experts_comparison_deepseek_r1.yml",
        "provider": "groq"
    },
        {
        "name": "Claude 3.7 Sonnet",
        "config_file": "configs/experts_comparison_claude37_sonnet.yml",
        "provider": "anthropic"
    },
    {
        "name": "GPT-4o",
        "config_file": "configs/experts_comparison_gpt_4o.yml",
        "provider": "openai"
    },
    {
        "name": "O3 (2025-04-16)",
        "config_file": "configs/experts_comparison_o3.yml",
        "provider": "openai"
    },
    {
        "name": "GPT-5 (2025-08-07)", 
        "config_file": "configs/experts_comparison_gpt5.yml",
        "provider": "openai"
    },
    {
        "name": "O1 (2024-12-17)",
        "config_file": "configs/experts_comparison_o1.yml",
        "provider": "openai"
    },
]

def check_api_keys():
    """Check that required API keys are set."""
    required_keys = {
        "GROQ_API_KEY": "Groq API",
        "OPENAI_API_KEY": "OpenAI API", 
        "ANTHROPIC_API_KEY": "Anthropic API"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{key} (for {description})")
    
    if missing_keys:
        print("❌ Missing required API keys:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPlease set these environment variables before running the experiments.")
        return False
    
    print("✅ All required API keys are set.")
    return True

def check_config_files():
    """Check that all config files exist."""
    missing_files = []
    for config in EXPERT_COMPARISON_CONFIGS:
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
    model_name = config["name"]
    
    print(f"\n{'='*60}")
    print(f"🚀 Running experiment: {model_name}")
    print(f"📄 Config: {config_file}")
    print(f"🔗 Provider: {config['provider']}")
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

def run_all_experiments(experiment_script="icl_delphi_tests.py", dry_run=False, selected_models=None):
    """Run all expert comparison experiments."""
    
    # Filter configs if specific models are selected
    configs_to_run = EXPERT_COMPARISON_CONFIGS
    if selected_models:
        configs_to_run = [
            config for config in EXPERT_COMPARISON_CONFIGS 
            if any(model.lower() in config["name"].lower() for model in selected_models)
        ]
    
    print(f"\n🎯 Expert Model Comparison Study")
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
        print(f"\n📍 Progress: {i}/{len(configs_to_run)}")
        
        success = run_experiment(config, experiment_script, dry_run)
        
        if success:
            successful_experiments.append(config["name"])
        else:
            failed_experiments.append(config["name"])
            
        # Add delay between experiments to be respectful to APIs
        if not dry_run and i < len(configs_to_run):
            print("⏳ Waiting 1 seconds before next experiment...")
            time.sleep(1)
    
    # Final report
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"📊 EXPERT COMPARISON STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"✅ Successful: {len(successful_experiments)}")
    print(f"❌ Failed: {len(failed_experiments)}")
    
    if successful_experiments:
        print(f"\n✅ Successful experiments:")
        for name in successful_experiments:
            print(f"  - {name}")
    
    if failed_experiments:
        print(f"\n❌ Failed experiments:")
        for name in failed_experiments:
            print(f"  - {name}")
    
    success_rate = len(successful_experiments) / len(configs_to_run) * 100
    print(f"\n📈 Success rate: {success_rate:.1f}%")
    
    return len(failed_experiments) == 0

def main():
    parser = argparse.ArgumentParser(
        description="Run expert model comparison experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experts_comparison.py                    # Run all experiments
  python run_experts_comparison.py --dry-run          # Show what would be run
  python run_experts_comparison.py --models gpt o3    # Run only GPT and O3 models
  python run_experts_comparison.py --script my_script.py  # Use different experiment script
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
        "--models",
        nargs="*",
        help="Run only models containing these keywords (e.g., --models gpt claude)"
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
        selected_models=args.models
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()