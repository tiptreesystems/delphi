#!/usr/bin/env python3
"""
Resume failed parameter sweep experiments.

This script identifies failed experiments in a sweep directory and re-runs them,
with proper error logging to avoid littering logs everywhere.
"""

import subprocess
import sys
import os
import time
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def load_results(sweep_dir):
    """Load existing sweep results."""
    sweep_dir = Path(sweep_dir)

    # Load results summary
    results_summary_path = sweep_dir / "results_summary.json"
    if not results_summary_path.exists():
        print(f"Error: No results_summary.json found in {sweep_dir}")
        return None

    with open(results_summary_path, "r") as f:
        results = json.load(f)

    return results


def run_single_experiment_with_logging(
    config_path, experiment_script="icl_delphi_tests.py", timeout=36000, log_dir=None
):
    """Run a single experiment with proper error logging."""

    try:
        start_time = time.time()

        # Create log file path if log_dir provided
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            config_name = Path(config_path).stem
            log_file = log_dir / f"{config_name}.log"
            error_file = log_dir / f"{config_name}_error.log"
        else:
            log_file = None
            error_file = None

        # Set environment for unbuffered output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Run with output capture for logging
        if log_file:
            with open(log_file, "w") as log_f, open(error_file, "w") as err_f:
                result = subprocess.run(
                    [sys.executable, "-u", experiment_script, config_path],
                    env=env,
                    timeout=timeout,
                    stdout=log_f,
                    stderr=err_f,
                )
        else:
            # No logging, direct output
            result = subprocess.run(
                [sys.executable, "-u", experiment_script, config_path],
                env=env,
                timeout=timeout,
            )

        end_time = time.time()
        duration = end_time - start_time

        success = result.returncode == 0

        if success:
            print(f"✅ Experiment completed successfully in {duration:.1f}s")
            # Remove error file if experiment succeeded and it's empty
            if error_file and error_file.exists():
                if error_file.stat().st_size == 0:
                    error_file.unlink()
        else:
            print(f"❌ Experiment failed with return code {result.returncode}")
            if error_file:
                print(f"   Error log: {error_file}")

        return success, duration, log_file, error_file

    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment timed out after {timeout}s")
        return False, timeout, log_file, error_file
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False, 0, log_file, error_file


def resume_failed_experiments(
    sweep_dir,
    experiment_script="icl_delphi_tests.py",
    dry_run=False,
    timeout=3600,
    max_parallel=1,
    force_rerun_all=False,
):
    """Resume failed experiments from a sweep directory."""

    sweep_dir = Path(sweep_dir)

    # Load existing results
    print(f"Loading sweep results from {sweep_dir}...")
    results = load_results(sweep_dir)

    if not results:
        return False

    # Identify failed experiments
    failed_experiments = []
    successful_experiments = []

    for result in results:
        if not result["success"] or force_rerun_all:
            failed_experiments.append(result)
        else:
            successful_experiments.append(result)

    print(f"\n📊 Sweep Status:")
    print(f"✅ Successful experiments: {len(successful_experiments)}")
    print(f"❌ Failed experiments: {len(failed_experiments)}")
    print(f"📋 Total experiments: {len(results)}")

    if not failed_experiments:
        print("\n🎉 All experiments have already completed successfully!")
        return True

    if force_rerun_all:
        print(f"🔄 Force re-running ALL {len(results)} experiments")
    else:
        print(f"🔄 Re-running {len(failed_experiments)} failed experiments")

    # Create logs directory
    logs_dir = sweep_dir / "logs"

    print(f"\n📁 Experiment logs will be saved to: {logs_dir}")

    if dry_run:
        print("\n🔍 DRY RUN MODE - Would re-run these experiments:")
        for exp in failed_experiments:
            config_name = Path(exp["config_file"]).name
            param_desc = f"{exp['parameter']} = {exp['value']}"
            print(f"  - {config_name}: {param_desc}")
        return True

    # Re-run failed experiments
    print(f"\n🚀 Starting experiment re-runs (max {max_parallel} parallel)...")

    updated_results = successful_experiments.copy()  # Keep successful ones

    if max_parallel > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            future_to_exp = {}

            for exp in failed_experiments:
                future = executor.submit(
                    run_single_experiment_with_logging,
                    exp["config_file"],
                    experiment_script,
                    timeout,
                    logs_dir,
                )
                future_to_exp[future] = exp

            completed = 0
            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                completed += 1

                config_name = Path(exp["config_file"]).stem
                param_desc = f"{exp['parameter']} = {exp['value']}"

                print(f"\n[{completed}/{len(failed_experiments)}] {config_name}")
                print(f"Parameters: {param_desc}")

                try:
                    success, duration, log_file, error_file = future.result()
                    status = "✅" if success else "❌"
                    print(f"{status} Duration: {duration:.1f}s")

                    if log_file:
                        print(f"📄 Log: {log_file}")
                    if error_file and error_file.exists():
                        print(f"🚨 Error log: {error_file}")

                except Exception as e:
                    success, duration = False, 0
                    print(f"❌ Exception: {e}")

                # Update result
                updated_exp = exp.copy()
                updated_exp["success"] = success
                updated_exp["duration"] = duration
                updated_exp["resumed_at"] = datetime.now().isoformat()
                updated_results.append(updated_exp)
    else:
        # Sequential execution
        for i, exp in enumerate(failed_experiments, 1):
            config_name = Path(exp["config_file"]).stem
            param_desc = f"{exp['parameter']} = {exp['value']}"

            print(f"\n[{i}/{len(failed_experiments)}] {config_name}")
            print(f"Parameters: {param_desc}")
            print(f"Config: {exp['config_file']}")

            success, duration, log_file, error_file = (
                run_single_experiment_with_logging(
                    exp["config_file"], experiment_script, timeout, logs_dir
                )
            )

            if log_file:
                print(f"📄 Log: {log_file}")
            if error_file and error_file.exists():
                print(f"🚨 Error log: {error_file}")

            # Update result
            updated_exp = exp.copy()
            updated_exp["success"] = success
            updated_exp["duration"] = duration
            updated_exp["resumed_at"] = datetime.now().isoformat()
            updated_results.append(updated_exp)

            # Add delay between experiments in sequential mode
            if i < len(failed_experiments):
                print("⏳ Waiting 2 seconds before next experiment...")
                time.sleep(2)

    # Save updated results
    results_summary_path = sweep_dir / "results_summary.json"
    backup_path = (
        sweep_dir
        / f"results_summary_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Backup original results
    if results_summary_path.exists():
        import shutil

        shutil.copy2(results_summary_path, backup_path)
        print(f"\n📋 Original results backed up to: {backup_path}")

    # Save updated results
    with open(results_summary_path, "w") as f:
        json.dump(updated_results, f, indent=2)

    # Final summary
    new_successful = sum(1 for r in updated_results if r["success"])
    new_failed = len(updated_results) - new_successful

    print(f"\n{'=' * 60}")
    print(f"📊 RESUME SWEEP COMPLETE")
    print(f"{'=' * 60}")
    print(f"✅ Total successful: {new_successful}")
    print(f"❌ Total failed: {new_failed}")
    print(f"📈 Success rate: {new_successful / len(updated_results) * 100:.1f}%")

    if new_failed == 0:
        print(f"🎉 All experiments completed successfully!")
    else:
        print(f"⚠️  {new_failed} experiments still failed")
        print(f"📁 Check error logs in: {logs_dir}")

    print(f"📁 Updated results saved to: {results_summary_path}")

    return new_failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Resume failed parameter sweep experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resume failed experiments
  python resume_sweep.py results/20250818_225639
  
  # Resume with parallel execution
  python resume_sweep.py results/20250818_225639 --parallel 4
  
  # Dry run to see what would be re-run
  python resume_sweep.py results/20250818_225639 --dry-run
  
  # Force re-run ALL experiments (including successful ones)
  python resume_sweep.py results/20250818_225639 --force
        """,
    )

    parser.add_argument("sweep_dir", help="Path to sweep results directory")

    parser.add_argument(
        "--script",
        default="icl_delphi_tests.py",
        help="Experiment script to use (default: icl_delphi_tests.py)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be re-run without executing experiments",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout for each experiment in seconds (default: 3600)",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum number of parallel experiments (default: 1 for sequential)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run ALL experiments, including successful ones",
    )

    args = parser.parse_args()

    # Verify sweep directory exists
    if not Path(args.sweep_dir).exists():
        print(f"❌ Sweep directory not found: {args.sweep_dir}")
        sys.exit(1)

    # Verify experiment script exists
    if not Path(args.script).exists():
        print(f"❌ Experiment script not found: {args.script}")
        sys.exit(1)

    # Resume failed experiments
    success = resume_failed_experiments(
        sweep_dir=args.sweep_dir,
        experiment_script=args.script,
        dry_run=args.dry_run,
        timeout=args.timeout,
        max_parallel=args.parallel,
        force_rerun_all=args.force,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
