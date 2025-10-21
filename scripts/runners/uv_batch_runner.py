#!/usr/bin/env python3
"""
Queue runner for sequential uv runs.

Edit the COMMANDS list below to include the exact shell commands you want to run.
Each command is executed only after the previous one completes.

Examples (uncomment and customize):
COMMANDS = [
      "uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evolution_evaluation/delphi_eval_gpt_oss_120b_3_experts_3_examples.yml --seed 5",
      "uv run analyze/aggregate_brier_rounds_across_seeds.py --config configs/evolution_evaluation/delphi_eval_gpt_oss_120b_5_experts_5_examples.yml --mode seeds",
  ]
"""

from __future__ import annotations
import subprocess
import sys
import time
import os


# ── EDIT ME: put your uv commands here as full shell strings ──────────────────
COMMANDS: list[str] = [
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_gpt_4o/delphi_eval_gpt_4o_3_experts_3_examples_no_system_prompt.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_gpt_4o/delphi_eval_gpt_4o_3_experts_3_examples_no_system_prompt.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_gpt_4o/delphi_eval_gpt_4o_3_experts_3_examples_no_system_prompt.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_frequency_prompt_no_examples.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_frequency_prompt_no_examples.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_frequency_prompt_no_examples.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_frequency_prompt.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_frequency_prompt.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_frequency_prompt.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_3_experts_3_examples_no_system_prompt.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_3_experts_3_examples_no_system_prompt.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_3_experts_3_examples_no_system_prompt.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_gpt_4o/delphi_eval_gpt_4o_5_experts_3_examples_no_system_prompt.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_gpt_4o/delphi_eval_gpt_4o_5_experts_3_examples_no_system_prompt.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_gpt_4o/delphi_eval_gpt_4o_5_experts_3_examples_no_system_prompt.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_5_experts_3_examples_no_system_prompt.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_5_experts_3_examples_no_system_prompt.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_5_experts_3_examples_no_system_prompt.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_o3/delphi_eval_o3_3_experts_3_examples_no_system_prompt.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_o3/delphi_eval_o3_3_experts_3_examples_no_system_prompt.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_o3/delphi_eval_o3_3_experts_3_examples_no_system_prompt.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_o3/delphi_eval_o3_3_experts_no_examples_no_system_prompt.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_o3/delphi_eval_o3_3_experts_no_examples_no_system_prompt.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_o3/delphi_eval_o3_3_experts_no_examples_no_system_prompt.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_no_system_prompt_o3_mediator.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_no_system_prompt_o3_mediator.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_no_system_prompt_o3_mediator.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_no_system_prompt.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_no_system_prompt.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_no_system_prompt.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_no_system_prompt_identical_expert.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_no_system_prompt_identical_expert.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_no_system_prompt_identical_expert.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_3_experts_no_examples_no_system_prompt.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_3_experts_no_examples_no_system_prompt.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_3_experts_no_examples_no_system_prompt.yml --seed 3',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_3_experts_3_examples_evolved_mediator_prompt.yml --seed 1',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_3_experts_3_examples_evolved_mediator_prompt.yml --seed 2',
    # 'uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_20b/delphi_eval_gpt_oss_20b_3_experts_3_examples_evolved_mediator_prompt.yml --seed 3',
    "uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_evolved_mediator_prompt.yml --seed 1",
    "uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_evolved_mediator_prompt.yml --seed 2",
    "uv run /home/williaar/projects/delphi/runners/icl_delphi_tests.py /home/williaar/projects/delphi/configs/evaluation_real_oss_120b/delphi_eval_gpt_oss_120b_3_experts_3_examples_evolved_mediator_prompt.yml --seed 3",
]

missing = []
for cmd in COMMANDS:
    for part in cmd.split():
        if part.endswith(".yml") or part.endswith(".yaml"):
            if not os.path.isfile(part):
                missing.append(part)
if missing:
    print("Warning: The following YAML files do not exist:")
    for f in set(missing):
        print("  ", f)
    print("Proceeding anyway...\n")

# Stop the queue on the first non‑zero exit code
STOP_ON_ERROR = True


def run_queue(cmds: list[str]) -> int:
    if not cmds:
        print("No commands specified in COMMANDS list.")
        return 0

    for i, cmd in enumerate(cmds, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(cmds)}] Running: {cmd}")
        print("=" * 80)
        t0 = time.time()
        # Use shell=True so the string runs exactly as you wrote it
        proc = subprocess.run(cmd, shell=True)
        dt = time.time() - t0
        print(f"-- Exit code: {proc.returncode} | Duration: {dt:.1f}s")
        if proc.returncode != 0 and STOP_ON_ERROR:
            print("Aborting queue due to failure.")
            return proc.returncode
    print("\nAll commands completed.")
    return 0


def main() -> None:
    rc = run_queue(COMMANDS)
    sys.exit(rc)


if __name__ == "__main__":
    main()
