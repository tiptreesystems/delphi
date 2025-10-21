#!/usr/bin/env python3
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Dict, List, Tuple

# Patterns to parse the invoked script's stdout
RE_FILE = re.compile(r"^file:\s*(.+)$")
RE_EXP = re.compile(r"^experts_per_round:\s*(\d+)$")
RE_ROUND = re.compile(
    r"^round\s+(.*?):\s+finals=(\d+)\/(\d+)\s+success_rate=([0-9.]+)$"
)
RE_FINALS = re.compile(r"^finals_total:\s*(\d+)$")
RE_OVERALL = re.compile(r"^overall_success_rate:\s*([0-9.]+)$")


def run_counter(counter_script: Path, json_file: Path) -> Tuple[Path, str]:
    """Invoke the counting script for a single JSON file and return its stdout."""
    proc = subprocess.run(
        [sys.executable, str(counter_script), str(json_file)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # Bubble up the error loudly (the child script is intentionally strict)
        raise RuntimeError(
            f"Child script failed for {json_file} (exit {proc.returncode}).\nSTDERR:\n{proc.stderr}"
        )
    return json_file, proc.stdout


def parse_output(stdout: str) -> Dict:
    """
    Parse the child script's stdout.
    Returns a dict:
      {
        "file": str,
        "experts_per_round": int,
        "rounds": List[{"label": str, "finals": int, "experts": int, "rate": float}],
        "finals_total": int,
        "overall_success_rate": float
      }
    """
    info = {
        "file": None,
        "experts_per_round": None,
        "rounds": [],
        "finals_total": None,
        "overall_success_rate": None,
    }
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if m := RE_FILE.match(line):
            info["file"] = m.group(1)
        elif m := RE_EXP.match(line):
            info["experts_per_round"] = int(m.group(1))
        elif m := RE_ROUND.match(line):
            label = m.group(1)
            finals = int(m.group(2))
            experts = int(m.group(3))
            rate = float(m.group(4))
            info["rounds"].append(
                {"label": label, "finals": finals, "experts": experts, "rate": rate}
            )
        elif m := RE_FINALS.match(line):
            info["finals_total"] = int(m.group(1))
        elif m := RE_OVERALL.match(line):
            info["overall_success_rate"] = float(m.group(1))
    # Minimal sanity checks
    if info["file"] is None or info["experts_per_round"] is None:
        raise ValueError(
            f"Could not parse required fields from child output:\n{stdout}"
        )
    return info


def aggregate(infos: List[Dict]) -> Dict:
    """
    Aggregate across files:
      - sum of finals across all rounds/files
      - sum of expert slots across all rounds/files
      - overall success rate = total_finals / total_expert_slots
      - track experts_per_round values encountered (for visibility)
    """
    total_finals = 0
    total_expert_slots = 0
    experts_per_round_values = set()

    for inf in infos:
        experts_per_round_values.add(inf["experts_per_round"])
        for r in inf["rounds"]:
            total_finals += r["finals"]
            total_expert_slots += r["experts"]

    overall_rate = (
        (total_finals / total_expert_slots) if total_expert_slots > 0 else 0.0
    )
    return {
        "files": len(infos),
        "total_rounds": sum(len(inf["rounds"]) for inf in infos),
        "total_finals": total_finals,
        "total_expert_slots": total_expert_slots,
        "overall_success_rate": overall_rate,
        "experts_per_round_values": sorted(experts_per_round_values),
    }


def find_jsons(root: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(p for p in root.rglob(pattern) if p.is_file())
    return sorted(p for p in root.glob(pattern) if p.is_file())


def main():
    p = argparse.ArgumentParser(
        description="Invoke per-file FINAL PROB counter over a directory and compute aggregate stats."
    )
    p.add_argument(
        "--counter_script",
        type=Path,
        help="Path to the existing counting script (the one you already wrote).",
        default="count_final_probs.py",
    )
    p.add_argument("--root", type=Path, help="Directory containing JSON files.")
    p.add_argument(
        "--pattern", default="*.json", help="Glob pattern for files (default: *.json)."
    )
    p.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories."
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Parallel workers (default: 4). Use 1 for sequential.",
    )
    p.add_argument(
        "--print-per-file",
        action="store_true",
        help="Echo child script output per file.",
    )
    args = p.parse_args()

    files = find_jsons(args.root, args.pattern, args.recursive)
    if not files:
        print("No JSON files matched.", file=sys.stderr)
        sys.exit(1)

    results: List[Dict] = []
    if args.jobs == 1:
        for jf in files:
            _, out = run_counter(args.counter_script, jf)
            if args.print_per_file:
                print(out, end="")
            results.append(parse_output(out))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_counter, args.counter_script, jf): jf for jf in files}
            for fut in as_completed(futs):
                jf = futs[fut]
                try:
                    _, out = fut.result()
                    if args.print_per_file:
                        print(out, end="")
                    results.append(parse_output(out))
                except Exception as e:
                    # Fail fast: surface which file broke
                    raise RuntimeError(f"Failure while processing {jf}") from e

    agg = aggregate(results)

    # --- Aggregate report ---
    print("=== Aggregate Stats ===")
    print(f"files_processed: {agg['files']}")
    print(f"total_rounds:    {agg['total_rounds']}")
    print(f"finals_total:    {agg['total_finals']}")
    print(f"expert_slots:    {agg['total_expert_slots']}")
    print(f"overall_success: {agg['overall_success_rate']:.4f}")

    # Show distinct experts_per_round values seen across files (visibility for mismatches)
    exp_vals = agg["experts_per_round_values"]
    print(f"experts_per_round_values_seen: {exp_vals}")
    if len(exp_vals) != 1:
        # Loud visibility; we do NOT abort because cross-file differences may be expected.
        print(
            "WARNING: Inconsistent experts_per_round across files. "
            "Inspect per-file outputs (use --print-per-file) for details.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
