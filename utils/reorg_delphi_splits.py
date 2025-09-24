#!/usr/bin/env python3
"""
Reorganize existing Delphi logs into val/test by question IDs.

Use case: Some runs wrote both validation and evolution_evaluation logs under the
same split (e.g., "test"). This tool moves files so that:

  - test: only contains EVOLUTION_EVALUATION_QUESTION_IDS
  - val:  contains all other question IDs

It operates per-generation and per-candidate under <run>/evolved_prompts/gen_XXX/cand_*/delphi_logs.

Example:
  uv run scripts/reorg_delphi_splits.py --run-dir results/.../2025..._abcd --gen 5

Add --dry-run to preview without moving files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Set

try:
    # Source of the canonical evolution-evaluation IDs
    from utils.sampling import EVOLUTION_EVALUATION_QUESTION_IDS as EVAL_IDS
except Exception:
    EVAL_IDS: Set[str] = set()


def move_file(src: Path, dst: Path, dry_run: bool) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[skip] Destination exists: {dst}")
        return False
    if dry_run:
        print(f"[dry-run] mv {src} -> {dst}")
        return True
    shutil.move(str(src), str(dst))
    print(f"[moved] {src} -> {dst}")
    return True


def reorganize_run(run_dir: Path, gen: int, dry_run: bool = False) -> None:
    gen_dir = run_dir / "evolved_prompts" / f"gen_{gen:03d}"
    if not gen_dir.exists():
        # Also try dry_run variant
        alt = run_dir / "dry_run" / "evolved_prompts" / f"gen_{gen:03d}"
        if alt.exists():
            gen_dir = alt
        else:
            raise SystemExit(f"Generation dir not found: {gen_dir}")

    cand_dirs = [
        d for d in gen_dir.iterdir() if d.is_dir() and d.name.startswith("cand_")
    ]
    if not cand_dirs:
        print(f"No candidate dirs found under {gen_dir}")
        return

    print(f"Reorganizing {len(cand_dirs)} candidates under {gen_dir}")

    total_moves = 0
    for cdir in sorted(cand_dirs):
        logs_base = cdir / "delphi_logs"
        if not logs_base.exists():
            continue
        test_dir = logs_base / "test"
        val_dir = logs_base / "val"

        # Move non-evolution-eval questions out of test -> val
        if test_dir.exists():
            for fp in sorted(test_dir.glob("*.json")):
                qid = fp.stem
                if qid not in EVAL_IDS:
                    dst = val_dir / fp.name
                    if move_file(fp, dst, dry_run):
                        total_moves += 1

        # Move evolution-eval questions out of val -> test (in case any landed there)
        if val_dir.exists():
            for fp in sorted(val_dir.glob("*.json")):
                qid = fp.stem
                if qid in EVAL_IDS:
                    dst = test_dir / fp.name
                    if move_file(fp, dst, dry_run):
                        total_moves += 1

    print(f"Done. Files moved: {total_moves}{' (dry-run)' if dry_run else ''}")


def main():
    ap = argparse.ArgumentParser(
        description="Reorganize Delphi logs val/test by evolution-evaluation question IDs"
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a specific run directory (timestamp folder)",
    )
    ap.add_argument(
        "--gen", type=int, required=True, help="Generation number (e.g., 5)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Preview moves without changing files"
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    if not EVAL_IDS:
        print(
            "Warning: EVOLUTION_EVALUATION_QUESTION_IDS could not be imported; proceeding with empty set."
        )

    reorganize_run(run_dir, args.gen, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
