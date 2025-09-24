#!/usr/bin/env python3
"""
Sort Delphi JSON logs under a base directory into subfolders by test set.

- Detects question_id from each JSON (preferred) or from filename pattern like
  delphi_eval_{question_id}_{resolution_date}.json
- Classifies into:
  - evaluation            → utils.sampling.EVALUATION_QUESTION_IDS
  - evolution_evaluation  → utils.sampling.EVOLUTION_EVALUATION_QUESTION_IDS
  - unknown               → anything else

By default copies files into <dest>/<set>/<relative path>, preserving the subfolder structure
under <base>. Can move or symlink instead via flags.

Usage examples:
  python scripts/sort_jsons_by_test_set.py \
    --base results/evolution_evaluation \
    --dest results/evolution_evaluation/_sorted_by_set

  python scripts/sort_jsons_by_test_set.py --move
  python scripts/sort_jsons_by_test_set.py --link
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
from pathlib import Path
import filecmp
from typing import Optional, Tuple, Iterable

from utils.sampling import (
    TRAIN_QUESTION_IDS,
    EVALUATION_QUESTION_IDS,
    EVOLUTION_EVALUATION_QUESTION_IDS,
)


def infer_qid_from_filename(filename: str) -> Optional[str]:
    """Try to infer question_id from filenames like delphi_eval_{qid}_{date}.json."""
    try:
        stem = os.path.splitext(os.path.basename(filename))[0]
        # Grab YYYY-MM-DD from right and take the remainder as qid
        m = re.search(r"(.*?)[_-]?(\d{4}-\d{2}-\d{2})$", stem)
        if not m:
            return None
        qid = m.group(1)
        # Common prefix in our patterns
        if qid.startswith("delphi_eval_"):
            qid = qid[len("delphi_eval_") :]
        return qid or None
    except Exception:
        return None


def get_question_id(json_path: Path) -> Optional[str]:
    """Extract question_id from JSON content or filename fallback."""
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        qid = data.get("question_id")
        if isinstance(qid, str) and qid:
            return qid
    except Exception:
        pass
    return infer_qid_from_filename(str(json_path))


def classify_set(qid: str) -> str:
    # Priority: explicit train list, then eval, then evolution_eval
    if qid in set(TRAIN_QUESTION_IDS):
        return "train"
    if qid in set(EVALUATION_QUESTION_IDS):
        return "eval"
    if qid in set(EVOLUTION_EVALUATION_QUESTION_IDS):
        return "evolution_eval"
    return "unknown"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, mode: str = "copy", overwrite: bool = False):
    if dst.exists():
        if overwrite:
            if dst.is_file():
                dst.unlink()
        else:
            return  # skip
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "link":
        try:
            os.symlink(os.path.abspath(src), dst)
        except FileExistsError:
            pass
    else:
        raise ValueError(f"Unknown mode: {mode}")


def normalize_seed_dir(
    seed_dir: Path, *, mode: str = "move", dedupe: bool = True, dry_run: bool = False
):
    """Merge legacy subdirs into canonical names and remove duplicates.

    - Move evaluation → eval
    - Move evolution_evaluation → evolution_eval
    """
    canonical = {
        "evaluation": "eval",
        "evolution_evaluation": "evolution_eval",
    }
    for legacy, canon in canonical.items():
        legacy_dir = seed_dir / legacy
        if not legacy_dir.is_dir():
            continue
        canon_dir = seed_dir / canon
        # Move all jsons from legacy into canonical
        for src in legacy_dir.rglob("*.json"):
            rel = src.relative_to(legacy_dir)
            dst = canon_dir / rel
            if dry_run:
                print(f"[normalize] {legacy}/→{canon}/ {rel}")
                continue
            if dedupe and dst.exists():
                try:
                    if filecmp.cmp(src, dst, shallow=False):
                        src.unlink()
                        continue
                except Exception:
                    pass
            place_file(src, dst, mode=mode, overwrite=False)
        # Remove legacy dir if empty
        try:
            if not dry_run and legacy_dir.exists() and not any(legacy_dir.rglob("*")):
                legacy_dir.rmdir()
        except Exception:
            pass


# Skip re-processing files already sorted into any of these subfolders
SET_DIRS = {
    "train",
    "eval",
    "evolution_eval",
    "evaluation",
    "evolution_evaluation",
    "unknown",
}


def main():
    ap = argparse.ArgumentParser(description="Sort Delphi JSON logs by test set")
    ap.add_argument(
        "--base",
        default="results/evolution_evaluation",
        help="Base directory to scan recursively for .json logs",
    )
    ap.add_argument(
        "--dest",
        default=None,
        help="Destination directory root (ignored with --inplace)",
    )
    ap.add_argument(
        "--mode",
        choices=["copy", "move", "link"],
        default="copy",
        help="How to sort files into dest (copy/move/symlink)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files at destination",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Sort within each file's existing directory into set-named subfolders (default)",
    )
    ap.add_argument(
        "--no-inplace",
        action="store_true",
        help="Disable inplace sorting; use --dest/_sorted_by_set layout",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print actions without performing them"
    )
    ap.add_argument(
        "--dedupe",
        action="store_true",
        help="When a file already exists at the destination and contents match, delete the source to clean up duplicates",
    )
    ap.add_argument(
        "--scope",
        choices=["all", "seeds"],
        default="seeds",
        help="Scan scope: entire base recursively or only seed_* subdirs under each config (default: seeds)",
    )
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists() or not base.is_dir():
        print(f"Base directory not found: {base}")
        return 1

    # Determine placement strategy
    inplace = True
    if args.no_inplace:
        inplace = False
    if args.inplace:
        inplace = True

    dest_root = None
    if not inplace:
        dest_root = Path(args.dest or (base / "_sorted_by_set"))
        ensure_dir(dest_root)

    # Build list of directories to scan based on scope
    scan_dirs = []
    if args.scope == "seeds":
        # seed dirs under each config: <base>/<config>/seed_*
        for cfg_dir in base.iterdir():
            if not cfg_dir.is_dir():
                continue
            for sd in cfg_dir.glob("seed_*"):
                if sd.is_dir():
                    scan_dirs.append(sd)
    else:
        scan_dirs = [base]

    # Normalize legacy subdirs before sorting
    if args.inplace and args.scope == "seeds":
        for cfg_dir in base.iterdir():
            if not cfg_dir.is_dir():
                continue
            for sd in cfg_dir.glob("seed_*"):
                if not sd.is_dir():
                    continue
                normalize_seed_dir(
                    sd,
                    mode="move" if args.mode == "move" else "copy",
                    dedupe=args.dedupe,
                    dry_run=args.dry_run,
                )

    # Walk and collect json files
    json_files = []
    for root in scan_dirs:
        json_files.extend([p for p in root.rglob("*.json") if p.is_file()])
    if not json_files:
        print(f"No JSON files found under {base}")
        return 0

    # Sort
    total = 0
    for src in json_files:
        # Skip files already placed under set subfolders to avoid re-processing
        try:
            if src.parent.name in SET_DIRS:
                continue
        except Exception:
            pass
        # Build relative path after base
        try:
            rel = src.relative_to(base)
        except Exception:
            rel = Path(src.name)

        qid = get_question_id(src)
        if not qid:
            target_set = "unknown"
        else:
            target_set = classify_set(qid)

        if inplace:
            dst = src.parent / target_set / src.name
            display_rel = Path(src.parent.name) / src.name
        else:
            dst = dest_root / target_set / rel
            display_rel = rel
        print(f"[{target_set}] {display_rel}")
        total += 1
        if not args.dry_run:
            # If dedupe requested and destination exists with identical content, remove the source
            if args.dedupe and dst.exists():
                try:
                    if filecmp.cmp(src, dst, shallow=False):
                        # Only prune sources that are not already under a set dir
                        if src.parent.name not in SET_DIRS:
                            src.unlink()
                            continue
                except Exception:
                    pass
            place_file(src, dst, mode=args.mode, overwrite=args.overwrite)

    out_path = base if inplace else dest_root
    print(f"Processed {total} JSON files. Output at: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
