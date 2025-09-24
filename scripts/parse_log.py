#!/usr/bin/env python3
"""
Parse a .log of repeating prompt/question/metrics blocks and output JSON grouped by generation.

Usage:
  python parse_log_to_json.py --log path/to/file.log --out parsed.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any

import debugpy

print("Waiting for debugger attach...")
debugpy.listen(5679)
debugpy.wait_for_client()
print("Debugger attached.")


GEN_RE = re.compile(r"\bGeneration\s+(\d+)\b")
PROMPT_RE = re.compile(r"Loading system prompt:\s*(.+?)\s*$")
PROMPT_FIRST_RE = re.compile(r"Loaded prompt first 100 chars:\s*(.+?)\s*$")
# Example: "      QRRPONTTL: improvement=0.165, variance=0.002, smoothness=0.465, fitness=0.8059"
QUESTION_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_\-]+):\s*(.*)$")
KV_RE = re.compile(r"([A-Za-z_]+)=([\-+]?\d+(?:\.\d+)?)")
SUPER_RE = re.compile(
    r"Superforecaster median:\s*([\-+]?\d+(?:\.\d+)?),\s*pred_prob:\s*([\-+]?\d+(?:\.\d+)?),\s*actual:\s*([\-+]?\d+(?:\.\d+)?)"
)
EVOLUTION_SUMMARY_START_RE = re.compile(r"^=+?\s*EVOLUTION COMPLETED\s*=+?$")
TOTAL_GENERATIONS_RE = re.compile(r"Total generations:\s*(\d+)")
BEST_FITNESS_RE = re.compile(r"Best fitness:\s*([\-+]?\d+(?:\.\d+)?)")
FINAL_MUT_RATE_RE = re.compile(r"Final mutation rate:\s*([\-+]?\d+(?:\.\d+)?)")

# Phase marker (generic). Examples it should match:
# - "Evaluating best prompt on validation set of (NN questions)..."
# - "Evaluating best prompt on full validation set of (NN questions)..."
# - "Evaluating best prompt on separate test set of (NN questions...)"
# - "Evaluating best prompt on evolution test set of (NN questions...)"
PHASE_SWITCH_RE = re.compile(
    r"Evaluating best prompt on\s+(.*?)\s+of\s*\(\d+\s+questions",
    re.IGNORECASE,
)

FIELDS_IN_RECORD = (
    "prompt",
    "prompt_first_100",
    "question",
    "improvement",
    "variance",
    "smoothness",
    "fitness",
    "superforecaster_median",
    "pred_prob",
    "actual",
)


def maybe_float(x: str) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def flush_record(
    groups: Dict[str, List[Dict[str, Any]]], gen_key: str, rec: Dict[str, Any]
) -> None:
    """Append record to groups[gen_key] if it has the minimum expected fields."""
    if not rec:
        return
    # Require at least question + fitness line OR superforecaster line
    has_q = bool(rec.get("question"))
    has_any_metrics = any(
        k in rec for k in ("improvement", "variance", "smoothness", "fitness")
    )
    has_sf = all(k in rec for k in ("superforecaster_median", "pred_prob", "actual"))

    if has_q and (has_any_metrics or has_sf):
        # Keep only known fields; drop partial/noise keys.
        cleaned = {k: rec[k] for k in FIELDS_IN_RECORD if k in rec}
        groups.setdefault(gen_key, []).append(cleaned)


def parse_log(path: Path) -> Dict[str, Any]:
    """
    Returns:
      {
        "evolution": {
          "generations": { "0": [records...], "1": [...], ... },
          "summary": {
            "total_generations": int | None,
            "best_fitness": float | None,
            "final_mutation_rate": float | None
          }
        },
        "validation": [records...],        # optional
        "evaluation": [records...]         # optional
      }
    """
    out: Dict[str, Any] = {
        "evolution": {
            "generations": {},
            "summary": {
                "total_generations": None,
                "best_fitness": None,
                "final_mutation_rate": None,
            },
        }
    }

    # --- parsing state ---
    current_section = "evolution"  # "evolution" | "validation" | "evaluation"
    current_gen = "unknown"
    current_record: Dict[str, Any] = {}
    in_evolution_summary_block = False

    # helpers
    def flush_current():
        nonlocal current_record, current_gen, current_section
        if not current_record:
            return
        if current_section == "evolution":
            flush_record(out["evolution"]["generations"], current_gen, current_record)
        else:
            key = current_section
            out.setdefault(key, [])
            # Reuse the same schema/cleaning as evolution records
            cleaned = {
                k: current_record[k] for k in FIELDS_IN_RECORD if k in current_record
            }
            if cleaned:
                out[key].append(cleaned)
        current_record = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # ---- phase switching ----
            pm = PHASE_SWITCH_RE.search(line)
            if pm:
                phrase = pm.group(1).strip().lower()
                flush_current()
                if "valid" in phrase:
                    current_section = "validation"
                elif "test" in phrase:
                    current_section = "evaluation"
                else:
                    # Unknown phrase; keep prior section
                    pass
                continue

            # ---- evolution summary detection ----
            if EVOLUTION_SUMMARY_START_RE.match(line):
                in_evolution_summary_block = True
                continue
            if in_evolution_summary_block:
                m = TOTAL_GENERATIONS_RE.search(line)
                if m:
                    out["evolution"]["summary"]["total_generations"] = int(m.group(1))
                    continue
                m = BEST_FITNESS_RE.search(line)
                if m:
                    out["evolution"]["summary"]["best_fitness"] = maybe_float(
                        m.group(1)
                    )
                    continue
                m = FINAL_MUT_RATE_RE.search(line)
                if m:
                    out["evolution"]["summary"]["final_mutation_rate"] = maybe_float(
                        m.group(1)
                    )
                    continue
                # End summary block once we hit a blank line after header region
                if not line.strip():
                    in_evolution_summary_block = False
                # Keep scanning next lines either way
                continue

            # ---- generation tracking (evolution only) ----
            gmatch = GEN_RE.search(line)
            if gmatch:
                # Only treat as a new generation in the evolution section
                if current_section == "evolution":
                    flush_current()
                    current_gen = gmatch.group(1)
                continue

            # ---- prompt / content parsing (shared schema) ----
            pm = PROMPT_RE.search(line)
            if pm:
                flush_current()
                current_record = {}
                current_record["prompt"] = pm.group(1).strip()
                continue

            pf = PROMPT_FIRST_RE.search(line)
            if pf:
                current_record["prompt_first_100"] = pf.group(1).strip()
                continue

            qm = QUESTION_LINE_RE.match(line)
            if qm:
                current_record["question"] = qm.group(1).strip()
                rest = qm.group(2).strip()
                if rest:
                    for k, v in KV_RE.findall(rest):
                        if k in ("improvement", "variance", "smoothness", "fitness"):
                            fv = maybe_float(v)
                            if fv is not None:
                                current_record[k] = fv
                continue

            if (
                ("improvement=" in line)
                or ("smoothness=" in line)
                or ("fitness=" in line)
            ):
                for k, v in KV_RE.findall(line):
                    if k in ("improvement", "variance", "smoothness", "fitness"):
                        fv = maybe_float(v)
                        if fv is not None:
                            current_record[k] = fv
                continue

            sf = SUPER_RE.search(line)
            if sf:
                current_record["superforecaster_median"] = maybe_float(sf.group(1))
                current_record["pred_prob"] = maybe_float(sf.group(2))
                current_record["actual"] = maybe_float(sf.group(3))
                continue

        # EOF
        flush_current()

    # Sort evolution generations numerically
    gens = out["evolution"]["generations"]

    def gen_sort_key(k: str):
        try:
            return int(k)
        except ValueError:
            return float("inf")

    out["evolution"]["generations"] = {
        k: gens[k] for k in sorted(gens.keys(), key=gen_sort_key)
    }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True, help="Path to input .log file")
    ap.add_argument("--indent", type=int, default=2, help="Indent for JSON output")
    args = ap.parse_args()

    args.out = args.log.with_suffix(".json")

    data = parse_log(args.log)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=args.indent)

    # Nice summary line
    evo_gen_counts = {
        g: len(v) for g, v in data.get("evolution", {}).get("generations", {}).items()
    }
    n_val = len(data.get("validation", []))
    n_eval = len(data.get("evaluation", []))
    print(
        "Wrote",
        args.out,
        f"| evolution generations: {evo_gen_counts}",
        f"| validation records: {n_val}",
        f"| evaluation records: {n_eval}",
    )


if __name__ == "__main__":
    main()
