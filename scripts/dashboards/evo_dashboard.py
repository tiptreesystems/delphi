#!/usr/bin/env python3
"""
Evolution Dashboard (offline CLI)

Generates visualizations from a single run_dir:

Subcommands:
  summary         Plot train/val fitness and gap over generations (monitor.csv)
  metrics         Plot selected mean metrics over generations for train/val
  list-prompts    List prompt candidates discovered for a generation
  plot-delphi     Plot expert probabilities across rounds for a single Delphi log

Examples:
  uv run scripts/evo_dashboard.py summary --run-dir <run_dir>
  uv run scripts/evo_dashboard.py metrics --run-dir <run_dir> \
      --metrics mean_brier,mean_abs_error,delphi_total_improvement
  uv run scripts/evo_dashboard.py list-prompts --run-dir <run_dir> --gen 0
  uv run scripts/evo_dashboard.py plot-delphi --run-dir <run_dir> \
      --gen 0 --cand cand_01_87daa3d9 --split val \
      --question-id <question_uuid>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt


def _ensure_plots_dir(run_dir: Path) -> Path:
    p = run_dir / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_monitor_csv(path: Path) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    if not path.exists():
        return data
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except Exception:
                    data[k].append(float("nan"))
    return data


def cmd_summary(run_dir: Path) -> None:
    mon = read_monitor_csv(run_dir / "monitor.csv")
    if not mon:
        print(f"No monitor.csv in {run_dir}")
        return
    gens = mon.get("generation", [])
    plt.figure(figsize=(10, 6))
    if "best_train" in mon:
        plt.plot(gens, mon["best_train"], label="best_train", color="tab:blue")
    if "mean_train" in mon:
        plt.plot(gens, mon["mean_train"], "--", label="mean_train", color="tab:blue")
    if "best_val" in mon:
        plt.plot(gens, mon["best_val"], label="best_val", color="tab:orange")
    if "mean_val" in mon:
        plt.plot(gens, mon["mean_val"], "--", label="mean_val", color="tab:orange")
    if "gap" in mon:
        plt.plot(gens, mon["gap"], label="gap", color="tab:red")
    if "mutation_rate" in mon:
        plt.plot(gens, mon["mutation_rate"], label="mutation_rate", color="tab:green")
    plt.xlabel("Generation")
    plt.ylabel("Fitness / Gap")
    plt.title("Evolution summary (fitness, gap, mutation rate)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    out_dir = _ensure_plots_dir(run_dir)
    out_path = out_dir / "summary.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Wrote {out_path}")


def cmd_metrics(run_dir: Path, metrics: List[str]) -> None:
    mon = read_monitor_csv(run_dir / "monitor.csv")
    if not mon:
        print(f"No monitor.csv in {run_dir}")
        return
    gens = mon.get("generation", [])
    plt.figure(figsize=(11, 6))
    for base in metrics:
        tkey = f"train_{base}"
        vkey = f"val_{base}"
        if tkey in mon:
            plt.plot(gens, mon[tkey], label=tkey)
        if vkey in mon:
            plt.plot(gens, mon[vkey], label=vkey)
    plt.xlabel("Generation")
    plt.ylabel("Metric mean")
    plt.title("Selected metric means over generations")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best", fontsize="small")
    out_dir = _ensure_plots_dir(run_dir)
    safe = "_".join(m.replace("/", "_") for m in metrics) or "metrics"
    out_path = out_dir / f"metrics_{safe}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Wrote {out_path}")


def cmd_list_prompts(run_dir: Path, gen: int) -> None:
    gen_dir = run_dir / "evolved_prompts" / f"gen_{gen:03d}"
    if not gen_dir.exists():
        print(f"No directory: {gen_dir}")
        return
    cands = sorted(
        [p.name for p in gen_dir.iterdir() if p.is_dir() and p.name.startswith("cand_")]
    )
    for c in cands:
        print(c)


def _load_delphi_log(
    run_dir: Path, gen: int, cand: str, split: str, question_id: str
) -> Dict[str, Any]:
    path = (
        run_dir
        / "evolved_prompts"
        / f"gen_{gen:03d}"
        / cand
        / "delphi_logs"
        / split
        / f"{question_id}.json"
    )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cmd_plot_delphi(
    run_dir: Path, gen: int, cand: str, split: str, question_id: str
) -> None:
    log = _load_delphi_log(run_dir, gen, cand, split, question_id)
    rounds = log.get("rounds", [])
    if not rounds:
        print("No rounds in log")
        return
    # Collect per-expert probability series across rounds (including round 0)
    expert_ids = sorted(
        set().union(*[set(r.get("experts", {}).keys()) for r in rounds])
    )
    series: Dict[str, List[float]] = {sfid: [] for sfid in expert_ids}
    xs: List[int] = []
    for r in rounds:
        xs.append(int(r.get("round", 0)))
        for sfid in expert_ids:
            entry = r.get("experts", {}).get(sfid)
            prob = None
            if isinstance(entry, dict):
                prob = entry.get("prob")
                # Some logs may hold nested dicts under 'response'; ignore here
            series[sfid].append(
                prob if isinstance(prob, (int, float)) else float("nan")
            )

    plt.figure(figsize=(10, 6))
    for sfid, vals in series.items():
        plt.plot(xs, vals, marker="o", label=sfid)
    plt.xlabel("Round")
    plt.ylabel("Expert probability")
    title_q = (log.get("question_text") or str(question_id))[:80]
    plt.title(f"Expert probabilities across rounds\n{cand} | {split} | Q: {title_q}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best", fontsize="small", ncol=2)
    out_dir = _ensure_plots_dir(run_dir)
    out_path = (
        out_dir
        / f"delphi_expert_probs_gen{gen:03d}_{cand}_{split}_{question_id[:8]}.png"
    )
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Evolution dashboard (offline plotting)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary", help="Plot train/val fitness & gap over generations")
    s.add_argument("--run-dir", type=Path, required=True)

    m = sub.add_parser("metrics", help="Plot selected mean metrics across generations")
    m.add_argument("--run-dir", type=Path, required=True)
    m.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Comma-separated metric bases (e.g., mean_brier,mean_abs_error,delphi_total_improvement)",
    )

    l = sub.add_parser("list-prompts", help="List prompt candidates in a generation")
    l.add_argument("--run-dir", type=Path, required=True)
    l.add_argument("--gen", type=int, required=True)

    d = sub.add_parser(
        "plot-delphi", help="Plot expert probabilities across rounds for one Delphi log"
    )
    d.add_argument("--run-dir", type=Path, required=True)
    d.add_argument("--gen", type=int, required=True)
    d.add_argument(
        "--cand",
        type=str,
        required=True,
        help="Candidate folder name, e.g., cand_01_87daa3d9",
    )
    d.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    d.add_argument("--question-id", type=str, required=True)

    args = ap.parse_args()

    if args.cmd == "summary":
        cmd_summary(args.run_dir)
    elif args.cmd == "metrics":
        metrics = [s.strip() for s in args.metrics.split(",") if s.strip()]
        cmd_metrics(args.run_dir, metrics)
    elif args.cmd == "list-prompts":
        cmd_list_prompts(args.run_dir, args.gen)
    elif args.cmd == "plot-delphi":
        cmd_plot_delphi(args.run_dir, args.gen, args.cand, args.split, args.question_id)


if __name__ == "__main__":
    main()
