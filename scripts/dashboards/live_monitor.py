#!/usr/bin/env python3
"""
Live monitor for genetic evolution runs.

Reads monitor.csv in a run_dir and updates plots in real time.

Usage:
  uv run scripts/live_monitor.py --run-dir <path/to/run_dir>

Options:
  --interval  Refresh interval in seconds (default: 2)
  --extra     Comma-separated list of extra metric base names to plot (e.g., mean_brier, mean_abs_error)
  --save-png  Also save a PNG snapshot on each refresh into the run_dir
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
from typing import Dict, List, Any

import matplotlib.pyplot as plt


CORE_COLS = [
    "generation",
    "best_train",
    "mean_train",
    "best_val",
    "mean_val",
    "gap",
    "mutation_rate",
    "timestamp",
]


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
                    # Non-numeric or missing
                    data[k].append(float("nan"))
    return data


def main():
    ap = argparse.ArgumentParser(
        description="Live monitor for evolution runs (reads monitor.csv)."
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a run directory (timestamp_uuid).",
    )
    ap.add_argument(
        "--interval", type=float, default=2.0, help="Refresh interval in seconds."
    )
    ap.add_argument(
        "--extra",
        type=str,
        default="",
        help="Comma-separated extra metric bases to plot (e.g., mean_brier,mean_abs_error).",
    )
    ap.add_argument(
        "--save-png",
        action="store_true",
        help="Also save PNG snapshot on each refresh.",
    )
    args = ap.parse_args()

    monitor_csv = args.run_dir / "monitor.csv"
    if not monitor_csv.exists():
        print(f"No monitor.csv found at: {monitor_csv}")
        return

    extra_metrics = [m.strip() for m in args.extra.split(",") if m.strip()]

    plt.ion()
    fig, axes = plt.subplots(
        1 + (1 if extra_metrics else 0), 1, figsize=(10, 6), squeeze=False
    )
    ax_main = axes[0, 0]
    ax_extra = axes[1, 0] if extra_metrics else None

    last_len = -1
    while True:
        data = read_monitor_csv(monitor_csv)
        if not data:
            time.sleep(args.interval)
            continue

        gens = data.get("generation", [])
        if len(gens) == last_len:
            # no update
            time.sleep(args.interval)
            continue
        last_len = len(gens)

        # Clear and redraw main axes
        ax_main.clear()
        ax_main.set_title("Train/Val fitness and gap (live)")
        ax_main.set_xlabel("Generation")
        ax_main.set_ylabel("Fitness / Gap")
        if "best_train" in data:
            ax_main.plot(gens, data["best_train"], label="best_train", color="tab:blue")
        if "mean_train" in data:
            ax_main.plot(
                gens,
                data["mean_train"],
                label="mean_train",
                color="tab:blue",
                linestyle="--",
            )
        if "best_val" in data:
            ax_main.plot(gens, data["best_val"], label="best_val", color="tab:orange")
        if "mean_val" in data:
            ax_main.plot(
                gens,
                data["mean_val"],
                label="mean_val",
                color="tab:orange",
                linestyle="--",
            )
        if "gap" in data:
            ax_main.plot(gens, data["gap"], label="gap", color="tab:red")
        ax_main.legend(loc="best")
        ax_main.grid(True, linestyle="--", alpha=0.4)

        # Draw optional extra metrics
        if ax_extra is not None:
            ax_extra.clear()
            ax_extra.set_title("Extra metrics (means)")
            ax_extra.set_xlabel("Generation")
            for base in extra_metrics:
                tkey = f"train_{base}"
                vkey = f"val_{base}"
                if tkey in data:
                    ax_extra.plot(gens, data[tkey], label=tkey)
                if vkey in data:
                    ax_extra.plot(gens, data[vkey], label=vkey)
            ax_extra.legend(loc="best", fontsize="small")
            ax_extra.grid(True, linestyle="--", alpha=0.4)

        fig.tight_layout()
        # plt.pause(0.01)

        if args.save_png:
            out_png = args.run_dir / "monitor_live.png"
            fig.savefig(out_png)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
