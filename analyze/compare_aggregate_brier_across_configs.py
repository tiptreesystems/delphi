#!/usr/bin/env python3
"""
Compare Delphi aggregated Brier curves across multiple experiment configs on one plot.

For each config:
- Scans seed_* subfolders under experiment.output_dir
- Optionally limits to a set subfolder (eval/evolution_eval/train) if present or via --set
- Computes mean Brier per round across questions for each seed
- Aggregates across seeds: mean ± std per round
- Plots all configs together with vertical error bars at each round (markers on the line)

Usage example:
  uv run analyze/compare_aggregate_brier_across_configs.py \
    --configs \
      configs/evaluation_real/delphi_eval_gpt_oss_120b_3_experts_3_examples_no_system_prompt.yml \
      configs/evolution_evaluation/delphi_eval_gpt_oss_120b_5_experts_3_examples_no_system_prompt.yml \
      configs/evolution_evaluation/delphi_eval_gpt_oss_120b_superagent_aggregated_3x3.yml \
    --labels "3x3" "5x3" "superagent 3x3" \
    --set eval \
    --out results/compare_across_configs
"""

from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import yaml

# Robustly load question ID sets even if utils.py shadows the utils package
try:
    from utils.sampling import (
        TRAIN_QUESTION_IDS,
        EVALUATION_QUESTION_IDS,
        EVOLUTION_EVALUATION_QUESTION_IDS,
    )
except Exception:
    import importlib.util as _ilu

    _root = Path(__file__).resolve().parents[1]
    _sampling_fp = _root / "utils" / "sampling.py"
    _spec = _ilu.spec_from_file_location("project_utils_sampling", str(_sampling_fp))
    _mod = _ilu.module_from_spec(_spec)
    assert _spec and _spec.loader, f"Cannot load sampling module at {_sampling_fp}"
    _spec.loader.exec_module(_mod)
    TRAIN_QUESTION_IDS = set(getattr(_mod, "TRAIN_QUESTION_IDS", []))
    EVALUATION_QUESTION_IDS = set(getattr(_mod, "EVALUATION_QUESTION_IDS", []))
    EVOLUTION_EVALUATION_QUESTION_IDS = set(
        getattr(_mod, "EVOLUTION_EVALUATION_QUESTION_IDS", [])
    )


def load_experiment_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_delphi_log_filename(filename: str, file_pattern: str) -> Tuple[str, str]:
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    prefix_end = file_pattern.find("{question_id}")
    if prefix_end == -1:
        raise ValueError(f"Pattern {file_pattern} must contain {{question_id}}")
    prefix = file_pattern[:prefix_end]
    if not stem.startswith(prefix):
        raise ValueError(f"Filename {filename} doesn't match pattern {file_pattern}")
    remainder = stem[len(prefix) :]
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    date_matches = re.findall(date_pattern, remainder)
    if not date_matches:
        raise ValueError(f"Cannot find date in filename: {filename}")
    date_str = date_matches[-1]
    date_pos = remainder.rfind("_" + date_str)
    if date_pos == -1:
        date_pos = remainder.rfind(date_str)
        if date_pos == -1:
            raise ValueError(f"Cannot parse question_id and date from: {filename}")
        question_id = remainder[:date_pos].rstrip("_")
    else:
        question_id = remainder[:date_pos].rstrip("_")
    return question_id, date_str


def median(lst: List[float]) -> Optional[float]:
    vals = [float(x) for x in lst if isinstance(x, (int, float))]
    if not vals:
        return None
    return float(np.median(vals))


def mean(lst: List[float]) -> Optional[float]:
    vals = [float(x) for x in lst if isinstance(x, (int, float))]
    if not vals:
        return None
    return float(np.mean(vals))


def collect_round_probs(delphi_log: Dict[str, Any]) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    rounds = delphi_log.get("rounds", [])
    for r in rounds:
        r_idx = int(r.get("round", 0))
        expert_dict = r.get("experts", {})
        probs = []
        for _, entry in expert_dict.items():
            p = entry.get("prob")
            if isinstance(p, (int, float)):
                probs.append(float(p))
        out[r_idx] = probs
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def compute_seed_curve(
    seed_dir: Path,
    file_pattern: str,
    allowed_qids: Optional[set] = None,
    preferred_subdir: Optional[str] = None,
    expert_agg: str = "median",
) -> Dict[int, float]:
    prefix = file_pattern.split("{")[0]
    # Scan only preferred subdir if present
    roots = [seed_dir]
    if preferred_subdir and (seed_dir / preferred_subdir).is_dir():
        roots = [seed_dir / preferred_subdir]

    per_round_accum: Dict[int, List[float]] = {}
    expected_rounds: Optional[set[int]] = None

    logs: List[Path] = []
    for root in roots:
        logs.extend(
            [
                p
                for p in root.rglob("*.json")
                if p.is_file() and p.name.startswith(prefix)
            ]
        )

    for fp in logs:
        try:
            qid, _ = parse_delphi_log_filename(fp.name, file_pattern)
        except Exception:
            continue
        if allowed_qids is not None and qid not in allowed_qids:
            continue
        try:
            with fp.open("r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            continue
        probs_by_round = collect_round_probs(log)
        rounds_set = set(probs_by_round.keys())
        if expected_rounds is None:
            expected_rounds = rounds_set
        elif rounds_set != expected_rounds:
            raise RuntimeError(
                f"Inconsistent rounds in seed {seed_dir.name}: expected {sorted(expected_rounds)}, got {sorted(rounds_set)} for {fp.name}"
            )
        # compute Briers per round using expert aggregator per round, then average across questions
        # Note: actual outcome is not used here, because we don't have resolution loader in this lightweight script.
        # Instead, we rely on logged per-round Briers not being present; thus we compute Brier from probs-only requires outcome.
        # To keep this script self-contained, we assume comparison across configs uses the aggregated per-round Brier already
        # saved? If not, we replicate the approach from the other script that recomputes with outcomes — but that needs loader.
        # Here we proceed by reading per-round probs and computing per-qid aggregated brier requires actuals; so we cannot avoid loader.
        # Therefore, this function should be called with outcomes baked in elsewhere. To keep consistency, we will handle outcome outside.
        pass

    return {}


def compute_mean_std_across_seeds(
    cfg: dict, set_name: str, allowed_qids: Optional[set]
) -> Tuple[List[int], List[float], List[float]]:
    """Replicate the aggregation logic from aggregate_brier_rounds_across_seeds, but lighter.
    Requires dataset loader to compute briers vs actuals.
    """
    from dataset.dataloader import ForecastDataLoader

    loader = ForecastDataLoader()

    base_dir = Path(cfg["experiment"]["output_dir"])
    seed_dirs = [
        d for d in base_dir.iterdir() if d.is_dir() and re.match(r"^seed_\d+$", d.name)
    ]
    if not seed_dirs:
        return [], [], []
    seed_dirs.sort(key=lambda p: p.name)
    file_pattern = cfg["output"]["file_pattern"]
    preferred_subdir = None
    if set_name in {"eval", "evolution_eval", "train"}:
        preferred_subdir = set_name

    def _compute_seed(seed_dir: Path) -> Dict[int, float]:
        prefix = file_pattern.split("{")[0]
        roots = [seed_dir]
        if preferred_subdir and (seed_dir / preferred_subdir).is_dir():
            roots = [seed_dir / preferred_subdir]
        per_round_accum: Dict[int, List[float]] = {}
        expected_rounds: Optional[set[int]] = None
        logs: List[Path] = []
        for root in roots:
            logs.extend(
                [
                    p
                    for p in root.rglob("*.json")
                    if p.is_file() and p.name.startswith(prefix)
                ]
            )
        for fp in logs:
            try:
                qid, res_date = parse_delphi_log_filename(fp.name, file_pattern)
            except Exception:
                continue
            if allowed_qids is not None and qid not in allowed_qids:
                continue
            try:
                res = loader.get_resolution(qid, res_date)
                if not res or not getattr(res, "resolved", False):
                    continue
                actual = float(res.resolved_to)
            except Exception:
                continue
            try:
                with fp.open("r", encoding="utf-8") as f:
                    log = json.load(f)
            except Exception:
                continue
            probs_by_round = collect_round_probs(log)
            rounds_set = set(probs_by_round.keys())
            if expected_rounds is None:
                expected_rounds = rounds_set
            elif rounds_set != expected_rounds:
                raise RuntimeError(
                    f"Inconsistent rounds in seed {seed_dir.name}: expected {sorted(expected_rounds)}, got {sorted(rounds_set)} for {fp.name}"
                )
            for r_idx, probs in probs_by_round.items():
                agg = (
                    median(probs)
                    if cfg.get("expert_agg", "median") == "median"
                    else mean(probs)
                )
                if agg is None:
                    continue
                brier = (agg - actual) ** 2
                per_round_accum.setdefault(r_idx, []).append(brier)
        return {r: float(np.mean(vals)) for r, vals in per_round_accum.items() if vals}

    per_seed_curves: Dict[str, Dict[int, float]] = {}
    for sd in seed_dirs:
        per_seed_curves[sd.name] = _compute_seed(sd)

    # Align rounds and compute mean/std across seeds per round
    round_to_seed_values: Dict[int, List[float]] = {}
    for _, curve in per_seed_curves.items():
        for r, v in curve.items():
            round_to_seed_values.setdefault(r, []).append(v)
    rounds_sorted = sorted(round_to_seed_values.keys())
    mean_curve = (
        [float(np.mean(round_to_seed_values[r])) for r in rounds_sorted]
        if rounds_sorted
        else []
    )
    std_curve = (
        [float(np.std(round_to_seed_values[r])) for r in rounds_sorted]
        if rounds_sorted
        else []
    )
    return rounds_sorted, mean_curve, std_curve


def compute_sf_public_baselines_across_seeds(
    cfg: dict, set_name: str, allowed_qids: Optional[set]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute Superforecaster and Public baselines (mean ± std across seeds).

    For each seed, compute the average Brier across questions for:
      - SF baseline: median SF prob per question
      - Public baseline: median Public prob per question
    Then return mean and std across seeds for each baseline.
    """
    from dataset.dataloader import ForecastDataLoader

    loader = ForecastDataLoader()

    base_dir = Path(cfg["experiment"]["output_dir"])
    seed_dirs = [
        d for d in base_dir.iterdir() if d.is_dir() and re.match(r"^seed_\d+$", d.name)
    ]
    if not seed_dirs:
        return None, None, None, None
    seed_dirs.sort(key=lambda p: p.name)
    file_pattern = cfg["output"]["file_pattern"]
    preferred_subdir = None
    if set_name in {"eval", "evolution_eval", "train"}:
        preferred_subdir = set_name

    sf_seed_vals: List[float] = []
    pub_seed_vals: List[float] = []

    for sd in seed_dirs:
        prefix = file_pattern.split("{")[0]
        roots = [sd]
        if preferred_subdir and (sd / preferred_subdir).is_dir():
            roots = [sd / preferred_subdir]
        logs: List[Path] = []
        for root in roots:
            logs.extend(
                [
                    p
                    for p in root.rglob("*.json")
                    if p.is_file() and p.name.startswith(prefix)
                ]
            )
        sf_briers: List[float] = []
        pub_briers: List[float] = []
        for fp in logs:
            try:
                qid, res_date = parse_delphi_log_filename(fp.name, file_pattern)
            except Exception:
                continue
            if allowed_qids is not None and qid not in allowed_qids:
                continue
            try:
                res = loader.get_resolution(qid, res_date)
                if not res or not getattr(res, "resolved", False):
                    continue
                actual = float(res.resolved_to)
            except Exception:
                continue
            try:
                sfs = loader.get_super_forecasts(
                    question_id=qid, resolution_date=res_date
                )
                sf_probs = [float(s.forecast) for s in sfs]
                if sf_probs:
                    m_sf = float(np.median(sf_probs))
                    sf_briers.append((m_sf - actual) ** 2)
            except Exception:
                pass
            try:
                pubs = loader.get_public_forecasts(
                    question_id=qid, resolution_date=res_date
                )
                pub_probs = [float(p.forecast) for p in pubs]
                if pub_probs:
                    m_pub = float(np.median(pub_probs))
                    pub_briers.append((m_pub - actual) ** 2)
            except Exception:
                pass
        if sf_briers:
            sf_seed_vals.append(float(np.mean(sf_briers)))
        if pub_briers:
            pub_seed_vals.append(float(np.mean(pub_briers)))

    sf_mean = float(np.mean(sf_seed_vals)) if sf_seed_vals else None
    sf_std = float(np.std(sf_seed_vals)) if sf_seed_vals else None
    pub_mean = float(np.mean(pub_seed_vals)) if pub_seed_vals else None
    pub_std = float(np.std(pub_seed_vals)) if pub_seed_vals else None
    return sf_mean, sf_std, pub_mean, pub_std


def main():
    ap = argparse.ArgumentParser(
        description="Compare Delphi aggregated Brier curves across configs"
    )
    ap.add_argument(
        "--configs", nargs="+", required=True, help="List of YAML configs to compare"
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels (same length as --configs)",
    )
    ap.add_argument(
        "--set",
        default="auto",
        choices=["auto", "train", "eval", "evolution_eval"],
        help="Restrict to a question set; auto derives from each YAML",
    )
    ap.add_argument(
        "--out", default=None, help="Output directory for plot and summary JSON"
    )
    args = ap.parse_args()

    cfgs = [load_experiment_config(p) for p in args.configs]
    labels = args.labels or [Path(p).stem for p in args.configs]
    if len(labels) != len(cfgs):
        raise ValueError("Number of labels must match number of configs")

    series = []  # list of (label, xs, means, stds)
    set_suffix = args.set
    for cfg, label in zip(cfgs, labels):
        # Determine allowed set
        set_name = "auto"
        allowed_qids = None
        if args.set != "auto":
            set_name = args.set
            if args.set == "train":
                allowed_qids = set(TRAIN_QUESTION_IDS)
            elif args.set == "eval":
                allowed_qids = set(EVALUATION_QUESTION_IDS)
            elif args.set == "evolution_eval":
                allowed_qids = set(EVOLUTION_EVALUATION_QUESTION_IDS)
        else:
            sampling_method = (cfg.get("data", {}).get("sampling", {}) or {}).get(
                "method"
            )
            if sampling_method == "evaluation":
                allowed_qids = set(EVALUATION_QUESTION_IDS)
                set_name = "eval"
            elif sampling_method == "evolution_evaluation":
                allowed_qids = set(EVOLUTION_EVALUATION_QUESTION_IDS)
                set_name = "evolution_eval"
            elif sampling_method in ("train", "train_small"):
                allowed_qids = set(TRAIN_QUESTION_IDS)
                set_name = "train"
        if set_suffix == "auto":
            set_suffix = set_name

        rounds, means, stds = compute_mean_std_across_seeds(cfg, set_name, allowed_qids)
        series.append((label, [r + 1 for r in rounds], means, stds))

    # Choose output dir
    out_dir = (
        Path(args.out)
        if args.out
        else (Path(cfgs[0]["experiment"]["output_dir"]) / "set_{}".format(set_suffix))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot with a light→dark sequential palette matching series order
    fig, ax = plt.subplots(figsize=(14, 7))
    try:
        # Use a perceptually uniform colormap (light → dark across series)
        cmap = plt.get_cmap("viridis")
        colors = [cmap(x) for x in np.linspace(0.9, 0.25, max(1, len(series)))]
    except Exception:
        colors = [None] * len(series)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    for i, (label, xs, means, stds) in enumerate(series):
        if not xs:
            continue
        m = markers[i % len(markers)]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            fmt="-",
            marker=m,
            markersize=7,
            capsize=3,
            linewidth=2.2,
            color=colors[i] if i < len(colors) else None,
            label=label,
        )
    # Compute SF/Public baselines using the first config (assumes same set across configs)
    sf_mean = sf_std = pub_mean = pub_std = None
    if cfgs:
        # Determine allowed_qids for first config, consistent with above
        cfg0 = cfgs[0]
        if args.set != "auto":
            sn = args.set
            if sn == "train":
                allowed0 = set(TRAIN_QUESTION_IDS)
            elif sn == "eval":
                allowed0 = set(EVALUATION_QUESTION_IDS)
            elif sn == "evolution_eval":
                allowed0 = set(EVOLUTION_EVALUATION_QUESTION_IDS)
            else:
                allowed0 = None
            set0 = sn
        else:
            sm = (cfg0.get("data", {}).get("sampling", {}) or {}).get("method")
            if sm == "evaluation":
                allowed0 = set(EVALUATION_QUESTION_IDS)
                set0 = "eval"
            elif sm == "evolution_evaluation":
                allowed0 = set(EVOLUTION_EVALUATION_QUESTION_IDS)
                set0 = "evolution_eval"
            elif sm in ("train", "train_small"):
                allowed0 = set(TRAIN_QUESTION_IDS)
                set0 = "train"
            else:
                allowed0 = None
                set0 = "auto"
        sf_mean, sf_std, pub_mean, pub_std = compute_sf_public_baselines_across_seeds(
            cfg0, set0, allowed0
        )
        if pub_mean is not None:
            # Nicer orange for Public baseline
            ax.axhline(
                pub_mean,
                color="#ff7f0e",
                linestyle="-.",
                linewidth=1.8,
                alpha=0.95,
                label="Median of Public",
            )
        if sf_mean is not None:
            # Nicer green for SF baseline
            ax.axhline(
                sf_mean,
                color="#2ca02c",
                linestyle="--",
                linewidth=1.8,
                alpha=0.95,
                label="Median of Superforecasters",
            )
    ax.set_xlabel("Round", fontsize=18)
    ax.set_ylabel("Average Brier Score \n (mean ± std, ↓ lower is better)", fontsize=18)
    # plt.title('Delphi median Brier across rounds (mean±std across seeds)')
    if series and series[0][1]:
        all_rounds = sorted(set().union(*[set(xs) for _, xs, _, _ in series]))
        # Custom x-axis labels: 1 -> "initial forecasts", others -> "update k"
        xticklabels = [
            "initial forecasts" if r == 1 else f"update {r - 1}" for r in all_rounds
        ]
        ax.set_xticks(all_rounds)
        ax.set_xticklabels(xticklabels, fontsize=16)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=16)
    out_png = out_dir / f"compare_aggregate_brier_{set_suffix}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    # Also save as vector PDF for publication-quality
    out_pdf = out_dir / f"compare_aggregate_brier_{set_suffix}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved plot -> {out_png}")
    print(f"Saved plot -> {out_pdf}")

    # Save summary JSON
    summary = {
        "set": set_suffix,
        "series": [
            {
                "label": label,
                "rounds": xs,
                "mean_curve": means,
                "std_curve": stds,
            }
            for (label, xs, means, stds) in series
        ],
        "sf_baseline": {"mean": sf_mean, "std": sf_std},
        "public_baseline": {"mean": pub_mean, "std": pub_std},
    }
    out_json = out_dir / f"compare_aggregate_brier_{set_suffix}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON -> {out_json}")


if __name__ == "__main__":
    main()
