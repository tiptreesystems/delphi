import os
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset.dataloader import ForecastDataLoader
import debugpy
print("Waiting for debugger attach...")
debugpy.listen(5679)
debugpy.wait_for_client()
print("Debugger attached.")


loader = ForecastDataLoader()
# ----------------------------
# 1) Loader (uses your format)
# ----------------------------
def load_results(dirpath: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Loads results from files named:
      questionId_date_forecasterID_{type}.json
    where type is 'both-best-worst-feedback' or 'single-best-feedback'.
    Returns two nested dicts: {question_id: {forecaster_id: data}}
    """
    both_best_worst_feedback = defaultdict(dict)
    single_best_feedback = defaultdict(dict)

    for filename in os.listdir(dirpath):
        if not filename.endswith('.json'):
            continue

        name_without_ext = filename[:-5]  # remove ".json"
        parts = name_without_ext.rsplit('_', 3)  # question_id, date, forecaster_id, feedback_type
        if len(parts) != 4:
            # tolerate older underscore style by normalizing after split
            # or just skip malformed files
            # print(f"Skipping unexpected filename: {filename}")
            continue

        question_id, date_str, forecaster_id, feedback_type = parts
        feedback_type = feedback_type.replace('_', '-')  # normalize old->new

        path = os.path.join(dirpath, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if feedback_type == 'both-best-worst-feedback':
            both_best_worst_feedback[question_id][forecaster_id] = data
        elif feedback_type == 'single-best-feedback':
            single_best_feedback[question_id][forecaster_id] = data

        else:
            # print(f"Skipping unknown feedback type in {filename}: {feedback_type}")
            continue

    return both_best_worst_feedback, single_best_feedback


# --------------------------------
# 2) Flatten to a single DataFrame
# --------------------------------
def dicts_to_dataframe(
    both: Dict[str, Dict[str, Any]],
    single_best: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def push(dct: Dict[str, Dict[str, Any]], feedback_type_label: str):
        for qid, inner in dct.items():
            for fid, payload in inner.items():
                row = {
                    "question_id": payload.get("question_id", qid),
                    "superforecaster_id": payload.get("superforecaster_id", fid),
                    "resolution_date": payload.get("resolution_date"),
                    "forecast_due_date": payload.get("forecast_due_date"),
                    "resolution": payload.get("resolution"),
                    "original_forecast": payload.get("original_forecast"),
                    "updated_forecast": payload.get("updated_forecast"),
                    "improvement": payload.get("improvement"),
                    "original_error": payload.get("original_error"),
                    "updated_error": payload.get("updated_error"),
                    "feedback_type": feedback_type_label,
                }
                # optional nested best/worst forecasts (may be missing)
                best = payload.get("best_superforecast") or {}
                worst = payload.get("worst_superforecast") or {}
                row["best_forecast"] = best.get("forecast")
                row["worst_forecast"] = worst.get("forecast")
                rows.append(row)

    push(both, "both-best-worst-feedback")
    push(single_best, "single-best-feedback")

    df = pd.DataFrame(rows)

    # Parse dates if present
    for col in ["resolution_date", "forecast_due_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure numeric columns
    for col in [
        "resolution", "original_forecast", "updated_forecast",
        "improvement", "original_error", "updated_error",
        "best_forecast", "worst_forecast"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If improvement is missing but forecasts exist, compute it
    if "improvement" in df.columns:
        needs_impr = df["improvement"].isna()
        if "updated_forecast" in df.columns and "original_forecast" in df.columns:
            df.loc[needs_impr, "improvement"] = (
                df.loc[needs_impr, "updated_forecast"] - df.loc[needs_impr, "original_forecast"]
            )

    return df


# --------------------------
# 3) Plotting helper utils
# --------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def add_diag(ax):
    # add y=x diagonal within current axes limits
    lims = [
        np.nanmin([ax.get_xlim(), ax.get_ylim()]),
        np.nanmax([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

# --- Helpers for per-type suites and cross-type comparisons ---

def _add_diag(ax):
    lims = [
        np.nanmin([ax.get_xlim(), ax.get_ylim()]),
        np.nanmax([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_suite_for(df_sub: pd.DataFrame, tag: str, out_dir: str):
    """Produce the per-dict plots for a subset df_sub; tag is 'both' or 'single'."""
    _ensure_dir(out_dir)

    # 1) Improvement distribution
    if "improvement" in df_sub:
        fig = plt.figure()
        plt.hist(df_sub["improvement"].dropna(), bins=30)
        plt.title(f"[{tag}] Distribution of Improvement")
        plt.xlabel("Improvement")
        plt.ylabel("Count")
        fig.savefig(os.path.join(out_dir, f"{tag}_01_improvement_hist.png"), dpi=150)
        plt.close(fig)

    # 2) Error reduction
    if {"original_error", "updated_error"} <= set(df_sub.columns):
        sub = df_sub[["original_error", "updated_error"]].dropna()
        if not sub.empty:
            fig = plt.figure()
            plt.scatter(sub["original_error"], sub["updated_error"], s=12)
            _add_diag(plt.gca())
            plt.title(f"[{tag}] Error Reduction")
            plt.xlabel("Original Error")
            plt.ylabel("Updated Error")
            fig.savefig(os.path.join(out_dir, f"{tag}_02_error_reduction.png"), dpi=150)
            plt.close(fig)

    # 3) Forecast shift
    if {"original_forecast", "updated_forecast"} <= set(df_sub.columns):
        sub = df_sub[["original_forecast", "updated_forecast"]].dropna()
        if not sub.empty:
            fig = plt.figure()
            plt.scatter(sub["original_forecast"], sub["updated_forecast"], s=12)
            _add_diag(plt.gca())
            plt.title(f"[{tag}] Original vs Updated Forecast")
            plt.xlabel("Original Forecast")
            plt.ylabel("Updated Forecast")
            fig.savefig(os.path.join(out_dir, f"{tag}_03_forecast_shift.png"), dpi=150)
            plt.close(fig)

    # E) Side-by-side boxplot: improvement by topic
    if "improvement" in df_sub and "question_id" in df_sub:
        topics = []
        for qid in df_sub["question_id"]:
            try:
                topics.append(loader.get_topic(qid))
            except Exception:
                topics.append("Unknown")
        df_sub = df_sub.copy()
        df_sub["topic"] = topics

        fig = plt.figure(figsize=(max(6, len(df_sub["topic"].unique()) * 0.8), 5))
        topic_order = sorted(df_sub["topic"].unique())
        positions = []
        box_data = []
        labels = []
        width = 0.35  # half-width for side-by-side boxes

        for i, topic in enumerate(topic_order):
            sub_both = df_sub[(df_sub["topic"] == topic) & (df_sub["feedback_type"] == "both-best-worst-feedback")]["improvement"].dropna()
            sub_single_best = df_sub[(df_sub["topic"] == topic) & (df_sub["feedback_type"] == "single-best-feedback")]["improvement"].dropna()
            if sub_both.empty and sub_single_best.empty:
                continue
            # Positioning: both on left, single on right
            positions.extend([i - width/2, i + width/2])
            box_data.extend([sub_both, sub_single_best])
            labels.extend([f"{topic}\n(both)", f"{topic}\n(single best)"])

        plt.boxplot(box_data, positions=positions, widths=width, showfliers=False)
        plt.xticks(range(len(topic_order)), topic_order, rotation=45, ha="right")
        plt.title("Improvement by Topic and Feedback Type")
        plt.ylabel("Improvement")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{tag}_05_improvement_by_topic.png"), dpi=150)
        plt.close(fig)

    # 5) Updated forecast by resolution
    if {"resolution", "updated_forecast"} <= set(df_sub.columns):
        fig = plt.figure()
        true_mask = df_sub["resolution"] == 1
        false_mask = df_sub["resolution"] == 0
        plt.hist(df_sub.loc[true_mask, "updated_forecast"].dropna(), bins=30, alpha=0.5, label="Resolved = 1")
        plt.hist(df_sub.loc[false_mask, "updated_forecast"].dropna(), bins=30, alpha=0.5, label="Resolved = 0")
        plt.title(f"[{tag}] Updated Forecasts by Resolution")
        plt.xlabel("Updated Forecast")
        plt.ylabel("Count")
        plt.legend()
        fig.savefig(os.path.join(out_dir, f"{tag}_05_updated_by_resolution.png"), dpi=150)
        plt.close(fig)

    # 6) Best-vs-worst spread
    if {"best_forecast", "worst_forecast"} <= set(df_sub.columns):
        spread = (df_sub["worst_forecast"] - df_sub["best_forecast"]).dropna()
        if not spread.empty:
            fig = plt.figure()
            plt.hist(spread, bins=30)
            plt.title(f"[{tag}] Best vs Worst Superforecast Spread (worst - best)")
            plt.xlabel("Spread")
            plt.ylabel("Count")
            fig.savefig(os.path.join(out_dir, f"{tag}_06_best_worst_spread.png"), dpi=150)
            plt.close(fig)

    # 7) Time trend (weekly mean improvement)
    if {"forecast_due_date", "improvement"} <= set(df_sub.columns):
        sub = df_sub[["forecast_due_date", "improvement"]].dropna()
        if not sub.empty:
            sub = sub.sort_values("forecast_due_date").set_index("forecast_due_date").resample("W").mean(numeric_only=True)
            if not sub.empty:
                fig = plt.figure()
                plt.plot(sub.index, sub["improvement"])
                plt.title(f"[{tag}] Average Improvement Over Time (Weekly)")
                plt.xlabel("Week")
                plt.ylabel("Mean Improvement")
                fig.autofmt_xdate()
                fig.savefig(os.path.join(out_dir, f"{tag}_07_improvement_time_trend.png"), dpi=150)
                plt.close(fig)

def plot_comparisons(df: pd.DataFrame, out_dir: str):
    """Direct comparisons between both dicts (overlay/side-by-side)."""
    _ensure_dir(out_dir)
    a = df[df["feedback_type"] == "both-best-worst-feedback"].copy()
    b = df[df["feedback_type"] == "single-best-feedback"].copy()

    # A) Overlaid hist: improvement
    if "improvement" in df:
        fig = plt.figure()
        plt.hist(a["improvement"].dropna(), bins=30, alpha=0.5, label="both-best-worst")
        plt.hist(b["improvement"].dropna(), bins=30, alpha=0.5, label="single-best")
        plt.title("Improvement Distribution Comparison")
        plt.xlabel("Improvement")
        plt.ylabel("Count")
        plt.legend()
        fig.savefig(os.path.join(out_dir, "cmp_01_improvement_hist_overlay.png"), dpi=150)
        plt.close(fig)

    # B) ECDF of improvement (lines)
    def ecdf(x):
        x = np.sort(np.asarray(x))
        y = np.arange(1, len(x)+1) / len(x) if len(x) else np.array([])
        return x, y

    ax = a["improvement"].dropna()
    bx = b["improvement"].dropna()
    if len(ax) and len(bx):
        fig = plt.figure()
        x1, y1 = ecdf(ax)
        x2, y2 = ecdf(bx)
        plt.plot(x1, y1, label="both-best-worst")
        plt.plot(x2, y2, label="single-best")
        plt.title("Improvement ECDF Comparison")
        plt.xlabel("Improvement")
        plt.ylabel("ECDF")
        plt.legend()
        fig.savefig(os.path.join(out_dir, "cmp_02_improvement_ecdf.png"), dpi=150)
        plt.close(fig)

    # C) Error reduction scatter overlay
    needed = {"original_error", "updated_error"}
    if needed <= set(df.columns):
        fig = plt.figure()
        sub_a = a[list(needed)].dropna()
        sub_b = b[list(needed)].dropna()
        if not sub_a.empty:
            plt.scatter(sub_a["original_error"], sub_a["updated_error"], s=12, label="both-best-worst")
        if not sub_b.empty:
            plt.scatter(sub_b["original_error"], sub_b["updated_error"], s=12, label="single-best")
        _add_diag(plt.gca())
        plt.title("Error Reduction Comparison")
        plt.xlabel("Original Error")
        plt.ylabel("Updated Error")
        plt.legend()
        fig.savefig(os.path.join(out_dir, "cmp_03_error_reduction_overlay.png"), dpi=150)
        plt.close(fig)

    # D) Time trend overlay (weekly mean improvement)
    if {"forecast_due_date", "improvement"} <= set(df.columns):
        fig = plt.figure()
        plotted = False
        for label, sub in [("both-best-worst", a), ("single-best", b)]:
            sub = sub[["forecast_due_date", "improvement"]].dropna()
            if not sub.empty:
                sub = sub.sort_values("forecast_due_date").set_index("forecast_due_date").resample("W").mean(numeric_only=True)
                if not sub.empty:
                    plt.plot(sub.index, sub["improvement"], label=label)
                    plotted = True
        if plotted:
            plt.title("Average Improvement Over Time (Weekly) — Comparison")
            plt.xlabel("Week")
            plt.ylabel("Mean Improvement")
            plt.legend()
            fig.autofmt_xdate()
            fig.savefig(os.path.join(out_dir, "cmp_04_time_trend_overlay.png"), dpi=150)
        plt.close(fig)

    # E) Side-by-side boxplot: improvement
    if "improvement" in df:
        fig = plt.figure()
        groups = [a["improvement"].dropna().values, b["improvement"].dropna().values]
        labels = ["both-best-worst", "single-best"]
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.title("Improvement by Feedback Type")
        plt.xlabel("Feedback Type")
        plt.ylabel("Improvement")
        fig.savefig(os.path.join(out_dir, "cmp_05_improvement_boxplot.png"), dpi=150)
        plt.close(fig)

# --------------------------
# 4) Plotting entry-point
# --------------------------
def make_plots(results_path: str, out_dir: str = "results_ablation_forecasts_plots"):
    both, single_best = load_results(results_path)
    df = dicts_to_dataframe(both, single_best)

    # Per-dict suites
    plot_suite_for(df[df["feedback_type"] == "both-best-worst-feedback"], tag="both", out_dir=out_dir)
    plot_suite_for(df[df["feedback_type"] == "single-best-feedback"],    tag="single", out_dir=out_dir)

    # Direct comparisons
    plot_comparisons(df, out_dir=out_dir)
    ensure_dir(out_dir)

    # ---- 1) Distribution of improvement ----
    fig = plt.figure()
    series = df["improvement"].dropna()
    plt.hist(series, bins=30)
    plt.title("Distribution of Improvement (updated - original)")
    plt.xlabel("Improvement")
    plt.ylabel("Count")
    fig.savefig(os.path.join(out_dir, "01_improvement_hist.png"), dpi=150)
    plt.close(fig)

    # ---- 2) Error reduction: original_error vs updated_error ----
    fig = plt.figure()
    sub = df[["original_error", "updated_error"]].dropna()
    plt.scatter(sub["original_error"], sub["updated_error"], s=12)
    add_diag(plt.gca())
    plt.title("Error Reduction")
    plt.xlabel("Original Error")
    plt.ylabel("Updated Error")
    fig.savefig(os.path.join(out_dir, "02_error_reduction_scatter.png"), dpi=150)
    plt.close(fig)

    # ---- 3) Calibration shift: original_forecast vs updated_forecast ----
    fig = plt.figure()
    sub = df[["original_forecast", "updated_forecast"]].dropna()
    plt.scatter(sub["original_forecast"], sub["updated_forecast"], s=12)
    add_diag(plt.gca())
    plt.title("Original vs Updated Forecast")
    plt.xlabel("Original Forecast")
    plt.ylabel("Updated Forecast")
    fig.savefig(os.path.join(out_dir, "03_forecast_shift_scatter.png"), dpi=150)
    plt.close(fig)

    # ---- 4) Improvement by question (box plots for top-N questions by count) ----
    # choose up to 20 most frequent questions for readability
    q_counts = df["question_id"].value_counts().head(20).index.tolist()
    sub = df[df["question_id"].isin(q_counts)]
    groups = [grp["improvement"].dropna().values for _, grp in sub.groupby("question_id")]
    labels = [str(q)[:10] + "…" if len(str(q)) > 11 else str(q) for q in sub.groupby("question_id").groups.keys()]

    if len(groups) >= 2:
        fig = plt.figure()
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.title("Improvement by Question (Top 20 by count)")
        plt.xlabel("Question ID")
        plt.ylabel("Improvement")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "04_improvement_by_question_boxplot.png"), dpi=150)
        plt.close(fig)


    print(f"Saved plots to: {os.path.abspath(out_dir)}")


# --------------------------
# 5) Example usage
# --------------------------
if __name__ == "__main__":
    # Set your folder path containing the JSONs
    results_path = "results_ablation_forecasts"
    make_plots(results_path, out_dir="results_ablation_forecasts_plots")
