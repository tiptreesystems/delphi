# plot_metric_script.py
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import argparse
import re

import debugpy

print("Starting debugpy...")
debugpy.listen(5679)  # Adjust port as needed
debugpy.wait_for_client()  # Wait for the debugger to attach


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def load_parsed(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_question_prompt_table(
    data: Dict[str, Any],
    metric_key: str = "fitness",
    section: str = "evolution",
) -> List[Tuple[int, str, str, float]]:
    """Rows: (generation, question, prompt, metric_value).

    section:
      - "evolution": uses data["evolution"]["generations"][gen]
      - "validation" or "evaluation": uses data[section] and assigns generation=0
    """
    rows: List[Tuple[int, str, str, float]] = []

    def _norm_prompt_label(s: str) -> str:
        # cand_{i}_{hash}
        m = re.search(r"cand_(\d{1,2})_([0-9a-fA-F]{6,})", s)
        if m:
            return f"cand_{int(m.group(1)):02d}_{m.group(2)[:8]}"
        # evolved_g{gen}_i{i}_{hash}
        m = re.search(r"evolved_g(\d+)_i(\d+)_([0-9a-fA-F]{6,})", s)
        if m:
            return f"i{int(m.group(2)):02d}_{m.group(3)[:8]}"
        # best_evolved_g{gen}_{hash}
        m = re.search(r"best_evolved_g(\d+)_([0-9a-fA-F]{6,})", s)
        if m:
            return f"best_{m.group(2)[:8]}"
        # Otherwise, fall back to a trimmed snippet of the prompt text/path
        s = s.strip()
        return (s[:60] + "…") if len(s) > 60 else s

    if section == "evolution":
        generations = data.get("evolution", {}).get("generations", {})
        for gen_key, records in generations.items():
            try:
                gen = int(gen_key)
            except Exception:
                continue
            for rec in records or []:
                prompt = rec.get("prompt")
                question = rec.get("question")
                val_raw = rec.get(metric_key)
                if not (prompt and question and val_raw is not None):
                    continue
                val = _safe_float(val_raw)
                if math.isnan(val):
                    continue
                rows.append((gen, question, _norm_prompt_label(prompt), val))
    else:
        # validation/evaluation are flat lists
        records = data.get(section, []) or []
        for rec in records:
            prompt = rec.get("prompt")
            question = rec.get("question")
            val_raw = rec.get(metric_key)
            if not (prompt and question and val_raw is not None):
                continue
            val = _safe_float(val_raw)
            if math.isnan(val):
                continue
            rows.append((0, question, _norm_prompt_label(prompt), val))
    rows.sort(key=lambda r: (r[1], r[0], r[2]))
    return rows


def compute_global_ylim(rows: List[Tuple[int, str, str, float]]) -> Tuple[float, float]:
    vals = [v for (_, _, _, v) in rows]
    if not vals:
        return (0, 1)
    ymin, ymax = min(vals), max(vals)
    if ymin == ymax:
        # expand slightly if constant
        ymin -= 0.1 * abs(ymin) if ymin != 0 else -0.1
        ymax += 0.1 * abs(ymax) if ymax != 0 else 0.1
    return ymin, ymax


def plot_metric_by_question(
    rows: List[Tuple[int, str, str, float]],
    *,
    metric_name: str,
    top_n_prompts: int | None = 8,
    figsize_per_question=(6, 4),
    out_dir: Path | None = None,
):
    """One subplot per question; x=gen, y=metric, line=prompt."""
    by_question: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for gen, q, prompt, val in rows:
        by_question.setdefault(q, {}).setdefault(prompt, []).append((gen, val))

    ymin, ymax = compute_global_ylim(rows)

    n_q = len(by_question)
    ncols = 2 if n_q > 1 else 1
    nrows = math.ceil(n_q / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_question[0] * ncols, figsize_per_question[1] * nrows),
        squeeze=False,
    )

    for ax, (question, prompts) in zip(axes.flat, by_question.items()):
        if top_n_prompts is not None and len(prompts) > top_n_prompts:
            ranked = sorted(prompts.items(), key=lambda kv: kv[1][-1][1], reverse=True)
            prompts = dict(ranked[:top_n_prompts])

        for prompt, series in prompts.items():
            series.sort(key=lambda t: t[0])
            gens = [g for g, _ in series]
            vals = [v for _, v in series]
            ax.plot(gens, vals, marker="o", label=prompt)

        ax.set_title(f"Question: {question}")
        ax.set_xlabel("Generation")
        ax.set_ylabel(metric_name)
        ax.set_ylim(ymin, ymax)  # normalize across all subplots
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize="small")

    for ax in axes.flat[len(by_question) :]:
        ax.axis("off")

    plt.tight_layout()
    out_path = (
        (out_dir / f"{metric_name}_by_question.png")
        if out_dir
        else Path(f"{metric_name}_by_question.png")
    )
    plt.savefig(out_path)


def plot_metric_by_prompt(
    rows: List[Tuple[int, str, str, float]],
    *,
    metric_name: str,
    top_n_questions: int | None = 8,
    figsize_per_prompt=(6, 4),
    out_dir: Path | None = None,
):
    """One subplot per prompt; x=gen, y=metric, line=question."""
    by_prompt: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for gen, q, prompt, val in rows:
        by_prompt.setdefault(prompt, {}).setdefault(q, []).append((gen, val))

    ymin, ymax = compute_global_ylim(rows)

    n_p = len(by_prompt)
    ncols = 2 if n_p > 1 else 1
    nrows = math.ceil(n_p / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_prompt[0] * ncols, figsize_per_prompt[1] * nrows),
        squeeze=False,
    )

    for ax, (prompt, questions) in zip(axes.flat, by_prompt.items()):
        if top_n_questions is not None and len(questions) > top_n_questions:
            ranked = sorted(
                questions.items(), key=lambda kv: kv[1][-1][1], reverse=True
            )
            questions = dict(ranked[:top_n_questions])

        for q, series in questions.items():
            series.sort(key=lambda t: t[0])
            gens = [g for g, _ in series]
            vals = [v for _, v in series]
            ax.plot(gens, vals, marker="o", label=q)

        ax.set_title(f"Prompt: {prompt}")
        ax.set_xlabel("Generation")
        ax.set_ylabel(metric_name)
        ax.set_ylim(ymin, ymax)  # normalize across all subplots
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize="small")

    for ax in axes.flat[len(by_prompt) :]:
        ax.axis("off")

    plt.tight_layout()
    out_path = (
        (out_dir / f"{metric_name}_by_prompt.png")
        if out_dir
        else Path(f"{metric_name}_by_prompt.png")
    )
    plt.savefig(out_path)


def plot_metric_aggregate(
    rows: List[Tuple[int, str, str, float]],
    *,
    metric_name: str,
    top_n_prompts: int | None = 12,
    figsize=(10, 6),
    out_dir: Path | None = None,
):
    """One plot across all questions, average metric per (prompt, generation)."""
    per_prompt: Dict[str, Dict[int, List[float]]] = {}
    # Drop any rows with "DEBUG" in question or prompt
    rows = [row for row in rows if "DEBUG" not in row[1] and "DEBUG" not in row[2]]
    for gen, q, prompt, val in rows:
        per_prompt.setdefault(prompt, {}).setdefault(gen, []).append(val)

    prompt_series: Dict[str, List[Tuple[int, float]]] = {}
    for prompt, gens in per_prompt.items():
        series = []
        for gen, vals in gens.items():
            series.append((gen, sum(vals) / len(vals)))
        series.sort(key=lambda t: t[0])
        prompt_series[prompt] = series

    if top_n_prompts is not None and len(prompt_series) > top_n_prompts:
        ranked = sorted(
            prompt_series.items(), key=lambda kv: kv[1][-1][1], reverse=True
        )
        prompt_series = dict(ranked[:top_n_prompts])

    plt.figure(figsize=figsize)
    for prompt, series in prompt_series.items():
        gens = [g for g, _ in series]
        vals = [v for _, v in series]
        plt.plot(gens, vals, marker="o", linewidth=2, label=prompt)

    plt.xlabel("Generation")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} averaged over questions")
    plt.grid(True, linestyle="--", alpha=0.4)
    loc = "best" if len(prompt_series) <= 8 else "center left"
    bbox = None if len(prompt_series) <= 8 else (1.02, 0.5)
    plt.legend(title="Prompt", loc=loc, bbox_to_anchor=bbox)
    plt.tight_layout()
    out_path = (
        (out_dir / f"{metric_name}_aggregate.png")
        if out_dir
        else Path(f"{metric_name}_aggregate.png")
    )
    plt.savefig(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot prompt metric across generations from parse_log JSON."
    )
    parser.add_argument(
        "--json", required=True, type=str, help="Path to parsed JSON file"
    )
    parser.add_argument(
        "--metric",
        default="fitness",
        choices=[
            "fitness",
            "improvement",
            "variance",
            "smoothness",
            "pred_prob",
            "superforecaster_median",
            "actual",
        ],
        help="Which record key to plot on the y-axis",
    )
    parser.add_argument(
        "--section",
        default="evolution",
        choices=["evolution", "validation", "evaluation"],
        help="Which section of the parsed JSON to use",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["per_question", "per_prompt", "aggregate"],
        help="Plotting mode",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    data = load_parsed(json_path)
    out_dir = json_path.parent
    rows = build_question_prompt_table(
        data, metric_key=args.metric, section=args.section
    )

    if args.mode == "per_question":
        plot_metric_by_question(
            rows, metric_name=f"{args.metric} ({args.section})", out_dir=out_dir
        )
    elif args.mode == "per_prompt":
        plot_metric_by_prompt(
            rows, metric_name=f"{args.metric} ({args.section})", out_dir=out_dir
        )
    elif args.mode == "aggregate":
        plot_metric_aggregate(
            rows, metric_name=f"{args.metric} ({args.section})", out_dir=out_dir
        )
