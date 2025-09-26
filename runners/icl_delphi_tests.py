import argparse
import os
import json as json
from datetime import datetime
from matplotlib.pylab import seed
import numpy as np

from dotenv import load_dotenv

from dataset.dataloader import ForecastDataLoader
from runners.delphi_runner import initialize_experts, run_delphi_rounds, select_experts
from utils.forecast_loader import load_forecasts
from utils.logs import save_delphi_log
from utils.llm_config import get_llm_from_config
from utils.config_types import RootConfig, load_typed_experiment_config
from utils.utils import setup_environment
from utils.sampling import (
    EVALUATION_QUESTION_IDS,
    EVOLUTION_EVALUATION_QUESTION_IDS,
)
import asyncio
import yaml
import tempfile


load_dotenv()


def generate_temp_config(args):
    with open(args.config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    if "experiment" not in config_data:
        config_data["experiment"] = {}
    if args.seed is not None:
        config_data["experiment"]["seed"] = args.seed
    # Save the modified config to a temporary file
    tmp_config = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".yaml")
    yaml.safe_dump(config_data, tmp_config)
    tmp_config.close()
    return tmp_config


def parse_subdirs(output_dir, names=["eval", "evolution_eval"]):
    set_subdirs = []
    for name in names:
        p = os.path.join(output_dir, name)
        if os.path.isdir(p):
            set_subdirs.append(p)
    if not set_subdirs:
        set_subdirs = [output_dir]
    return set_subdirs


def get_valid_qids(sampling_method):
    valid_qids = None
    if sampling_method == "evaluation":
        valid_qids = set(EVALUATION_QUESTION_IDS)
    elif sampling_method == "evolution_evaluation":
        valid_qids = set(EVOLUTION_EVALUATION_QUESTION_IDS)
    return valid_qids


async def run_delphi_experiment(config: RootConfig):
    """Run the Delphi experiment based on configuration."""
    # Setup environment
    setup_environment(config)

    llm = get_llm_from_config(config, role="expert")
    loader = ForecastDataLoader()

    # Load forecasts
    questions, llmcasts_by_qid_sfid, example_pairs_by_qid_sfid = await load_forecasts(
        config, loader, llm
    )

    # Get configuration values
    # Seed-scoped output directory: <base>/seed_<seed>
    output_dir = config.experiment.output_dir
    output_dir = os.path.join(output_dir, f"seed_{int(config.experiment.seed)}")

    # Detect set subdirs (eval/evolution_eval) to support organized outputs
    question_set_subdirs = parse_subdirs(output_dir)

    # Choose save_dir: if set subdirs exist, route to one matching sampling method
    save_dir = get_save_dir(
        config.data.sampling.method, output_dir, question_set_subdirs
    )
    os.makedirs(save_dir, exist_ok=True)

    # Process each question
    for question in questions:
        output_filename = config.output.file_pattern.format(
            question_id=question.id, resolution_date=config.data.resolution_date
        )

        # Check for existing logs across set subdirs (if present)
        if config.processing.skip_existing:
            exists_anywhere = any(
                os.path.exists(os.path.join(d, output_filename))
                for d in question_set_subdirs
            )
            if exists_anywhere:
                print(f"Skipping existing log for {question.id} at {output_filename}")
                continue

        llmcasts_by_sfid = llmcasts_by_qid_sfid.get(question.id, {})
        if not llmcasts_by_sfid:
            print(f"No forecasts found for question {question.id}, skipping")
            continue
        example_pairs = example_pairs_by_qid_sfid.get(question.id, {})

        # Initialize and select experts
        experts = initialize_experts(llmcasts_by_sfid, config, llm)
        experts = select_experts(experts, config)

        print(f"Running Delphi for question {question.id} with {len(experts)} experts")

        # Run Delphi rounds
        delphi_log = await run_delphi_rounds(question, experts, config, example_pairs)

        # Save the log
        save_delphi_log(delphi_log, os.path.join(save_dir, output_filename))


def get_save_dir(sampling_method, output_dir, question_set_subdirs):
    save_dir = output_dir
    if question_set_subdirs:
        target = None
        if sampling_method == "evaluation":
            target = "eval"
        elif sampling_method == "evolution_evaluation":
            target = "evolution_eval"
        if target is not None and os.path.isdir(os.path.join(output_dir, target)):
            save_dir = os.path.join(output_dir, target)
        else:
            save_dir = question_set_subdirs[0]
    return save_dir


async def compute_and_save_metrics(config: RootConfig):
    # After generation, compute metrics by scanning logs matching the sampling method's question set

    output_dir = config.experiment.output_dir
    output_dir = os.path.join(output_dir, f"seed_{int(config.experiment.seed)}")
    loader = ForecastDataLoader()

    per_question = []
    sampling_method = config.data.sampling.method
    valid_qids = get_valid_qids(sampling_method)

    # Prefer scanning set subdirectories (eval/evolution_eval) if they exist
    scan_dirs = parse_subdirs(output_dir)

    for scan_dir in scan_dirs:
        for fname in sorted(os.listdir(scan_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(scan_dir, fname), "r", encoding="utf-8") as f:
                log = json.load(f)

            qid = log.get("question_id") or None

            # Sampling-method alignment: only include question IDs that belong to the configured set
            if qid not in valid_qids:
                continue

            # Determine resolution date preference: from log, else config
            log_res_date = (log.get("config") or {}).get("resolution_date")
            res_date = log_res_date or config.data.resolution_date
            outcome = loader.get_resolution(qid, res_date).resolved_to

            rounds = log.get("rounds") or []
            final_round = rounds[-1]
            experts_map = final_round.get("experts") or {}
            final_probs = [
                float(e.get("prob"))
                for e in experts_map.values()
                if isinstance(e.get("prob"), (int, float))
            ]
            delphi_pred = float(np.median(final_probs)) if final_probs else np.nan
            delphi_brier = (delphi_pred - outcome) ** 2 if final_probs else np.nan

            # Baselines
            try:
                sf_forecasts = loader.get_super_forecasts(
                    question_id=qid, resolution_date=res_date
                )
                sf_vals = [float(f.forecast) for f in sf_forecasts]
                sf_median = float(np.median(sf_vals)) if sf_vals else np.nan
                sf_brier = (sf_median - outcome) ** 2 if sf_vals else np.nan
            except Exception:
                sf_median = np.nan
                sf_brier = np.nan

            try:
                public_forecasts = loader.get_public_forecasts(
                    question_id=qid, resolution_date=res_date
                )
                pub_vals = [float(f.forecast) for f in public_forecasts]
                public_median = float(np.median(pub_vals)) if pub_vals else np.nan
                public_brier = (public_median - outcome) ** 2 if pub_vals else np.nan
            except Exception:
                public_median = np.nan
                public_brier = np.nan

            # Attempt to fetch topic
            try:
                qobj = loader.get_question(qid)
                topic = getattr(qobj, "topic", None)
            except Exception:
                topic = None

            per_question.append(
                {
                    "question_id": qid,
                    "topic": topic,
                    "actual": outcome,
                    "delphi_pred": delphi_pred,
                    "delphi_brier": delphi_brier,
                    "sf_median": sf_median,
                    "sf_brier": sf_brier,
                    "public_median": public_median,
                    "public_brier": public_brier,
                }
            )

    # Aggregate metrics and save summary
    if per_question:

        def _nanmean(vals):
            arr = np.array(
                [
                    v
                    for v in vals
                    if v is not None and not (isinstance(v, float) and (np.isnan(v)))
                ]
            )
            return float(arr.mean()) if arr.size else None

        def _nanmedian(vals):
            arr = np.array(
                [
                    v
                    for v in vals
                    if v is not None and not (isinstance(v, float) and (np.isnan(v)))
                ]
            )
            return float(np.median(arr)) if arr.size else None

        n = len(per_question)
        delphi_briers = [pq["delphi_brier"] for pq in per_question]
        sf_briers = [pq["sf_brier"] for pq in per_question]
        public_briers = [pq["public_brier"] for pq in per_question]
        delphi_preds = [pq["delphi_pred"] for pq in per_question]
        actuals = [pq["actual"] for pq in per_question]
        abs_errors = [
            abs(p - a)
            for p, a in zip(delphi_preds, actuals)
            if not (np.isnan(p) or np.isnan(a))
        ]

        summary = {
            "num_questions": n,
            "mean_brier": _nanmean(delphi_briers),
            "median_brier": _nanmedian(delphi_briers),
            "sf_mean_brier": _nanmean(sf_briers),
            "public_mean_brier": _nanmean(public_briers),
            "mean_abs_error": _nanmean(abs_errors),
            "questions_evaluated": [pq["question_id"] for pq in per_question],
        }

        print("\nDelphi Evaluation Summary:")
        print(f"  Questions evaluated: {summary['num_questions']}")
        if summary["mean_abs_error"] is not None:
            print(f"  Mean absolute error: {summary['mean_abs_error']:.3f}")
        if summary["mean_brier"] is not None:
            print(f"  Delphi mean Brier: {summary['mean_brier']:.3f}")
        if summary["sf_mean_brier"] is not None:
            print(f"  Superforecaster median Brier: {summary['sf_mean_brier']:.3f}")
        if summary["public_mean_brier"] is not None:
            print(f"  Public median Brier: {summary['public_mean_brier']:.3f}")

        # Save alongside logs
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(output_dir, f"delphi_eval_summary_{ts}.json")
        try:
            with open(metrics_path, "w", encoding="utf-8") as mf:
                json.dump(
                    {"summary": summary, "per_question": per_question}, mf, indent=2
                )
            print(f"  Saved summary: {metrics_path}")
        except Exception as e:
            print(f"  Warning: failed to write summary: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Delphi experiment")
    parser.add_argument(
        "config_path", help="Path to experiment configuration YAML file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to overwrite experiment.seed in config",
        default=None,
    )
    args = parser.parse_args()

    tmp_config = generate_temp_config(args)

    args.config_path = tmp_config.name

    # Load configuration (typed + raw for legacy helpers)
    typed_config = load_typed_experiment_config(args.config_path)

    # Run experiment
    asyncio.run(run_delphi_experiment(typed_config))
    # Compute and save metrics
    asyncio.run(compute_and_save_metrics(typed_config))
