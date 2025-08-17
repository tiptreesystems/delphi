# plot_delphi_distribution.py
# Usage:
#   python plot_delphi_distribution.py --log /path/to/delphi_log_...json --out outputs/delphi_distribution.png
#
# Produces a violin + scatter plot showing the distribution of expert probabilities per round.

import argparse
import json
import os
import re
import yaml
from typing import List, Dict, Any, Optional
from dataset.dataloader import ForecastDataLoader

import numpy as np
import matplotlib.pyplot as plt

def parse_delphi_log_filename(filename: str, file_pattern: str):
    """
    Parse filename based on the config pattern like "groq_delphi_log_{question_id}_{resolution_date}.json"
    Returns (question_id, date).
    """
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    
    # Find the prefix before {question_id}
    prefix_end = file_pattern.find('{question_id}')
    if prefix_end == -1:
        raise ValueError(f"Pattern {file_pattern} must contain {{question_id}}")
    
    prefix = file_pattern[:prefix_end]
    
    # Remove the prefix from the filename
    if not stem.startswith(prefix):
        raise ValueError(f"Filename {filename} doesn't match pattern {file_pattern}")
    
    remainder = stem[len(prefix):]
    
    # Split by underscore and take the last part as date, the rest as question_id
    parts = remainder.split('_')
    if len(parts) < 2:
        raise ValueError(f"Cannot parse question_id and date from: {filename}")
    
    date_str = parts[-1]
    question_id = '_'.join(parts[:-1])
    
    return question_id, date_str


# Optional fallback in case some rounds only stored raw text responses
_PROB_PAT = re.compile(r'FINAL PROBABILITY:\s*(0?\.\d+|1\.0|0|1)', re.IGNORECASE)

def _extract_prob(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    matches = _PROB_PAT.findall(text)
    if matches:
        try:
            p = float(matches[-1])
            return max(0.0, min(1.0, p))
        except ValueError:
            pass
    # fallback: last bare number
    nums = re.findall(r'0?\.\d+|1\.0|0|1', text)
    if nums:
        try:
            p = float(nums[-1])
            return max(0.0, min(1.0, p))
        except ValueError:
            pass
    return None

def _collect_round_probs(delphi_log: Dict[str, Any]) -> Dict[int, List[float]]:
    """
    Returns {round_idx: [probabilities]} using stored numeric probs when available,
    falling back to parsing the response text if needed.
    """
    out: Dict[int, List[float]] = {}
    rounds = delphi_log.get("rounds", [])
    for r in rounds:
        r_idx = int(r.get("round", 0))
        expert_dict = r.get("experts", {})
        probs = []
        for sfid, entry in expert_dict.items():
            # Prefer stored numeric prob
            p = entry.get("prob")
            if isinstance(p, (int, float)):
                p = float(p)
            else:
                # Fallback to parse from response text
                p = _extract_prob(entry.get("response"))
            if p is not None:
                probs.append(max(0.0, min(1.0, p)))
        out[r_idx] = probs
    return dict(sorted(out.items(), key=lambda kv: kv[0]))

def plot_distribution_by_round(
    round_probs: Dict[int, List[float]],
    title: str = "Delphi: Distribution of Forecasts by Round",
    save_path: Optional[str] = None,
    show_points: bool = True,
    resolution = None
) -> None:
    """
    Makes a violin plot across rounds, with optional jittered points and per-round medians.
    """
    if not round_probs:
        raise ValueError("No round probabilities found to plot.")

    rounds = list(round_probs.keys())
    data = [round_probs[r] for r in rounds]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Violin plot of distributions
    parts = ax.violinplot(data, positions=rounds, showmeans=False, showextrema=False, showmedians=False)

    # Style violins lightly (no explicit colors per instruction; use default)
    for pc in parts['bodies']:
        pc.set_alpha(0.4)

    # Overlay per-round medians and IQR
    medians = [np.median(d) if len(d) else np.nan for d in data]
    q1 = [np.percentile(d, 25) if len(d) else np.nan for d in data]
    q3 = [np.percentile(d, 75) if len(d) else np.nan for d in data]

    ax.plot(rounds, medians, marker='o', linewidth=1.5, label='Median')
    ax.vlines(rounds, q1, q3, linewidth=2, label='IQR')

    # Optional jittered scatter of individual expert probs
    if show_points:
        rng = np.random.default_rng(42)
        for x, d in zip(rounds, data):
            if not d:
                continue
            jitter = (rng.random(len(d)) - 0.5) * 0.15  # small horizontal jitter
            ax.scatter(np.full(len(d), x) + jitter, d, s=18, alpha=0.7)

    if resolution:
        # The resolution is a probability between 0 and 1, so plot it as a horizontal line
        ax.axhline(resolution.resolved_to, color='red', linestyle='--', label=f"Resolution: {resolution.resolved_to:.2f}")

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Probability")
    ax.set_xticks(rounds)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='best')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

def load_experiment_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Could not parse YAML config file {config_path}: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Plot evolution of Delphi forecast distributions across rounds.")
    parser.add_argument("config_path", help="Path to experiment configuration YAML file")
    parser.add_argument("--log", required=True, help="Path to a delphi_log_...json file or directory containing JSON files.")
    parser.add_argument("--title", default=None, help="Custom plot title.")
    parser.add_argument("--no-points", action="store_true", help="Disable showing individual expert points.")
    args = parser.parse_args()
    
    # Load configuration
    config = load_experiment_config(args.config_path)
    output_dir = os.path.join(config['experiment']['output_dir'], "forecast_evolutions")
    
    # Check if args.log is a directory or file
    if os.path.isdir(args.log):
        # Process all JSON files in the directory
        file_pattern = config['output']['file_pattern']
        # Extract the prefix before {question_id} to match actual files
        prefix_end = file_pattern.find('{question_id}')
        if prefix_end != -1:
            pattern_prefix = file_pattern[:prefix_end]
        else:
            # Fallback: use the pattern without the template variables
            pattern_prefix = file_pattern.split('_')[0]
        
        json_files = [
            os.path.join(args.log, f)
            for f in os.listdir(args.log)
            if f.startswith(pattern_prefix) and f.endswith(".json")
        ]
        
        if not json_files:
            print(f"No matching JSON files found in directory: {args.log}")
            return
            
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            process_single_file(json_file, config, output_dir, args)
    else:
        # Process single file
        if not os.path.exists(args.log):
            print(f"Error: File not found: {args.log}")
            return
        process_single_file(args.log, config, output_dir, args)

def process_single_file(json_file_path, config, output_dir, args):
    """Process a single JSON file and create a plot."""
    try:
        with open(json_file_path, "r") as f:
            delphi_log = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON file {json_file_path}: {e}")
        return
    except Exception as e:
        print(f"Warning: Error reading file {json_file_path}: {e}")
        return

    # Build output directory and filename from the JSON log path
    json_basename = os.path.splitext(os.path.basename(json_file_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{json_basename}.png")

    try:
        round_probs = _collect_round_probs(delphi_log)
        
        if not round_probs:
            print(f"Warning: No round data found in {json_file_path}")
            return

        question_id, resolution_date = parse_delphi_log_filename(json_file_path, config['output']['file_pattern'])

        loader = ForecastDataLoader()
        resolution = loader.get_resolution(question_id=question_id, resolution_date=resolution_date)

        title = args.title or f"Delphi: Distribution of Forecasts by Round (Q={delphi_log.get('question_text', 'unknown')[:50]}...)"
        plot_distribution_by_round(
            round_probs,
            title=title,
            save_path=out_path,
            show_points=not args.no_points,
            resolution=resolution
        )

        print(f"Plot saved to {out_path}")
        
    except Exception as e:
        print(f"Warning: Could not create plot for {json_file_path}: {e}")
        return

if __name__ == "__main__":
    main()
