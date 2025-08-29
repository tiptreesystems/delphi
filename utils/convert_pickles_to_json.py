import pickle
import json
from pathlib import Path

def convert_pkl_to_json(pkl_path: str, json_path: str) -> None:
    """
    Convert a pickle file containing a list of dicts into a JSON file.
    For the 'examples_used' key, only keep the Question.id values.

    Parameters
    ----------
    pkl_path : str
        Path to the pickle file.
    json_path : str
        Path to save the JSON file.
    """
    # Load pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Transform
    converted = []
    for entry in data:
        new_entry = entry.copy()
        if "examples_used" in new_entry:
            # Replace list of (Question, Forecast) tuples with list of Question.id
            new_entry["examples_used"] = [
                q.id for (q, _forecast) in new_entry["examples_used"]
            ]
        converted.append(new_entry)

    # Save as JSON
    with open(json_path, "w") as f:
        json.dump(converted, f, indent=2)



def batch_convert_pickles(input_dir: str, output_dir: str) -> None:
    """
    Convert all pickle files in a directory to JSON using convert_pkl_to_json.
    Saves outputs into a parallel directory.

    Parameters
    ----------
    input_dir : str
        Directory containing .pkl files.
    output_dir : str
        Directory to save converted .json files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for pkl_file in input_path.glob("*.pkl"):
        json_file = output_path / (pkl_file.stem + ".json")
        convert_pkl_to_json(str(pkl_file), str(json_file))
        print(f"Converted {pkl_file.name} -> {json_file.name}")


def crawl_and_convert_outputs_initial_forecasts(root: str | None = None) -> None:
    """
    Search recursively (starting at `root` or CWD) for directories whose names
    contain 'outputs_initial_forecasts'. For each such directory, create a
    sibling directory with a '_json' suffix and run batch_convert_pickles on it.
    """
    base = Path(root) if root else Path.cwd()

    # Snapshot directories first so we don't traverse ones we create.
    candidate_dirs = [p for p in base.rglob("*") if p.is_dir()]

    for d in candidate_dirs:
        name = d.name
        if "outputs_initial_forecasts" not in name:
            continue
        if name.endswith("_json"):
            # Skip already-converted targets
            continue

        # Only proceed if there are .pkl files inside
        if not any(d.glob("*.pkl")):
            continue

        out_dir = d.with_name(f"{name}_json")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Converting pickles in: {d} -> {out_dir}")
        batch_convert_pickles(str(d), str(out_dir))

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Convert Delphi pickle outputs to JSON format."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory to start searching for 'outputs_initial_forecasts' directories. Defaults to current working directory.",
    )
    args = parser.parse_args()

    crawl_and_convert_outputs_initial_forecasts(os.getcwd() if args.root is None else args.root)