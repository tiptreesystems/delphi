from pathlib import Path
import json
import pickle
import argparse


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert pickle files to JSON.")
    parser.add_argument("input_dir", type=str, help="Directory containing .pkl files.")
    parser.add_argument("output_dir", type=str, help="Directory to save .json files.")
    args = parser.parse_args()

    batch_convert_pickles(args.input_dir, args.output_dir)

    ## Example call
    # python batch_convert_to_json.py /path/to/input_dir /path/to/output_dir
