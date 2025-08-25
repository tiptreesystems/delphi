import json
from typing import Dict, Any


def save_delphi_log(delphi_log: Dict[str, Any], output_file: str):
    """Save the Delphi log to a JSON file."""
    try:
        with open(output_file, "w") as f:
            json.dump(delphi_log, f, indent=2)
    except TypeError as e:
        print(f"JSON serialization error: {e}")

        for key, value in delphi_log.items():
            try:
                json.dumps(value)
                print(f"  {key}: OK")
            except TypeError as sub_e:
                print(f"  {key}: ERROR - {sub_e}")
                if key == "rounds":
                    for i, round_data in enumerate(value):
                        try:
                            json.dumps(round_data)
                            print(f"    round {i}: OK")
                        except TypeError as round_e:
                            print(f"    round {i}: ERROR - {round_e}")
        raise e

    print(f"Delphi log saved to {output_file}")
