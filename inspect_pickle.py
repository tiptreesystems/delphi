#!/usr/bin/env python
"""
Script to compute median forecasts from all pickle files in a specified folder.
"""

import pickle
import sys
from pathlib import Path
import numpy as np

def load_pickle_forecasts(file_path):
    """Load a pickle file and extract forecasts."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract question_id and all forecasts
        question_id = None
        all_forecasts = []
        
        if isinstance(data, list):
            # Each item is a forecaster's data
            for item in data:
                if isinstance(item, dict):
                    if 'question_id' in item and question_id is None:
                        question_id = item['question_id']
                    if 'forecasts' in item:
                        forecasts = item['forecasts']
                        if isinstance(forecasts, list):
                            all_forecasts.extend(forecasts)
                        elif isinstance(forecasts, (int, float)):
                            all_forecasts.append(forecasts)
        elif isinstance(data, tuple):
            # Handle tuple format (less common)
            for item in data:
                if isinstance(item, dict):
                    if 'question_id' in item and question_id is None:
                        question_id = item['question_id']
                    if 'forecasts' in item:
                        forecasts = item['forecasts']
                        if isinstance(forecasts, dict):
                            all_forecasts.extend([v for v in forecasts.values() if isinstance(v, (int, float))])
                        elif isinstance(forecasts, list):
                            all_forecasts.extend(forecasts)
                        elif isinstance(forecasts, (int, float)):
                            all_forecasts.append(forecasts)
        elif isinstance(data, dict):
            # Single dict format
            if 'question_id' in data:
                question_id = data['question_id']
            if 'forecasts' in data:
                forecasts = data['forecasts']
                if isinstance(forecasts, dict):
                    all_forecasts = [v for v in forecasts.values() if isinstance(v, (int, float))]
                elif isinstance(forecasts, list):
                    all_forecasts = forecasts
                elif isinstance(forecasts, (int, float)):
                    all_forecasts = [forecasts]
        
        # Extract question_id from filename if not found in data
        if question_id is None:
            # Try to extract from filename pattern like: collected_fcasts_*_2025-07-21_QUESTIONID.pkl
            filename = file_path.stem
            parts = filename.split('_')
            # The question ID is typically after the date (YYYY-MM-DD pattern)
            for i, part in enumerate(parts):
                if len(part) == 4 and part.isdigit() and i+2 < len(parts):  # Found year
                    if len(parts[i+1]) == 2 and parts[i+1].isdigit():  # Found month
                        if len(parts[i+2]) == 2 and parts[i+2].isdigit():  # Found day
                            # Question ID is everything after the date
                            question_id = '_'.join(parts[i+3:]) if i+3 < len(parts) else 'unknown'
                            break
            if question_id is None:
                question_id = file_path.stem
        
        return question_id, all_forecasts
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None, []

def compute_median(forecasts):
    """Compute median of forecast values."""
    valid_forecasts = [f for f in forecasts if isinstance(f, (int, float))]
    if valid_forecasts:
        return np.median(valid_forecasts)
    return None

def main():
    # Get directory from command line or use default
    if len(sys.argv) > 1:
        directory = Path(sys.argv[1])
    else:
        directory = Path("outputs_experts_comparison_o3_initial")
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Process all pickle files in the directory
    pkl_files = sorted(directory.glob("*.pkl"))
    
    if not pkl_files:
        print(f"No pickle files found in {directory}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing {len(pkl_files)} pickle files from {directory}")
    print(f"{'='*60}")
    print(f"{'Question ID':<40} {'Median Forecast':>15}")
    print(f"{'-'*40} {'-'*15}")
    
    results = []
    for pkl_file in pkl_files:
        question_id, forecasts = load_pickle_forecasts(pkl_file)
        if forecasts:
            median = compute_median(forecasts)
            if median is not None:
                results.append((question_id, median))
                print(f"{question_id:<40} {median:>15.3f}")
        else:
            print(f"{question_id or pkl_file.stem:<40} {'No forecasts':>15}", file=sys.stderr)
    
    if results:
        print(f"{'-'*60}")
        all_medians = [m for _, m in results]
        overall_mean = np.mean(all_medians)
        overall_median = np.median(all_medians)
        print(f"\nSummary Statistics:")
        print(f"  Total questions: {len(results)}")
        print(f"  Mean of medians: {overall_mean:.3f}")
        print(f"  Median of medians: {overall_median:.3f}")
        print(f"  Min median: {min(all_medians):.3f}")
        print(f"  Max median: {max(all_medians):.3f}")

if __name__ == "__main__":
    main()