#!/usr/bin/env python
"""
Script to check SF and Public median Brier scores from the dataset.
"""

import json
import numpy as np
from pathlib import Path
from dataset.dataloader import ForecastDataLoader

def compute_brier_score(prob, outcome):
    """Compute Brier score for a binary outcome."""
    return (prob - outcome) ** 2

def get_baseline_briers():
    """Get SF and Public Brier scores directly from the dataset."""
    loader = ForecastDataLoader()
    
    # Get all unique question IDs from one of the output directories
    # Let's use O3 as an example
    output_dir = Path('outputs_experts_comparison_o3')
    json_files = sorted([f for f in output_dir.glob("*.json") if f.is_file()])
    
    all_sf_briers = []
    all_public_briers = []
    questions_processed = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            question_id = data.get('question_id', '')
            resolution_date = data.get('resolution_date', '2025-07-21')
            resolution = data.get('resolution', {})
            y_true = resolution.get('resolved_to')
            
            if not question_id or y_true is None:
                continue
            
            # Get SF forecasts
            try:
                sf_forecasts = loader.get_super_forecasts(question_id=question_id, resolution_date=resolution_date)
                if sf_forecasts:
                    sf_probs = [sf.forecast for sf in sf_forecasts]
                    # Calculate median SF probability
                    median_sf_prob = np.median(sf_probs)
                    # Calculate Brier score of the median
                    median_sf_brier = compute_brier_score(median_sf_prob, y_true)
                    all_sf_briers.append(median_sf_brier)
                    
                    print(f"Q: {question_id[:40]:<40} SF median prob: {median_sf_prob:.3f}, Brier: {median_sf_brier:.3f}")
            except Exception as e:
                print(f"Could not get SF data for {question_id}: {e}")
            
            # Get Public forecasts
            try:
                public_forecasts = loader.get_public_forecasts(question_id=question_id, resolution_date=resolution_date)
                if public_forecasts:
                    public_probs = [pf.forecast for pf in public_forecasts]
                    # Calculate median Public probability
                    median_public_prob = np.median(public_probs)
                    # Calculate Brier score of the median
                    median_public_brier = compute_brier_score(median_public_prob, y_true)
                    all_public_briers.append(median_public_brier)
                    
                    print(f"    Public median prob: {median_public_prob:.3f}, Brier: {median_public_brier:.3f}")
            except Exception as e:
                print(f"Could not get Public data for {question_id}: {e}")
            
            questions_processed.append(question_id)
            print()
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Calculate overall statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if all_sf_briers:
        print(f"SF Median Brier Scores:")
        print(f"  Number of questions: {len(all_sf_briers)}")
        print(f"  Mean of median Briers: {np.mean(all_sf_briers):.4f}")
        print(f"  Median of median Briers: {np.median(all_sf_briers):.4f}")
        print(f"  Std of median Briers: {np.std(all_sf_briers):.4f}")
    else:
        print("No SF data found")
    
    print()
    
    if all_public_briers:
        print(f"Public Median Brier Scores:")
        print(f"  Number of questions: {len(all_public_briers)}")
        print(f"  Mean of median Briers: {np.mean(all_public_briers):.4f}")
        print(f"  Median of median Briers: {np.median(all_public_briers):.4f}")
        print(f"  Std of median Briers: {np.std(all_public_briers):.4f}")
    else:
        print("No Public data found")
    
    return all_sf_briers, all_public_briers

if __name__ == "__main__":
    get_baseline_briers()