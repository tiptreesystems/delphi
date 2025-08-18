#!/usr/bin/env python
"""
Script to compare initial predictions from pickle files with round 0 predictions from JSON files.
"""

import pickle
import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

def load_pickle_forecasts(file_path):
    """Load a pickle file and extract question_id and forecasts by expert."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        question_id = None
        expert_forecasts = {}
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if 'question_id' in item and question_id is None:
                        question_id = item['question_id']
                    if 'subject_id' in item and 'forecasts' in item:
                        expert_id = item['subject_id']
                        forecasts = item['forecasts']
                        if isinstance(forecasts, list) and forecasts:
                            expert_forecasts[expert_id] = forecasts[0]  # Take first forecast
                        elif isinstance(forecasts, (int, float)):
                            expert_forecasts[expert_id] = forecasts
        
        # Extract question_id from filename if not found
        if question_id is None:
            filename = file_path.stem
            parts = filename.split('_')
            # Find the date pattern and take everything after it
            for i, part in enumerate(parts):
                if len(part) == 4 and part.isdigit() and i+2 < len(parts):
                    if len(parts[i+1]) == 2 and parts[i+1].isdigit():
                        if len(parts[i+2]) == 2 and parts[i+2].isdigit():
                            question_id = '_'.join(parts[i+3:]) if i+3 < len(parts) else 'unknown'
                            break
            if question_id is None:
                question_id = file_path.stem
        
        return question_id, expert_forecasts
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None, {}

def load_json_round0_forecasts(file_path):
    """Load a JSON file and extract round 0 forecasts."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        question_id = data.get('question_id', Path(file_path).stem)
        
        # Extract round 0 forecasts
        expert_forecasts = {}
        rounds = data.get('rounds', [])
        
        for round_data in rounds:
            if round_data.get('round') == 0:
                experts = round_data.get('experts', {})
                for expert_id, expert_data in experts.items():
                    if 'prob' in expert_data:
                        expert_forecasts[expert_id] = expert_data['prob']
                break
        
        return question_id, expert_forecasts
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None, {}

def compare_directories(pickle_dir, json_dir):
    """Compare initial predictions from pickles with round 0 from JSONs."""
    
    pickle_dir = Path(pickle_dir)
    json_dir = Path(json_dir)
    
    if not pickle_dir.exists():
        print(f"Error: Pickle directory {pickle_dir} does not exist", file=sys.stderr)
        return
    
    if not json_dir.exists():
        print(f"Error: JSON directory {json_dir} does not exist", file=sys.stderr)
        return
    
    # Load all pickle files
    pickle_data = {}
    for pkl_file in sorted(pickle_dir.glob("*.pkl")):
        question_id, forecasts = load_pickle_forecasts(pkl_file)
        if question_id and forecasts:
            pickle_data[question_id] = forecasts
    
    # Load all JSON files
    json_data = {}
    for json_file in sorted(json_dir.glob("*.json")):
        if json_file.is_file():  # Skip directories
            question_id, forecasts = load_json_round0_forecasts(json_file)
            if question_id and forecasts:
                json_data[question_id] = forecasts
    
    # Find matching questions
    common_questions = set(pickle_data.keys()) & set(json_data.keys())
    
    if not common_questions:
        print("No matching questions found between pickle and JSON files")
        return
    
    print(f"\nComparing {len(common_questions)} questions")
    print("=" * 80)
    
    results = []
    detailed_comparison = []
    
    for question_id in sorted(common_questions):
        pkl_forecasts = pickle_data[question_id]
        json_forecasts = json_data[question_id]
        
        # Compute medians
        pkl_values = list(pkl_forecasts.values())
        json_values = list(json_forecasts.values())
        
        pkl_median = np.median(pkl_values) if pkl_values else None
        json_median = np.median(json_values) if json_values else None
        
        if pkl_median is not None and json_median is not None:
            diff = json_median - pkl_median
            results.append({
                'question_id': question_id,
                'pkl_median': pkl_median,
                'json_median': json_median,
                'diff': diff,
                'pkl_experts': len(pkl_values),
                'json_experts': len(json_values)
            })
            
            # Check if experts match
            pkl_experts = set(pkl_forecasts.keys())
            json_experts = set(json_forecasts.keys())
            common_experts = pkl_experts & json_experts
            
            if common_experts:
                expert_diffs = []
                for expert in common_experts:
                    expert_diff = json_forecasts[expert] - pkl_forecasts[expert]
                    expert_diffs.append(expert_diff)
                
                detailed_comparison.append({
                    'question_id': question_id,
                    'common_experts': len(common_experts),
                    'avg_expert_diff': np.mean(expert_diffs),
                    'max_expert_diff': max(expert_diffs),
                    'min_expert_diff': min(expert_diffs)
                })
    
    # Display results
    print(f"{'Question ID':<40} {'Pickle':<10} {'JSON R0':<10} {'Diff':<10}")
    print("-" * 70)
    
    for r in results[:20]:  # Show first 20
        qid = r['question_id'][:37] + "..." if len(r['question_id']) > 40 else r['question_id']
        print(f"{qid:<40} {r['pkl_median']:<10.3f} {r['json_median']:<10.3f} {r['diff']:+10.3f}")
    
    if len(results) > 20:
        print(f"... and {len(results) - 20} more questions")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)
    
    if results:
        diffs = [r['diff'] for r in results]
        abs_diffs = [abs(d) for d in diffs]
        
        print(f"Total questions compared: {len(results)}")
        print(f"Mean difference (JSON - Pickle): {np.mean(diffs):+.4f}")
        print(f"Median difference: {np.median(diffs):+.4f}")
        print(f"Std dev of differences: {np.std(diffs):.4f}")
        print(f"Mean absolute difference: {np.mean(abs_diffs):.4f}")
        print(f"Max positive difference: {max(diffs):+.4f}")
        print(f"Max negative difference: {min(diffs):+.4f}")
        
        # Check if they're essentially the same
        very_close = sum(1 for d in abs_diffs if d < 0.001)
        close = sum(1 for d in abs_diffs if d < 0.01)
        print(f"\nQuestions with <0.001 difference: {very_close} ({100*very_close/len(results):.1f}%)")
        print(f"Questions with <0.01 difference: {close} ({100*close/len(results):.1f}%)")
    
    # Expert-level comparison if available
    if detailed_comparison:
        print("\n" + "=" * 80)
        print("EXPERT-LEVEL COMPARISON (for matching expert IDs)")
        print("-" * 80)
        
        all_expert_diffs = []
        for dc in detailed_comparison:
            print(f"\nQuestion: {dc['question_id'][:40]}")
            print(f"  Common experts: {dc['common_experts']}")
            print(f"  Avg expert diff: {dc['avg_expert_diff']:+.4f}")
            print(f"  Max expert diff: {dc['max_expert_diff']:+.4f}")
            print(f"  Min expert diff: {dc['min_expert_diff']:+.4f}")
            
            if len(detailed_comparison) > 5:
                break  # Just show first 5 for detail
        
        if len(detailed_comparison) > 5:
            print(f"\n... and {len(detailed_comparison) - 5} more questions with matching experts")

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_initial_vs_round0.py <pickle_dir> <json_dir>")
        print("Example: python compare_initial_vs_round0.py outputs_experts_comparison_o3_initial outputs_experts_comparison_o3")
        sys.exit(1)
    
    pickle_dir = sys.argv[1]
    json_dir = sys.argv[2]
    
    compare_directories(pickle_dir, json_dir)

if __name__ == "__main__":
    main()