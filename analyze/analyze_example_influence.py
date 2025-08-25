#!/usr/bin/env python3
"""
Analyze the relationship between example probabilities provided to experts
and their final predictions in Delphi experiments.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import re
from collections import defaultdict
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def extract_example_probabilities(text):
    """
    Extract probabilities from reference examples in expert responses.
    Looks for patterns like "Probability: 0.16" in the text.
    """
    # Pattern to match "Probability: X.XX" or similar formats
    patterns = [
        r'Probability:\s*([0-9]+\.?[0-9]*)',
        r'PROBABILITY:\s*([0-9]+\.?[0-9]*)',
        r'probability:\s*([0-9]+\.?[0-9]*)',
        r'Forecast:\s*([0-9]+\.?[0-9]*)',
        r'FORECAST:\s*([0-9]+\.?[0-9]*)',
        r'forecast:\s*([0-9]+\.?[0-9]*)',
    ]
    
    probabilities = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                prob = float(match)
                if 0 <= prob <= 1:
                    probabilities.append(prob)
                elif 0 <= prob <= 100:
                    # Convert percentage to probability
                    probabilities.append(prob / 100)
            except ValueError:
                continue
    
    return probabilities

def extract_final_probability(text):
    """
    Extract the final probability from an expert's response.
    Looks for "FINAL PROBABILITY:" pattern.
    """
    patterns = [
        r'FINAL PROBABILITY:\s*([0-9]+\.?[0-9]*)',
        r'Final Probability:\s*([0-9]+\.?[0-9]*)',
        r'FINAL FORECAST:\s*([0-9]+\.?[0-9]*)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                # Take the last match if multiple found
                prob = float(matches[-1])
                if 0 <= prob <= 1:
                    return prob
                elif 0 <= prob <= 100:
                    return prob / 100
            except ValueError:
                continue
    
    return None

def extract_data_from_files(output_dir):
    """
    Extract example probabilities and expert predictions from Delphi output files.
    
    Returns:
        list of dicts with expert data
    """
    output_path = Path(output_dir)
    expert_data = []
    
    # Find all JSON files
    json_files = list(output_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process")
    
    for i, json_file in enumerate(json_files):
        if i % 5 == 0:
            print(f"Processing file {i+1}/{len(json_files)}: {json_file.name}")
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            question_id = data.get('question_id', 'unknown')
            
            # Create a mapping of expert predictions from ALL rounds
            expert_predictions_by_round = {}
            for round_data in data.get('rounds', []):
                round_num = round_data.get('round', 0)
                experts_data = round_data.get('experts', {})
                
                for expert_id, expert_response in experts_data.items():
                    if isinstance(expert_response, dict):
                        expert_prob = expert_response.get('prob')
                        if expert_prob is not None:
                            if expert_id not in expert_predictions_by_round:
                                expert_predictions_by_round[expert_id] = {}
                            expert_predictions_by_round[expert_id][round_num] = expert_prob
            
            # Debug: print what we found
            if i == 0:  # Only for first file
                total_predictions = sum(len(rounds) for rounds in expert_predictions_by_round.values())
                print(f"  Found {total_predictions} expert predictions across {len(expert_predictions_by_round)} experts")
                print(f"  Has example_pairs: {'example_pairs' in data}")
                if 'example_pairs' in data:
                    print(f"  Example pairs for {len(data['example_pairs'])} experts")
            
            # Now match with example pairs for ALL rounds
            if 'example_pairs' in data:
                example_pairs = data['example_pairs']
                
                for expert_id, pairs in example_pairs.items():
                    # Get the expert's predictions across all rounds
                    expert_rounds = expert_predictions_by_round.get(expert_id, {})
                    
                    if i == 0:  # Debug for first file
                        print(f"    Expert {expert_id}: has_predictions={len(expert_rounds)}, pairs_type={type(pairs)}, pairs_len={len(pairs) if isinstance(pairs, list) else 'N/A'}")
                    
                    if isinstance(pairs, list) and expert_rounds:
                        example_probs = []
                        
                        # Extract probabilities from example pairs (same for all rounds)
                        if len(pairs) > 0 and isinstance(pairs[0], list):
                            examples_list = pairs[0]
                            for pair in examples_list:
                                if isinstance(pair, list) and len(pair) >= 2:
                                    forecast_data = pair[1]
                                    if isinstance(forecast_data, dict) and 'forecast' in forecast_data:
                                        try:
                                            prob = float(forecast_data['forecast'])
                                            if 0 <= prob <= 1:
                                                example_probs.append(prob)
                                            elif 0 <= prob <= 100:
                                                example_probs.append(prob / 100)
                                        except (ValueError, TypeError):
                                            pass
                        
                        # Store the data for each round this expert participated in
                        if example_probs:
                            for round_num, expert_prob in expert_rounds.items():
                                expert_data.append({
                                    'expert_id': expert_id,
                                    'question_id': question_id,
                                    'round': round_num,
                                    'example_probs': example_probs,
                                    'expert_prob': expert_prob,
                                    'source': 'example_pairs_matched'
                                })
                            
            # Also check for experts with predictions but no example pairs
            for expert_id, rounds_data in expert_predictions_by_round.items():
                if expert_id not in (data.get('example_pairs', {}).keys() if 'example_pairs' in data else []):
                    for round_num, expert_prob in rounds_data.items():
                        expert_data.append({
                            'expert_id': expert_id,
                            'question_id': question_id,
                            'round': round_num,
                            'example_probs': [],
                            'expert_prob': expert_prob,
                            'source': 'no_examples'
                        })
                        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"Extracted data for {len(expert_data)} expert-question pairs")
    
    # Print summary of data found
    with_examples = sum(1 for d in expert_data if d['example_probs'])
    without_examples = len(expert_data) - with_examples
    print(f"  - {with_examples} with example probabilities")
    print(f"  - {without_examples} without example probabilities")
    
    if with_examples > 0:
        example_counts = [len(d['example_probs']) for d in expert_data if d['example_probs']]
        print(f"  - Example counts: min={min(example_counts)}, max={max(example_counts)}, avg={np.mean(example_counts):.1f}")
    
    return expert_data

def analyze_example_influence(expert_data):
    """
    Analyze the relationship between example probabilities and expert predictions by round.
    """
    # Prepare data for analysis
    analysis_data = []
    
    for entry in expert_data:
        if 'expert_prob' in entry and entry.get('example_probs'):
            example_probs = entry['example_probs']
            expert_prob = entry['expert_prob']
            
            analysis_data.append({
                'expert_id': entry['expert_id'],
                'question_id': entry['question_id'],
                'round': entry.get('round', 0),
                'expert_prob': expert_prob,
                'example_mean': np.mean(example_probs),
                'example_std': np.std(example_probs) if len(example_probs) > 1 else 0,
                'example_min': np.min(example_probs),
                'example_max': np.max(example_probs),
                'example_median': np.median(example_probs),
                'n_examples': len(example_probs),
                'example_range': np.max(example_probs) - np.min(example_probs),
            })
    
    if not analysis_data:
        print("No data with both examples and expert predictions found!")
        return None
    
    df = pd.DataFrame(analysis_data)
    
    # Calculate deviation from example mean
    df['deviation_from_mean'] = df['expert_prob'] - df['example_mean']
    df['within_range'] = (df['expert_prob'] >= df['example_min']) & (df['expert_prob'] <= df['example_max'])
    df['close_to_mean'] = np.abs(df['deviation_from_mean']) < 0.1
    
    # Overall analysis
    print("\n=== OVERALL ANALYSIS (All Rounds) ===")
    print(f"Total expert-question pairs analyzed: {len(df)}")
    print(f"Unique experts: {df['expert_id'].nunique()}")
    print(f"Unique questions: {df['question_id'].nunique()}")
    print(f"Rounds present: {sorted(df['round'].unique())}")
    
    # Round-by-round analysis
    print("\n=== ROUND-BY-ROUND ANALYSIS ===")
    
    for round_num in sorted(df['round'].unique()):
        round_df = df[df['round'] == round_num]
        
        print(f"\n--- ROUND {round_num} ---")
        print(f"Predictions: {len(round_df)}")
        print(f"Experts: {round_df['expert_id'].nunique()}")
        
        # Correlations for this round
        correlations = {
            'Mean': round_df['expert_prob'].corr(round_df['example_mean']),
            'Median': round_df['expert_prob'].corr(round_df['example_median']),
            'Min': round_df['expert_prob'].corr(round_df['example_min']),
            'Max': round_df['expert_prob'].corr(round_df['example_max']),
            'Std': round_df['expert_prob'].corr(round_df['example_std']),
            'Range': round_df['expert_prob'].corr(round_df['example_range']),
        }
        
        print("Correlations with example statistics:")
        for stat, corr in correlations.items():
            print(f"  Example {stat}: {corr:.3f}")
        
        # Anchoring metrics for this round
        print("\nAnchoring metrics:")
        print(f"  Mean deviation from examples: {round_df['deviation_from_mean'].mean():+.3f}")
        print(f"  Std deviation: {round_df['deviation_from_mean'].std():.3f}")
        print(f"  Within example range: {round_df['within_range'].mean() * 100:.1f}%")
        print(f"  Close to example mean (±0.1): {round_df['close_to_mean'].mean() * 100:.1f}%")
    
    # Compare anchoring across rounds
    print("\n=== ANCHORING COMPARISON ACROSS ROUNDS ===")
    round_summary = df.groupby('round').agg({
        'deviation_from_mean': ['mean', 'std'],
        'within_range': 'mean',
        'close_to_mean': 'mean',
        'expert_prob': 'count'
    }).round(3)
    
    print("\nSummary by round:")
    print("Round | Count | Mean Dev | Std Dev | Within Range | Close to Mean")
    print("------|-------|----------|---------|--------------|---------------")
    
    for round_num in sorted(df['round'].unique()):
        row = round_summary.loc[round_num]
        count = int(row[('expert_prob', 'count')])
        mean_dev = row[('deviation_from_mean', 'mean')]
        std_dev = row[('deviation_from_mean', 'std')]
        within_range = row[('within_range', 'mean')] * 100
        close_mean = row[('close_to_mean', 'mean')] * 100
        
        print(f"  {round_num}   |  {count:3d}  | {mean_dev:+8.3f} | {std_dev:7.3f} | {within_range:10.1f}% | {close_mean:11.1f}%")
    
    # Expert consistency across rounds
    print("\n=== EXPERT CONSISTENCY ACROSS ROUNDS ===")
    expert_round_counts = df.groupby('expert_id')['round'].nunique()
    multi_round_experts = expert_round_counts[expert_round_counts > 1]
    
    print(f"Experts appearing in multiple rounds: {len(multi_round_experts)}")
    if len(multi_round_experts) > 0:
        print("Expert anchoring evolution (deviation from examples):")
        
        for expert_id in multi_round_experts.index[:10]:  # Show top 10
            expert_df = df[df['expert_id'] == expert_id].sort_values('round')
            deviations = expert_df['deviation_from_mean'].tolist()
            rounds = expert_df['round'].tolist()
            
            dev_str = " → ".join([f"R{r}:{d:+.2f}" for r, d in zip(rounds, deviations)])
            print(f"  {expert_id}: {dev_str}")
    
    return df

def create_visualizations(df, output_dir):
    """
    Create visualizations showing the relationship between examples and predictions by round.
    """
    output_path = Path(output_dir)
    rounds = sorted(df['round'].unique())
    
    # 1. Round comparison overview
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Correlation trends across rounds
    ax = axes[0, 0]
    correlation_stats = ['Mean', 'Median', 'Min', 'Max']
    correlation_data = {}
    
    for stat_name in correlation_stats:
        correlation_data[stat_name] = []
        for round_num in rounds:
            round_df = df[df['round'] == round_num]
            corr = round_df['expert_prob'].corr(round_df[f'example_{stat_name.lower()}'])
            correlation_data[stat_name].append(corr)
    
    for stat_name, correlations in correlation_data.items():
        ax.plot(rounds, correlations, marker='o', label=f'Example {stat_name}', linewidth=2)
    
    ax.set_xlabel('Delphi Round')
    ax.set_ylabel('Correlation with Expert Predictions')
    ax.set_title('Anchoring Strength Across Rounds\n(Correlation with Example Statistics)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(rounds)
    
    # Plot 2: Anchoring metrics across rounds
    ax = axes[0, 1]
    round_metrics = []
    for round_num in rounds:
        round_df = df[df['round'] == round_num]
        within_range_pct = round_df['within_range'].mean() * 100
        close_to_mean_pct = round_df['close_to_mean'].mean() * 100
        mean_abs_dev = np.abs(round_df['deviation_from_mean']).mean()
        
        round_metrics.append({
            'round': round_num,
            'within_range': within_range_pct,
            'close_to_mean': close_to_mean_pct,
            'mean_abs_deviation': mean_abs_dev
        })
    
    metrics_df = pd.DataFrame(round_metrics)
    
    ax2 = ax.twinx()
    line1 = ax.plot(metrics_df['round'], metrics_df['within_range'], 'b-o', label='Within Example Range', linewidth=2)
    line2 = ax.plot(metrics_df['round'], metrics_df['close_to_mean'], 'g-o', label='Close to Mean (±0.1)', linewidth=2)
    line3 = ax2.plot(metrics_df['round'], metrics_df['mean_abs_deviation'], 'r-s', label='Mean Abs. Deviation', linewidth=2)
    
    ax.set_xlabel('Delphi Round')
    ax.set_ylabel('Percentage (%)')
    ax2.set_ylabel('Mean Absolute Deviation')
    ax.set_title('Anchoring Behavior Across Rounds')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    
    ax.grid(True, alpha=0.3)
    ax.set_xticks(rounds)
    ax2.set_xticks(rounds)
    
    # Plot 3: Distribution of deviations by round
    ax = axes[1, 0]
    round_colors = plt.cm.viridis(np.linspace(0, 1, len(rounds)))
    
    for i, round_num in enumerate(rounds):
        round_df = df[df['round'] == round_num]
        ax.hist(round_df['deviation_from_mean'], bins=20, alpha=0.6, 
                label=f'Round {round_num}', color=round_colors[i], density=True)
    
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='No deviation')
    ax.set_xlabel('Deviation from Example Mean')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Deviations by Round')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Expert evolution across rounds (for experts in multiple rounds)
    ax = axes[1, 1]
    expert_round_counts = df.groupby('expert_id')['round'].nunique()
    multi_round_experts = expert_round_counts[expert_round_counts > 1].index
    
    if len(multi_round_experts) > 0:
        # Show evolution for a sample of experts
        sample_experts = list(multi_round_experts)[:8]  # Show up to 8 experts
        
        for expert_id in sample_experts:
            expert_df = df[df['expert_id'] == expert_id].sort_values('round')
            ax.plot(expert_df['round'], expert_df['deviation_from_mean'], 
                   marker='o', label=expert_id, alpha=0.7)
        
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Delphi Round')
        ax.set_ylabel('Deviation from Example Mean')
        ax.set_title('Individual Expert Evolution\n(Sample of Multi-Round Experts)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rounds)
    else:
        ax.text(0.5, 0.5, 'No experts found\nin multiple rounds', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Individual Expert Evolution')
    
    plt.suptitle('Anchoring Analysis Across Delphi Rounds', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save plot
    plot_filename = output_path / "anchoring_by_rounds_overview.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved round overview to: {plot_filename}")
    plt.close()
    
    # 2. Individual round detailed plots
    for round_num in rounds:
        round_df = df[df['round'] == round_num]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Mean vs prediction
        ax = axes[0, 0]
        ax.scatter(round_df['example_mean'], round_df['expert_prob'], alpha=0.6, s=50)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel('Example Mean Probability')
        ax.set_ylabel('Expert Prediction')
        ax.set_title(f'Round {round_num}: Predictions vs Example Mean\n(r={round_df["expert_prob"].corr(round_df["example_mean"]):.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Median
        ax = axes[0, 1]
        ax.scatter(round_df['example_median'], round_df['expert_prob'], alpha=0.6, s=50)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel('Example Median Probability')
        ax.set_ylabel('Expert Prediction')
        ax.set_title(f'Round {round_num}: Predictions vs Example Median\n(r={round_df["expert_prob"].corr(round_df["example_median"]):.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Range influence
        ax = axes[0, 2]
        sc = ax.scatter(round_df['example_mean'], round_df['expert_prob'], 
                        c=round_df['example_range'], cmap='viridis', alpha=0.6, s=50)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax.set_xlabel('Example Mean Probability')
        ax.set_ylabel('Expert Prediction')
        ax.set_title(f'Round {round_num}: Colored by Example Range')
        plt.colorbar(sc, ax=ax, label='Example Range')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Deviation distribution
        ax = axes[1, 0]
        ax.hist(round_df['deviation_from_mean'], bins=15, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', label='No deviation')
        ax.set_xlabel('Deviation from Example Mean')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Round {round_num}: Distribution of Deviations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Min-Max range with predictions
        ax = axes[1, 1]
        for idx, (_, row) in enumerate(round_df.iterrows()):
            ax.plot([idx, idx], [row['example_min'], row['example_max']], 
                    'b-', alpha=0.3, linewidth=2)
            ax.plot(idx, row['example_mean'], 'bo', markersize=4)
            ax.plot(idx, row['expert_prob'], 'ro', markersize=4)
        
        ax.set_xlabel('Prediction Index')
        ax.set_ylabel('Probability')
        ax.set_title(f'Round {round_num}: Predictions vs Example Ranges')
        ax.legend(['Example range', 'Example mean', 'Expert prediction'])
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Stats summary
        ax = axes[1, 2]
        within_range_pct = round_df['within_range'].mean() * 100
        close_to_mean_pct = round_df['close_to_mean'].mean() * 100
        mean_dev = round_df['deviation_from_mean'].mean()
        std_dev = round_df['deviation_from_mean'].std()
        
        stats_text = f"""Round {round_num} Summary:
        
Predictions: {len(round_df)}
Experts: {round_df['expert_id'].nunique()}

Anchoring Metrics:
• Within range: {within_range_pct:.1f}%
• Close to mean: {close_to_mean_pct:.1f}%
• Mean deviation: {mean_dev:+.3f}
• Std deviation: {std_dev:.3f}

Correlations:
• Mean: {round_df["expert_prob"].corr(round_df["example_mean"]):.3f}
• Median: {round_df["expert_prob"].corr(round_df["example_median"]):.3f}
• Min: {round_df["expert_prob"].corr(round_df["example_min"]):.3f}
• Max: {round_df["expert_prob"].corr(round_df["example_max"]):.3f}"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.suptitle(f'Detailed Analysis - Round {round_num}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save individual round plot
        plot_filename = output_path / f"anchoring_round_{round_num}_detailed.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved Round {round_num} analysis to: {plot_filename}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze example influence on expert predictions")
    parser.add_argument("output_dir", help="Directory containing Delphi experiment output JSON files")
    
    args = parser.parse_args()
    
    print(f"Starting analysis of directory: {args.output_dir}")
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Directory {args.output_dir} does not exist")
        return 1
    
    # Extract data from files
    print("\nExtracting example probabilities and expert predictions...")
    expert_data = extract_data_from_files(args.output_dir)
    
    if not expert_data:
        print("No expert data found!")
        return 1
    
    # Analyze the relationship
    print("\nAnalyzing example influence...")
    df = analyze_example_influence(expert_data)
    
    if df is not None and len(df) > 0:
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(df, args.output_dir)
        
        # Save data to CSV for further analysis
        csv_filename = Path(args.output_dir) / "example_influence_data.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nSaved analysis data to: {csv_filename}")
    
    print("\nAnalysis completed!")
    return 0

if __name__ == "__main__":
    exit(main())