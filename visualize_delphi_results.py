#!/usr/bin/env python3
"""
Delphi Results Visualization Script
Generates comprehensive visualizations and statistical analyses for Delphi experiment results.
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy.stats as stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(results_path: Path) -> Tuple[Dict, pd.DataFrame]:
    """Load detailed results and summary CSV."""
    detailed_path = results_path / "detailed_results.json"
    summary_path = results_path / "summary_results.csv"
    
    with open(detailed_path, 'r') as f:
        detailed_data = json.load(f)
    
    summary_df = pd.read_csv(summary_path)
    
    return detailed_data, summary_df


def create_visualizations_dir(results_path: Path) -> Path:
    """Create visualizations directory."""
    viz_dir = results_path / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    return viz_dir


def plot_brier_score_comparison(summary_df: pd.DataFrame, viz_dir: Path):
    """Plot Brier score comparison between AI and human forecasters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by n_experts
    grouped = summary_df.groupby('n_experts').agg({
        'ai_brier': 'mean',
        'ai_brier_std': 'mean',  # Average of standard deviations across questions
        'human_brier': 'mean',
        'ai_mae': 'mean',
        'ai_mae_std': 'mean',
        'human_mae': 'mean',
        # Add human group metrics
        'human_group_brier': 'mean',
        'human_group_brier_std': 'mean',
        'human_group_mae': 'mean',
        'human_group_mae_std': 'mean'
    }).reset_index()

    # Calculate standard error if we have multiple groups or questions
    use_within_group_std = False
    if 'n_groups' in summary_df.columns and summary_df['n_groups'].max() > 1:
        # First try to use standard error across questions if we have multiple questions
        n_questions_per_expert = summary_df.groupby('n_experts').size()
        
        if n_questions_per_expert.min() > 1:
            # We have multiple questions, calculate standard error across questions
            se_brier = summary_df.groupby('n_experts')['ai_brier'].sem().reset_index()
            se_mae = summary_df.groupby('n_experts')['ai_mae'].sem().reset_index()
            # Add human group standard errors
            se_human_brier = summary_df.groupby('n_experts')['human_group_brier'].sem().reset_index()
            se_human_mae = summary_df.groupby('n_experts')['human_group_mae'].sem().reset_index()
            
            grouped = grouped.merge(se_brier, on='n_experts', suffixes=('', '_se'))
            grouped = grouped.merge(se_mae, on='n_experts', suffixes=('', '_se'))
            grouped = grouped.merge(se_human_brier, on='n_experts', suffixes=('', '_se'))
            grouped = grouped.merge(se_human_mae, on='n_experts', suffixes=('', '_se'))
        else:
            # Only one question per n_experts, use within-group standard deviations
            use_within_group_std = True
            # Convert within-group std to standard error by dividing by sqrt(n_groups)
            n_groups = summary_df['n_groups'].iloc[0]
            grouped['ai_brier_se'] = grouped['ai_brier_std'] / np.sqrt(n_groups)
            grouped['ai_mae_se'] = grouped['ai_mae_std'] / np.sqrt(n_groups)
            grouped['human_group_brier_se'] = grouped['human_group_brier_std'] / np.sqrt(n_groups)
            grouped['human_group_mae_se'] = grouped['human_group_mae_std'] / np.sqrt(n_groups)
    else:
        # No groups or single group
        grouped['ai_brier_se'] = 0
        grouped['ai_mae_se'] = 0
        grouped['human_group_brier_se'] = 0
        grouped['human_group_mae_se'] = 0
    
    # Brier Score Comparison with error bars
    ax1.errorbar(grouped['n_experts'], grouped['ai_brier'], 
                 yerr=grouped['ai_brier_se'], 
                 fmt='o-', label='AI Aggregate', linewidth=2, markersize=8, capsize=5)
    
    # Plot human group performance with error bars if available
    if 'human_group_brier' in grouped.columns and grouped['human_group_brier'].notna().any():
        ax1.errorbar(grouped['n_experts'], grouped['human_group_brier'], 
                     yerr=grouped['human_group_brier_se'],
                     fmt='s-', label='Human Group Mean', linewidth=2, markersize=8, capsize=5, color='orange')
    
    # Also show overall human mean for reference
    ax1.plot(grouped['n_experts'], grouped['human_brier'], '^--', label='Human Overall Mean', linewidth=1.5, markersize=7, alpha=0.7)
    
    ax1.set_xlabel('Number of Experts', fontsize=12)
    ax1.set_ylabel('Average Brier Score', fontsize=12)
    title = 'Brier Score vs Number of Experts'
    if use_within_group_std:
        title += ' (error bars from group variation)'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_xticks(grouped['n_experts'])
    ax1.grid(True, alpha=0.3)
    
    # MAE Comparison with error bars
    ax2.errorbar(grouped['n_experts'], grouped['ai_mae'], 
                 yerr=grouped['ai_mae_se'],
                 fmt='o-', label='AI Aggregate', linewidth=2, markersize=8, capsize=5)
    
    # Plot human group performance with error bars if available
    if 'human_group_mae' in grouped.columns and grouped['human_group_mae'].notna().any():
        ax2.errorbar(grouped['n_experts'], grouped['human_group_mae'], 
                     yerr=grouped['human_group_mae_se'],
                     fmt='s-', label='Human Group Mean', linewidth=2, markersize=8, capsize=5, color='orange')
    
    # Also show overall human mean for reference
    ax2.plot(grouped['n_experts'], grouped['human_mae'], '^--', label='Human Overall Mean', linewidth=1.5, markersize=7, alpha=0.7)
    
    ax2.set_xlabel('Number of Experts', fontsize=12)
    ax2.set_ylabel('Average MAE', fontsize=12)
    title = 'Mean Absolute Error vs Number of Experts'
    if use_within_group_std:
        title += ' (error bars from group variation)'
    ax2.set_title(title, fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_xticks(grouped['n_experts'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'brier_mae_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_question_performance(summary_df: pd.DataFrame, viz_dir: Path):
    """Plot performance by question for each n_experts value."""
    # Get unique n_experts values
    n_experts_values = sorted(summary_df['n_experts'].unique())
    
    # Create a subplot for each n_experts value
    n_plots = len(n_experts_values)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 6 * n_plots))
    
    # If only one n_experts value, make axes a list
    if n_plots == 1:
        axes = [axes]
    
    for idx, n_exp in enumerate(n_experts_values):
        ax = axes[idx]
        
        # Filter data for this n_experts value
        data_subset = summary_df[summary_df['n_experts'] == n_exp]
        
        # Sort by question_id for consistent ordering
        data_subset = data_subset.sort_values('question_id')
        
        x = np.arange(len(data_subset))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, data_subset['ai_brier'], width, label='AI Aggregate', alpha=0.8)
        bars2 = ax.bar(x + width/2, data_subset['human_brier'], width, label='Human Mean', alpha=0.8)
        
        # Add outcome indicators
        for i, (_, row) in enumerate(data_subset.iterrows()):
            color = 'green' if row['outcome'] == 1.0 else 'red'
            ax.plot(i, -0.02, 'o', color=color, markersize=8)
        
        ax.set_xlabel('Question ID', fontsize=12)
        ax.set_ylabel('Brier Score', fontsize=12)
        ax.set_title(f'Performance by Question (n_experts={n_exp}) - Green=Outcome 1, Red=Outcome 0', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(data_subset['question_id'], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'question_performance_by_n_experts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create individual plots for each n_experts value
    for n_exp in n_experts_values:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Filter data for this n_experts value
        data_subset = summary_df[summary_df['n_experts'] == n_exp]
        data_subset = data_subset.sort_values('question_id')
        
        x = np.arange(len(data_subset))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, data_subset['ai_brier'], width, label='AI Aggregate', alpha=0.8)
        bars2 = ax.bar(x + width/2, data_subset['human_brier'], width, label='Human Mean', alpha=0.8)
        
        # Add outcome indicators
        for i, (_, row) in enumerate(data_subset.iterrows()):
            color = 'green' if row['outcome'] == 1.0 else 'red'
            ax.plot(i, -0.02, 'o', color=color, markersize=8)
        
        ax.set_xlabel('Question ID', fontsize=12)
        ax.set_ylabel('Brier Score', fontsize=12)
        ax.set_title(f'Performance by Question (n_experts={n_exp}) - Green=Outcome 1, Red=Outcome 0', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(data_subset['question_id'], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'question_performance_n_experts_{n_exp}.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_convergence_analysis(detailed_data: Dict, viz_dir: Path):
    """Analyze how forecasts converge between rounds."""
    convergence_data = []
    
    for question_id, question_data in detailed_data['questions'].items():
        for n_experts, expert_data in question_data.items():
            # Handle new groups structure
            if isinstance(expert_data, dict) and 'groups' in expert_data:
                for group_idx, group_result in expert_data['groups'].items():
                    if 'round1_responses' in group_result and 'individual_forecasts' in group_result:
                        # Extract round 1 and round 2 forecasts
                        round1_forecasts = []
                        round2_forecasts = group_result['individual_forecasts']
                        
                        for response in group_result['round1_responses']:
                            # Extract probability from response text
                            text = response['response'].lower()
                            # Look for patterns like "0.XX" or "XX%"
                            import re
                            prob_matches = re.findall(r'(?:probability|forecast|estimate)[:\s]+(?:0\.)?(\d+)%?', text)
                            if prob_matches:
                                prob = float(prob_matches[0]) / 100 if float(prob_matches[0]) > 1 else float(prob_matches[0])
                                round1_forecasts.append(prob)
                        
                        if len(round1_forecasts) == len(round2_forecasts):
                            for r1, r2 in zip(round1_forecasts, round2_forecasts):
                                convergence_data.append({
                                    'question_id': question_id,
                                    'n_experts': int(n_experts),
                                    'group_idx': int(group_idx),
                                    'round1': r1,
                                    'round2': r2,
                                    'change': r2 - r1
                                })
            # Handle legacy structure
            elif isinstance(expert_data, dict) and 'round1_responses' in expert_data:
                # Extract round 1 and round 2 forecasts
                round1_forecasts = []
                round2_forecasts = expert_data['individual_forecasts']
                
                for response in expert_data['round1_responses']:
                    # Extract probability from response text
                    text = response['response'].lower()
                    # Look for patterns like "0.XX" or "XX%"
                    import re
                    prob_matches = re.findall(r'(?:probability|forecast|estimate)[:\s]+(?:0\.)?(\d+)%?', text)
                    if prob_matches:
                        prob = float(prob_matches[0]) / 100 if float(prob_matches[0]) > 1 else float(prob_matches[0])
                        round1_forecasts.append(prob)
                
                if len(round1_forecasts) == len(round2_forecasts):
                    for r1, r2 in zip(round1_forecasts, round2_forecasts):
                        convergence_data.append({
                            'question_id': question_id,
                            'n_experts': int(n_experts),
                            'round1': r1,
                            'round2': r2,
                            'change': r2 - r1
                        })
    
    if convergence_data:
        conv_df = pd.DataFrame(convergence_data)
        
        # Filter out extreme outliers (e.g., changes > 1 or < -1 which would be impossible for probabilities)
        # Also filter out the specific -8 outlier and any other unreasonable values
        conv_df_filtered = conv_df[
            (conv_df['change'] > -1) & 
            (conv_df['change'] < 1) &
            (conv_df['round1'] >= 0) & 
            (conv_df['round1'] <= 1) &
            (conv_df['round2'] >= 0) & 
            (conv_df['round2'] <= 1)
        ]
        
        # Count how many outliers were removed
        n_outliers = len(conv_df) - len(conv_df_filtered)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot of round 1 vs round 2
        ax1.scatter(conv_df_filtered['round1'], conv_df_filtered['round2'], alpha=0.6)
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax1.set_xlabel('Round 1 Forecast', fontsize=12)
        ax1.set_ylabel('Round 2 Forecast', fontsize=12)
        ax1.set_title('Forecast Changes Between Rounds', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        
        # Distribution of changes
        ax2.hist(conv_df_filtered['change'], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Change in Forecast (Round 2 - Round 1)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        title = 'Distribution of Forecast Changes'
        if n_outliers > 0:
            title += f' ({n_outliers} outlier{"s" if n_outliers > 1 else ""} removed)'
        ax2.set_title(title, fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        mean_change = conv_df_filtered['change'].mean()
        std_change = conv_df_filtered['change'].std()
        ax2.text(0.02, 0.98, f'Mean: {mean_change:.3f}\nStd: {std_change:.3f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_expert_diversity(detailed_data: Dict, viz_dir: Path):
    """Analyze diversity of expert opinions."""
    diversity_data = []
    
    for question_id, question_data in detailed_data['questions'].items():
        for n_experts, expert_data in question_data.items():
            # Handle new groups structure
            if isinstance(expert_data, dict) and 'groups' in expert_data:
                for group_idx, group_result in expert_data['groups'].items():
                    if 'individual_forecasts' in group_result:
                        forecasts = group_result['individual_forecasts']
                        if len(forecasts) > 1:
                            diversity_data.append({
                                'question_id': question_id,
                                'n_experts': int(n_experts),
                                'group_idx': int(group_idx),
                                'std_dev': np.std(forecasts),
                                'range': max(forecasts) - min(forecasts),
                                'cv': np.std(forecasts) / np.mean(forecasts) if np.mean(forecasts) > 0 else 0
                            })
            # Handle legacy structure
            elif isinstance(expert_data, dict) and 'individual_forecasts' in expert_data:
                forecasts = expert_data['individual_forecasts']
                if len(forecasts) > 1:
                    diversity_data.append({
                        'question_id': question_id,
                        'n_experts': int(n_experts),
                        'std_dev': np.std(forecasts),
                        'range': max(forecasts) - min(forecasts),
                        'cv': np.std(forecasts) / np.mean(forecasts) if np.mean(forecasts) > 0 else 0
                    })
    
    if diversity_data:
        div_df = pd.DataFrame(diversity_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Standard deviation by number of experts
        grouped = div_df.groupby('n_experts')['std_dev'].agg(['mean', 'std']).reset_index()
        ax1.errorbar(grouped['n_experts'], grouped['mean'], yerr=grouped['std'], 
                     fmt='o-', capsize=5, linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Experts', fontsize=12)
        ax1.set_ylabel('Average Standard Deviation', fontsize=12)
        ax1.set_title('Forecast Diversity vs Number of Experts', fontsize=14, fontweight='bold')
        ax1.set_xticks(grouped['n_experts'])
        ax1.grid(True, alpha=0.3)
        
        # Box plot of ranges by number of experts
        div_df.boxplot(column='range', by='n_experts', ax=ax2)
        ax2.set_xlabel('Number of Experts', fontsize=12)
        ax2.set_ylabel('Forecast Range', fontsize=12)
        ax2.set_title('Distribution of Forecast Ranges', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'expert_diversity.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_group_distribution(detailed_data: Dict, viz_dir: Path):
    """Analyze distribution of results across different expert groups."""
    if 'n_groups' not in detailed_data['metadata'] or detailed_data['metadata']['n_groups'] <= 1:
        return  # Skip if only one group
    
    group_data = []
    
    for question_id, question_data in detailed_data['questions'].items():
        for n_experts, expert_data in question_data.items():
            if isinstance(expert_data, dict) and 'groups' in expert_data:
                for group_idx, group_result in expert_data['groups'].items():
                    group_data.append({
                        'question_id': question_id,
                        'n_experts': int(n_experts),
                        'group_idx': int(group_idx),
                        'aggregate': group_result['aggregate'],
                        'brier': group_result['brier'],
                        'mae': group_result['mae']
                    })
    
    if group_data:
        gd_df = pd.DataFrame(group_data)
        
        # Create subplots for each n_experts value
        n_experts_values = sorted(gd_df['n_experts'].unique())
        n_plots = len(n_experts_values)
        
        fig, axes = plt.subplots(n_plots, 2, figsize=(14, 5 * n_plots))
        if n_plots == 1:
            axes = axes.reshape(1, -1)
        
        for idx, n_exp in enumerate(n_experts_values):
            subset = gd_df[gd_df['n_experts'] == n_exp]
            
            # Box plot of Brier scores by group
            subset.boxplot(column='brier', by='group_idx', ax=axes[idx, 0])
            axes[idx, 0].set_xlabel('Group Index', fontsize=12)
            axes[idx, 0].set_ylabel('Brier Score', fontsize=12)
            axes[idx, 0].set_title(f'Brier Score Distribution by Group (n_experts={n_exp})', fontsize=14)
            axes[idx, 0].grid(True, alpha=0.3)
            plt.sca(axes[idx, 0])
            plt.suptitle('')
            
            # Scatter plot of aggregates by question and group
            pivot = subset.pivot_table(index='question_id', columns='group_idx', values='aggregate')
            x = np.arange(len(pivot.index))
            width = 0.8 / len(pivot.columns)
            
            for i, group_idx in enumerate(pivot.columns):
                axes[idx, 1].bar(x + i * width - 0.4 + width/2, 
                               pivot[group_idx], 
                               width, 
                               label=f'Group {group_idx}',
                               alpha=0.7)
            
            axes[idx, 1].set_xlabel('Question Index', fontsize=12)
            axes[idx, 1].set_ylabel('Aggregate Forecast', fontsize=12)
            axes[idx, 1].set_title(f'Forecasts by Question and Group (n_experts={n_exp})', fontsize=14)
            axes[idx, 1].set_xticks(x)
            axes[idx, 1].set_xticklabels(range(len(pivot.index)), rotation=0)
            axes[idx, 1].legend(fontsize=10)
            axes[idx, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'group_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a summary plot showing variance across groups
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate variance of Brier scores across groups for each n_experts
        variance_data = []
        for n_exp in n_experts_values:
            subset = gd_df[gd_df['n_experts'] == n_exp]
            # Variance across groups for each question
            question_vars = subset.groupby('question_id')['brier'].var()
            variance_data.append({
                'n_experts': n_exp,
                'mean_variance': question_vars.mean(),
                'std_variance': question_vars.std()
            })
        
        var_df = pd.DataFrame(variance_data)
        ax.errorbar(var_df['n_experts'], var_df['mean_variance'], 
                   yerr=var_df['std_variance'],
                   fmt='o-', linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel('Number of Experts', fontsize=12)
        ax.set_ylabel('Average Variance in Brier Score Across Groups', fontsize=12)
        ax.set_title('Consistency of Results Across Different Expert Groups', fontsize=14, fontweight='bold')
        ax.set_xticks(var_df['n_experts'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'group_variance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_statistical_summary(detailed_data: Dict, summary_df: pd.DataFrame, viz_dir: Path):
    """Generate comprehensive statistical summary."""
    stats_file = viz_dir / 'statistical_summary.txt'
    
    # Extract individual predictions for more detailed analysis
    ai_predictions = []
    human_predictions = []
    
    for _, row in summary_df.iterrows():
        ai_predictions.append(row['ai_aggregate'])
        human_predictions.append(row['human_mean'])
    
    with open(stats_file, 'w') as f:
        f.write("DELPHI EXPERIMENT STATISTICAL SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Experiment configuration
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of questions: {detailed_data['metadata']['n_questions']}\n")
        f.write(f"Expert counts tested: {detailed_data['metadata']['n_experts_range']}\n")
        if 'n_groups' in detailed_data['metadata']:
            f.write(f"Number of expert groups per configuration: {detailed_data['metadata']['n_groups']}\n")
        f.write(f"Model: {detailed_data['metadata']['provider']} - {detailed_data['metadata']['model']}\n\n")
        
        # Overall performance metrics
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average AI Brier Score: {summary_df['ai_brier'].mean():.4f}")
        if 'ai_brier_std' in summary_df.columns:
            f.write(f" (±{summary_df['ai_brier_std'].mean():.4f} within groups)")
        f.write("\n")
        f.write(f"Average Human Brier Score: {summary_df['human_brier'].mean():.4f}\n")
        f.write(f"Average AI MAE: {summary_df['ai_mae'].mean():.4f}")
        if 'ai_mae_std' in summary_df.columns:
            f.write(f" (±{summary_df['ai_mae_std'].mean():.4f} within groups)")
        f.write("\n")
        f.write(f"Average Human MAE: {summary_df['human_mae'].mean():.4f}\n\n")
        
        # Overall prediction statistics
        f.write("OVERALL PREDICTION STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"AI Predictions - Mean: {np.mean(ai_predictions):.4f}, Std: {np.std(ai_predictions):.4f}\n")
        f.write(f"Human Predictions - Mean: {np.mean(human_predictions):.4f}, Std: {np.std(human_predictions):.4f}\n\n")
        
        # Performance by number of experts
        f.write("PERFORMANCE BY NUMBER OF EXPERTS\n")
        f.write("-" * 30 + "\n")
        
        # Create a more comprehensive grouped summary
        agg_dict = {
            'ai_brier': 'mean',
            'human_brier': 'mean',
            'ai_mae': 'mean',
            'human_mae': 'mean',
            'ai_aggregate': ['mean', 'std'],
            'human_mean': ['mean', 'std']
        }
        
        # Add human group metrics if available
        if 'human_group_brier' in summary_df.columns:
            agg_dict.update({
                'human_group_brier': 'mean',
                'human_group_mae': 'mean'
            })
        
        grouped = summary_df.groupby('n_experts').agg(agg_dict)
        
        # Add group statistics if available
        if 'ai_brier_std' in summary_df.columns:
            grouped_std = summary_df.groupby('n_experts').agg({
                'ai_brier_std': 'mean',
                'ai_mae_std': 'mean'
            })
            # Add human group std if available
            if 'human_group_brier_std' in summary_df.columns:
                grouped_std_human = summary_df.groupby('n_experts').agg({
                    'human_group_brier_std': 'mean',
                    'human_group_mae_std': 'mean'
                })
                grouped_std = grouped_std.join(grouped_std_human)
            
            # Flatten column names first before joining
            grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
            grouped = grouped.join(grouped_std)
        else:
            # Flatten column names
            grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
        
        f.write(grouped.to_string() + "\n\n")
        
        # Group consistency analysis if multiple groups
        if 'n_groups' in summary_df.columns and summary_df['n_groups'].max() > 1:
            f.write("GROUP CONSISTENCY ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            # Calculate within-group variance for each n_experts
            for n_exp in sorted(summary_df['n_experts'].unique()):
                subset = summary_df[summary_df['n_experts'] == n_exp]
                avg_within_group_std = subset['ai_brier_std'].mean()
                f.write(f"n_experts={n_exp}:\n")
                f.write(f"  Average within-group Brier std: {avg_within_group_std:.4f}\n")
                f.write(f"  Average within-group MAE std: {subset['ai_mae_std'].mean():.4f}\n")
                
                # Add human group consistency if available
                if 'human_group_brier_std' in subset.columns:
                    f.write(f"  Average within-group Human Brier std: {subset['human_group_brier_std'].mean():.4f}\n")
                    f.write(f"  Average within-group Human MAE std: {subset['human_group_mae_std'].mean():.4f}\n")
            f.write("\n")
        
        # Human Group vs Overall Human Performance
        if 'human_group_brier' in summary_df.columns:
            f.write("HUMAN GROUP VS OVERALL HUMAN PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            
            # Compare human group performance to overall human performance
            overall_human_brier = summary_df['human_brier'].mean()
            overall_human_mae = summary_df['human_mae'].mean()
            
            for n_exp in sorted(summary_df['n_experts'].unique()):
                subset = summary_df[summary_df['n_experts'] == n_exp]
                group_brier = subset['human_group_brier'].mean()
                group_mae = subset['human_group_mae'].mean()
                
                f.write(f"n_experts={n_exp}:\n")
                f.write(f"  Human Group Brier: {group_brier:.4f} (Δ from overall: {group_brier - overall_human_brier:+.4f})\n")
                f.write(f"  Human Group MAE: {group_mae:.4f} (Δ from overall: {group_mae - overall_human_mae:+.4f})\n")
                
                # Statistical test comparing group to overall
                if len(subset) > 1:
                    t_stat_brier, p_value_brier = stats.ttest_1samp(subset['human_group_brier'].dropna(), overall_human_brier)
                    f.write(f"  t-test (group vs overall Brier): t={t_stat_brier:.4f}, p={p_value_brier:.4f}\n")
            f.write("\n")
        
        # Statistical tests
        f.write("STATISTICAL TESTS\n")
        f.write("-" * 30 + "\n")
        
        # Paired t-test for AI vs Human Brier scores
        t_stat, p_value = stats.ttest_rel(summary_df['ai_brier'], summary_df['human_brier'])
        f.write(f"Paired t-test (AI vs Human Brier): t={t_stat:.4f}, p={p_value:.4f}\n")
        
        # Paired t-test for AI vs Human predictions
        t_stat_pred, p_value_pred = stats.ttest_rel(summary_df['ai_aggregate'], summary_df['human_mean'])
        f.write(f"Paired t-test (AI vs Human Predictions): t={t_stat_pred:.4f}, p={p_value_pred:.4f}\n")
        
        # Correlation between number of experts and performance
        corr_brier = np.corrcoef(summary_df['n_experts'], summary_df['ai_brier'])[0, 1]
        corr_mae = np.corrcoef(summary_df['n_experts'], summary_df['ai_mae'])[0, 1]
        corr_pred = np.corrcoef(summary_df['n_experts'], summary_df['ai_aggregate'])[0, 1]
        f.write(f"Correlation (n_experts vs AI Brier): {corr_brier:.4f}\n")
        f.write(f"Correlation (n_experts vs AI MAE): {corr_mae:.4f}\n")
        f.write(f"Correlation (n_experts vs AI Predictions): {corr_pred:.4f}\n\n")
        
        # Question-level analysis
        f.write("QUESTION-LEVEL ANALYSIS\n")
        f.write("-" * 30 + "\n")
        question_stats = summary_df.groupby('question_id').agg({
            'ai_brier': ['mean', 'std'],
            'human_brier': ['mean', 'std'],
            'ai_aggregate': ['mean', 'std'],
            'human_mean': ['mean', 'std'],
            'outcome': 'first'
        })
        
        # Flatten column names for better readability
        question_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in question_stats.columns.values]
        f.write(question_stats.to_string() + "\n\n")
        
        # Additional analysis: prediction accuracy by outcome
        f.write("PREDICTION ACCURACY BY OUTCOME\n")
        f.write("-" * 30 + "\n")
        
        outcome_0 = summary_df[summary_df['outcome'] == 0]
        outcome_1 = summary_df[summary_df['outcome'] == 1]
        
        f.write("For questions with outcome = 0:\n")
        f.write(f"  AI predictions - Mean: {outcome_0['ai_aggregate'].mean():.4f}, Std: {outcome_0['ai_aggregate'].std():.4f}\n")
        f.write(f"  Human predictions - Mean: {outcome_0['human_mean'].mean():.4f}, Std: {outcome_0['human_mean'].std():.4f}\n")
        f.write(f"  AI Brier - Mean: {outcome_0['ai_brier'].mean():.4f}\n")
        f.write(f"  Human Brier - Mean: {outcome_0['human_brier'].mean():.4f}\n\n")
        
        f.write("For questions with outcome = 1:\n")
        f.write(f"  AI predictions - Mean: {outcome_1['ai_aggregate'].mean():.4f}, Std: {outcome_1['ai_aggregate'].std():.4f}\n")
        f.write(f"  Human predictions - Mean: {outcome_1['human_mean'].mean():.4f}, Std: {outcome_1['human_mean'].std():.4f}\n")
        f.write(f"  AI Brier - Mean: {outcome_1['ai_brier'].mean():.4f}\n")
        f.write(f"  Human Brier - Mean: {outcome_1['human_brier'].mean():.4f}\n")


def plot_calibration_curves(detailed_data: Dict, viz_dir: Path):
    """Plot calibration curves for AI aggregates."""
    calibration_data = []
    
    for question_id, question_data in detailed_data['questions'].items():
        for n_experts, expert_data in question_data.items():
            # Handle new groups structure
            if isinstance(expert_data, dict) and 'groups' in expert_data:
                for group_idx, group_result in expert_data['groups'].items():
                    if 'aggregate' in group_result and 'outcome' in group_result:
                        calibration_data.append({
                            'forecast': group_result['aggregate'],
                            'outcome': group_result['outcome'],
                            'n_experts': int(n_experts),
                            'group_idx': int(group_idx)
                        })
            # Handle legacy structure
            elif isinstance(expert_data, dict) and 'aggregate' in expert_data:
                calibration_data.append({
                    'forecast': expert_data['aggregate'],
                    'outcome': expert_data['outcome'],
                    'n_experts': int(n_experts)
                })
    
    if calibration_data:
        cal_df = pd.DataFrame(calibration_data)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create calibration bins
        n_bins = 5
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate calibration for different expert counts
        colors = plt.cm.viridis(np.linspace(0, 1, len(cal_df['n_experts'].unique())))
        
        for i, n_exp in enumerate(sorted(cal_df['n_experts'].unique())):
            subset = cal_df[cal_df['n_experts'] == n_exp]
            
            bin_outcomes = []
            bin_counts = []
            bin_errors = []
            
            for j in range(n_bins):
                mask = (subset['forecast'] >= bin_edges[j]) & (subset['forecast'] < bin_edges[j+1])
                if mask.sum() > 0:
                    outcomes_in_bin = subset.loc[mask, 'outcome'].values
                    mean_outcome = outcomes_in_bin.mean()
                    bin_outcomes.append(mean_outcome)
                    bin_counts.append(mask.sum())
                    # Calculate standard error for binomial proportion
                    se = np.sqrt(mean_outcome * (1 - mean_outcome) / mask.sum())
                    bin_errors.append(se)
                else:
                    bin_outcomes.append(np.nan)
                    bin_counts.append(0)
                    bin_errors.append(np.nan)
            
            # Plot calibration curve (binned) with error bars
            valid_bins = ~np.isnan(bin_outcomes)
            ax.errorbar(bin_centers[valid_bins], np.array(bin_outcomes)[valid_bins],
                       yerr=np.array(bin_errors)[valid_bins],
                       fmt='o-', color=colors[i], label=f'{n_exp} experts', 
                       linewidth=2, markersize=8, alpha=0.8, capsize=5)
            
            # Overlay Gaussian-smoothed calibration curve for additional insight
            sigma = 0.1  # bandwidth of Gaussian kernel
            grid = np.linspace(0, 1, 100)
            forecasts = subset['forecast'].values
            outcomes_vals = subset['outcome'].values
            smoothed = []
            for g in grid:
                w = np.exp(-0.5 * ((forecasts - g) / sigma) ** 2)
                if w.sum() > 0:
                    smoothed.append((w * outcomes_vals).sum() / w.sum())
                else:
                    smoothed.append(np.nan)
            ax.plot(grid, smoothed, color=colors[i], linestyle='--', alpha=0.5)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Perfect calibration')
        
        ax.set_xlabel('Forecast Probability', fontsize=12)
        ax.set_ylabel('Observed Frequency', fontsize=12)
        ax.set_title('Calibration Curves by Number of Experts', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_n_experts_analysis(summary_df: pd.DataFrame, viz_dir: Path):
    """Drill-down analysis of performance differences across n_experts values."""
    # Line plot: Brier score per question across n_experts
    pivot = summary_df.pivot_table(index='question_id', columns='n_experts', values='ai_brier')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot each question's brier scores across n_experts
    for idx, row in pivot.iterrows():
        ax1.plot(row.index, row.values, marker='o', linewidth=1.5, alpha=0.7, label=idx)
    ax1.set_xlabel('Number of Experts', fontsize=12)
    ax1.set_ylabel('AI Brier Score', fontsize=12)
    ax1.set_title('Per-Question Brier Score vs #Experts', fontsize=14, fontweight='bold')
    ax1.set_xticks(sorted(summary_df['n_experts'].unique()))
    ax1.grid(True, alpha=0.3)
    
    # Only show legend if few questions, else omit to reduce clutter
    if pivot.shape[0] <= 10:
        ax1.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Boxplot distribution of Brier scores by n_experts
    summary_df.boxplot(column='ai_brier', by='n_experts', ax=ax2)
    ax2.set_xlabel('Number of Experts', fontsize=12)
    ax2.set_ylabel('AI Brier Score', fontsize=12)
    ax2.set_title('Distribution of AI Brier Scores by #Experts', fontsize=14, fontweight='bold')
    plt.suptitle('')  # remove automatic boxplot title
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'n_experts_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also write a mini-summary file focusing on n_experts differences
    txt_path = viz_dir / 'n_experts_detailed_summary.txt'
    with open(txt_path, 'w') as f:
        f.write('DETAILED ANALYSIS: Why n_experts=2 performed best\n')
        f.write('-' * 55 + '\n')
        # Compute mean/median per n_experts
        stats_df = summary_df.groupby('n_experts')['ai_brier'].agg(['mean', 'median', 'std'])
        f.write(stats_df.to_string())
        f.write('\n\n')
        f.write('Relative Difference (mean brier vs n_experts=2)\n')
        baseline = stats_df.loc[2, 'mean'] if 2 in stats_df.index else np.nan
        for n, row in stats_df.iterrows():
            diff = row['mean'] - baseline
            f.write(f'  n={n}: Δ={diff:.4f}\n')
        f.write('\nQuestions where additional experts hurt performance most:\n')
        # Identify top questions where brier increases from 2 to 4+ the most
        if 2 in pivot.columns:
            for n in [4, 8, 16]:
                if n in pivot.columns:
                    diff_series = pivot[n] - pivot[2]
                    worst = diff_series.sort_values(ascending=False).head(3)
                    f.write(f'  Top 3 worst when going from 2→{n} experts:\n')
                    for qid, val in worst.items():
                        f.write(f'    {qid}: Δ={val:.4f}\n')


def plot_human_group_performance(detailed_data: Dict, viz_dir: Path):
    """Analyze performance distribution of sampled human expert groups."""
    human_group_data = []
    
    for question_id, question_data in detailed_data['questions'].items():
        for n_experts, expert_data in question_data.items():
            if isinstance(expert_data, dict) and 'groups' in expert_data:
                for group_idx, group_result in expert_data['groups'].items():
                    if 'human_group_mean' in group_result and group_result['human_group_mean'] is not None:
                        human_group_data.append({
                            'question_id': question_id,
                            'n_experts': int(n_experts),
                            'group_idx': int(group_idx),
                            'human_group_mean': group_result['human_group_mean'],
                            'human_group_brier': group_result.get('human_group_brier'),
                            'human_group_mae': group_result.get('human_group_mae'),
                            'ai_aggregate': group_result['aggregate'],
                            'ai_brier': group_result.get('brier'),
                            'ai_mae': group_result.get('mae')
                        })
    
    if human_group_data:
        hg_df = pd.DataFrame(human_group_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Box plot of human group Brier scores by n_experts
        hg_df.boxplot(column='human_group_brier', by='n_experts', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Number of Experts', fontsize=12)
        axes[0, 0].set_ylabel('Human Group Brier Score', fontsize=12)
        axes[0, 0].set_title('Human Group Brier Score Distribution', fontsize=14)
        plt.sca(axes[0, 0])
        plt.suptitle('')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scatter plot comparing AI vs Human group performance
        for n_exp in sorted(hg_df['n_experts'].unique()):
            subset = hg_df[hg_df['n_experts'] == n_exp]
            axes[0, 1].scatter(subset['human_group_brier'], subset['ai_brier'], 
                             label=f'{n_exp} experts', alpha=0.6, s=50)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('Human Group Brier Score', fontsize=12)
        axes[0, 1].set_ylabel('AI Brier Score', fontsize=12)
        axes[0, 1].set_title('AI vs Human Group Performance', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(-0.05, 1.05)
        axes[0, 1].set_ylim(-0.05, 1.05)
        
        # 3. Variance analysis across groups
        variance_data = []
        for n_exp in sorted(hg_df['n_experts'].unique()):
            subset = hg_df[hg_df['n_experts'] == n_exp]
            # Variance across groups for each question
            question_vars = subset.groupby('question_id')['human_group_brier'].var()
            variance_data.append({
                'n_experts': n_exp,
                'mean_variance': question_vars.mean(),
                'std_variance': question_vars.std(),
                'n_questions': len(question_vars)
            })
        
        var_df = pd.DataFrame(variance_data)
        axes[1, 0].errorbar(var_df['n_experts'], var_df['mean_variance'], 
                           yerr=var_df['std_variance'],
                           fmt='o-', linewidth=2, markersize=8, capsize=5)
        axes[1, 0].set_xlabel('Number of Experts', fontsize=12)
        axes[1, 0].set_ylabel('Average Variance in Human Group Brier', fontsize=12)
        axes[1, 0].set_title('Consistency of Human Groups Across Samples', fontsize=14)
        axes[1, 0].set_xticks(var_df['n_experts'])
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Distribution of performance improvements
        improvement_data = []
        for _, row in hg_df.iterrows():
            if row['human_group_brier'] is not None and row['ai_brier'] is not None:
                improvement = row['human_group_brier'] - row['ai_brier']
                improvement_data.append({
                    'n_experts': row['n_experts'],
                    'improvement': improvement
                })
        
        imp_df = pd.DataFrame(improvement_data)
        for n_exp in sorted(imp_df['n_experts'].unique()):
            subset = imp_df[imp_df['n_experts'] == n_exp]
            axes[1, 1].hist(subset['improvement'], bins=15, alpha=0.5, label=f'{n_exp} experts')
        
        axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Brier Score Difference (Human - AI)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Distribution of Performance Differences', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'human_group_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics table
        summary_stats = hg_df.groupby('n_experts').agg({
            'human_group_brier': ['mean', 'std', 'min', 'max'],
            'ai_brier': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        stats_file = viz_dir / 'human_group_performance_stats.txt'
        with open(stats_file, 'w') as f:
            f.write("HUMAN GROUP PERFORMANCE STATISTICS\n")
            f.write("=" * 50 + "\n\n")
            f.write("Summary by Number of Experts:\n")
            f.write(summary_stats.to_string())
            f.write("\n\n")
            
            # Add correlation analysis
            f.write("Correlation Analysis:\n")
            f.write("-" * 30 + "\n")
            for n_exp in sorted(hg_df['n_experts'].unique()):
                subset = hg_df[hg_df['n_experts'] == n_exp]
                if len(subset) > 2:
                    corr = subset['human_group_brier'].corr(subset['ai_brier'])
                    f.write(f"n_experts={n_exp}: correlation = {corr:.4f}\n")


def main(args=None):
    parser = argparse.ArgumentParser(description='Visualize Delphi experiment results')
    parser.add_argument('results_path', type=str, help='Path to results directory')
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"Error: Results directory {results_path} does not exist")
        return
    
    print(f"Loading data from {results_path}...")
    detailed_data, summary_df = load_data(results_path)
    
    print("Creating visualizations directory...")
    viz_dir = create_visualizations_dir(results_path)
    
    print("Generating visualizations...")
    
    # Generate all visualizations
    print("  - Brier score and MAE comparison...")
    plot_brier_score_comparison(summary_df, viz_dir)
    
    print("  - Question performance analysis...")
    plot_question_performance(summary_df, viz_dir)
    
    print("  - Convergence analysis...")
    plot_convergence_analysis(detailed_data, viz_dir)
    
    print("  - Expert diversity analysis...")
    plot_expert_diversity(detailed_data, viz_dir)
    
    print("  - Group distribution analysis...")
    plot_group_distribution(detailed_data, viz_dir)
    
    print("  - Calibration curves...")
    plot_calibration_curves(detailed_data, viz_dir)
    
    print("  - Statistical summary...")
    generate_statistical_summary(detailed_data, summary_df, viz_dir)
    
    print("  - n_experts detailed analysis...")
    plot_n_experts_analysis(summary_df, viz_dir)
    
    print("  - Human group performance analysis...")
    plot_human_group_performance(detailed_data, viz_dir)
    
    print(f"\nAll visualizations saved to {viz_dir}")
    print("Generated files:")
    for file in viz_dir.iterdir():
        print(f"  - {file.name}")


if __name__ == "__main__":
    main() 