#!/usr/bin/env python3
"""
Prompt Technique Comparison Analysis Script

This script analyzes and compares the results from the prompt technique comparison study.
It loads results from all prompt experiments and creates comprehensive comparisons.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from dataset.dataloader import ForecastDataLoader
import argparse

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class PromptComparisonAnalyzer:
    def __init__(self, output_dirs: List[str]):
        """Initialize analyzer with output directories from experiments."""
        self.output_dirs = output_dirs
        self.loader = ForecastDataLoader()
        self.results = {}
        self.resolution_date = "2025-07-21"
        
    def load_experiment_results(self):
        """Load results from all experiment directories."""
        print("📊 Loading prompt comparison results...")
        
        for output_dir in self.output_dirs:
            if not os.path.exists(output_dir):
                print(f"⚠️  Directory not found: {output_dir}")
                continue
                
            # Extract prompt technique name from directory
            technique_name = output_dir.replace("outputs_prompt_comparison_", "").replace("_", " ").title()
            
            # Load all JSON files from this experiment
            json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
            
            if not json_files:
                print(f"⚠️  No results found in: {output_dir}")
                continue
                
            print(f"📁 Loading {len(json_files)} results from {technique_name}...")
            
            experiment_results = []
            for json_file in json_files:
                try:
                    with open(os.path.join(output_dir, json_file), 'r') as f:
                        result = json.load(f)
                        experiment_results.append(result)
                except Exception as e:
                    print(f"⚠️  Error loading {json_file}: {e}")
                    continue
                    
            self.results[technique_name] = experiment_results
            print(f"✅ Loaded {len(experiment_results)} results for {technique_name}")
            
    def extract_forecasts_and_resolutions(self):
        """Extract final forecasts and actual resolutions for analysis."""
        print("\\n🔍 Extracting forecasts and resolutions...")
        
        all_data = []
        
        for technique_name, experiments in self.results.items():
            for experiment in experiments:
                question_id = experiment.get('question_id')
                if not question_id:
                    continue
                    
                # Get final forecast (last round)
                rounds = experiment.get('rounds', [])
                if not rounds:
                    continue
                    
                final_round = rounds[-1]
                experts_data = final_round.get('experts', {})
                
                if not experts_data:
                    continue
                    
                # Extract forecast values from experts (stored as 'prob')
                forecast_values = []
                expert_responses = []
                
                for expert_id, expert_data in experts_data.items():
                    if isinstance(expert_data, dict) and 'prob' in expert_data:
                        prob = expert_data['prob']
                        if prob is not None:
                            try:
                                forecast_values.append(float(prob))
                                # Also collect response for analysis
                                if 'response' in expert_data:
                                    expert_responses.append(expert_data['response'])
                            except (ValueError, TypeError):
                                continue
                
                if not forecast_values:
                    continue
                    
                ensemble_forecast = np.median(forecast_values)
                forecast_variance = np.var(forecast_values)
                
                # Get resolution
                resolution = self.loader.get_resolution(question_id=question_id, resolution_date=self.resolution_date)
                if resolution is None:
                    continue
                    
                resolution_value = 1.0 if resolution.resolved else 0.0
                
                # Calculate Brier score
                brier_score = (ensemble_forecast - resolution_value) ** 2
                
                # Calculate absolute error
                abs_error = abs(ensemble_forecast - resolution_value)
                
                # Calculate response length metrics
                response_lengths = [len(resp) for resp in expert_responses if resp]
                avg_response_length = np.mean(response_lengths) if response_lengths else 0
                
                all_data.append({
                    'technique': technique_name,
                    'question_id': question_id,
                    'question_text': experiment.get('question_text', ''),
                    'ensemble_forecast': ensemble_forecast,
                    'forecast_variance': forecast_variance,
                    'resolution': resolution_value,
                    'brier_score': brier_score,
                    'abs_error': abs_error,
                    'n_experts': len(forecast_values),
                    'n_rounds': len(rounds),
                    'individual_forecasts': forecast_values,
                    'avg_response_length': avg_response_length,
                    'response_lengths': response_lengths
                })
                
        self.df = pd.DataFrame(all_data)
        print(f"✅ Extracted {len(self.df)} forecasts across {self.df['technique'].nunique()} techniques")
        
        # Print summary by technique
        summary = self.df.groupby('technique').agg({
            'brier_score': ['count', 'mean', 'std'],
            'abs_error': ['mean', 'std'],
            'ensemble_forecast': 'mean',
            'forecast_variance': 'mean',
            'avg_response_length': 'mean'
        }).round(4)
        
        print("\\n📈 Summary by Prompt Technique:")
        print(summary)
        
    def create_performance_comparison(self):
        """Create performance comparison plots."""
        print("\\n📊 Creating performance comparison plots...")
        
        # Calculate performance metrics by technique
        performance = self.df.groupby('technique').agg({
            'brier_score': ['mean', 'std', 'count'],
            'abs_error': ['mean', 'std'],
            'forecast_variance': ['mean', 'std'],
            'avg_response_length': 'mean'
        }).round(4)
        
        performance.columns = ['brier_mean', 'brier_std', 'n_questions', 'mae_mean', 'mae_std', 
                              'variance_mean', 'variance_std', 'response_length']
        performance = performance.reset_index()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Prompt Technique Comparison - Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Brier Score Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(performance)), performance['brier_mean'], 
                       yerr=performance['brier_std'], capsize=5, alpha=0.7)
        ax1.set_title('Brier Score by Technique (Lower is Better)', fontweight='bold')
        ax1.set_ylabel('Brier Score')
        ax1.set_xticks(range(len(performance)))
        ax1.set_xticklabels(performance['technique'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, performance['brier_mean'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 2. Mean Absolute Error Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(performance)), performance['mae_mean'], 
                       yerr=performance['mae_std'], capsize=5, alpha=0.7, color='orange')
        ax2.set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_xticks(range(len(performance)))
        ax2.set_xticklabels(performance['technique'], rotation=45, ha='right')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, performance['mae_mean'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 3. Forecast Variance
        ax3 = axes[0, 2]
        bars3 = ax3.bar(range(len(performance)), performance['variance_mean'], 
                       yerr=performance['variance_std'], capsize=5, alpha=0.7, color='green')
        ax3.set_title('Forecast Variance (Diversity)', fontweight='bold')
        ax3.set_ylabel('Forecast Variance')
        ax3.set_xticks(range(len(performance)))
        ax3.set_xticklabels(performance['technique'], rotation=45, ha='right')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars3, performance['variance_mean'])):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 4. Response Length
        ax4 = axes[1, 0]
        bars4 = ax4.bar(range(len(performance)), performance['response_length'], 
                       alpha=0.7, color='purple')
        ax4.set_title('Average Response Length', fontweight='bold')
        ax4.set_ylabel('Characters')
        ax4.set_xticks(range(len(performance)))
        ax4.set_xticklabels(performance['technique'], rotation=45, ha='right')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars4, performance['response_length'])):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 5. Brier Score Distribution
        ax5 = axes[1, 1]
        techniques = self.df['technique'].unique()
        brier_data = [self.df[self.df['technique'] == technique]['brier_score'].values for technique in techniques]
        
        box_plot = ax5.boxplot(brier_data, patch_artist=True)
        ax5.set_title('Brier Score Distribution', fontweight='bold')
        ax5.set_ylabel('Brier Score')
        ax5.set_xticks(range(1, len(techniques) + 1))
        ax5.set_xticklabels(techniques, rotation=45, ha='right')
        
        # Color the boxes
        colors = sns.color_palette("husl", len(techniques))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 6. Accuracy vs Variance Trade-off
        ax6 = axes[1, 2]
        for i, technique in enumerate(techniques):
            technique_data = performance[performance['technique'] == technique]
            ax6.scatter(technique_data['variance_mean'], technique_data['brier_mean'], 
                       s=100, alpha=0.7, label=technique)
        
        ax6.set_xlabel('Forecast Variance (Diversity)')
        ax6.set_ylabel('Brier Score (Error)')
        ax6.set_title('Accuracy vs Diversity Trade-off', fontweight='bold')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots/prompt_comparison', exist_ok=True)
        plt.savefig('plots/prompt_comparison/performance_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: plots/prompt_comparison/performance_comparison.png")
        
        return performance
        
    def create_detailed_comparison_table(self):
        """Create detailed comparison table."""
        print("\\n📋 Creating detailed comparison table...")
        
        # Calculate comprehensive metrics
        detailed_metrics = []
        
        for technique in self.df['technique'].unique():
            technique_data = self.df[self.df['technique'] == technique]
            
            metrics = {
                'Technique': technique,
                'N_Questions': len(technique_data),
                'Brier_Score_Mean': technique_data['brier_score'].mean(),
                'Brier_Score_Std': technique_data['brier_score'].std(),
                'MAE_Mean': technique_data['abs_error'].mean(),
                'MAE_Std': technique_data['abs_error'].std(),
                'Avg_Forecast': technique_data['ensemble_forecast'].mean(),
                'Forecast_Variance_Mean': technique_data['forecast_variance'].mean(),
                'Forecast_Variance_Std': technique_data['forecast_variance'].std(),
                'Resolution_Rate': technique_data['resolution'].mean(),
                'Avg_Response_Length': technique_data['avg_response_length'].mean(),
                'Avg_N_Experts': technique_data['n_experts'].mean(),
                'Avg_N_Rounds': technique_data['n_rounds'].mean(),
            }
            
            # Calculate extreme forecasts (closer to 0 or 1)
            extreme_threshold = 0.2  # Forecasts < 0.2 or > 0.8
            extreme_forecasts = technique_data['ensemble_forecast'].apply(
                lambda x: x < extreme_threshold or x > (1 - extreme_threshold)
            )
            metrics['Extreme_Forecast_Rate'] = extreme_forecasts.mean()
            
            detailed_metrics.append(metrics)
        
        # Create DataFrame and save
        detailed_df = pd.DataFrame(detailed_metrics)
        detailed_df = detailed_df.round(4)
        
        # Sort by Brier score (best first)
        detailed_df = detailed_df.sort_values('Brier_Score_Mean')
        
        print("\\n📊 Detailed Prompt Technique Comparison:")
        print(detailed_df.to_string(index=False))
        
        # Save to CSV
        os.makedirs('results', exist_ok=True)
        detailed_df.to_csv('results/prompt_comparison_detailed.csv', index=False)
        print("✅ Saved: results/prompt_comparison_detailed.csv")
        
        return detailed_df
        
    def create_summary_report(self, performance_df, detailed_df):
        """Create a summary report."""
        print("\\n📝 Creating summary report...")
        
        # Find best and worst techniques
        best_technique = detailed_df.iloc[0]
        worst_technique = detailed_df.iloc[-1]
        
        # Find most and least verbose
        most_verbose = detailed_df.loc[detailed_df['Avg_Response_Length'].idxmax()]
        least_verbose = detailed_df.loc[detailed_df['Avg_Response_Length'].idxmin()]
        
        # Find highest and lowest variance
        highest_variance = detailed_df.loc[detailed_df['Forecast_Variance_Mean'].idxmax()]
        lowest_variance = detailed_df.loc[detailed_df['Forecast_Variance_Mean'].idxmin()]
        
        report = f"""
# Prompt Technique Comparison Study - Summary Report

## Overview
- **Total Techniques Tested**: {len(detailed_df)}
- **Total Questions**: {detailed_df['N_Questions'].iloc[0]}
- **Resolution Date**: {self.resolution_date}
- **Model Used**: DeepSeek R1 (consistent across all experiments)

## Performance Ranking (by Brier Score)

### 🥇 Best Performing Technique
**{best_technique['Technique']}**
- Brier Score: {best_technique['Brier_Score_Mean']:.4f} (±{best_technique['Brier_Score_Std']:.4f})
- Mean Absolute Error: {best_technique['MAE_Mean']:.4f} (±{best_technique['MAE_Std']:.4f})
- Average Forecast: {best_technique['Avg_Forecast']:.4f}
- Forecast Variance: {best_technique['Forecast_Variance_Mean']:.4f}

### 🥉 Worst Performing Technique  
**{worst_technique['Technique']}**
- Brier Score: {worst_technique['Brier_Score_Mean']:.4f} (±{worst_technique['Brier_Score_Std']:.4f})
- Mean Absolute Error: {worst_technique['MAE_Mean']:.4f} (±{worst_technique['MAE_Std']:.4f})
- Average Forecast: {worst_technique['Avg_Forecast']:.4f}
- Forecast Variance: {worst_technique['Forecast_Variance_Mean']:.4f}

## Full Ranking
"""
        
        for i, row in detailed_df.iterrows():
            rank = detailed_df.index.get_loc(i) + 1
            report += f"{rank}. **{row['Technique']}** - Brier: {row['Brier_Score_Mean']:.4f}, MAE: {row['MAE_Mean']:.4f}, Variance: {row['Forecast_Variance_Mean']:.4f}\\n"
        
        report += f"""
## Interesting Findings

### Response Length Analysis
- **Most Verbose**: {most_verbose['Technique']} ({most_verbose['Avg_Response_Length']:.0f} chars)
- **Most Concise**: {least_verbose['Technique']} ({least_verbose['Avg_Response_Length']:.0f} chars)

### Forecast Diversity Analysis  
- **Highest Variance**: {highest_variance['Technique']} ({highest_variance['Forecast_Variance_Mean']:.4f})
- **Lowest Variance**: {lowest_variance['Technique']} ({lowest_variance['Forecast_Variance_Mean']:.4f})

## Key Insights
- **Performance Spread**: {(worst_technique['Brier_Score_Mean'] - best_technique['Brier_Score_Mean']):.4f} Brier score difference between best and worst
- **Length vs Performance**: Response length correlation with accuracy
- **Variance vs Accuracy**: Trade-off between forecast diversity and accuracy
- **Extreme Forecasts**: How often techniques make confident (near 0% or 100%) predictions

## Methodology Notes
- All experiments used the same model (DeepSeek R1) for fair comparison
- Same questions, same resolution date, same panel size
- Only the prompt technique varied between experiments
- Consistent seed (42) used for reproducibility

## Files Generated
- 📊 `plots/prompt_comparison/performance_comparison.png` - Performance analysis
- 📋 `results/prompt_comparison_detailed.csv` - Detailed metrics table
- 📝 `results/prompt_comparison_summary.md` - This summary report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        with open('results/prompt_comparison_summary.md', 'w') as f:
            f.write(report)
        
        print("✅ Saved: results/prompt_comparison_summary.md")
        print("\\n📋 Summary Report:")
        print(report)

def main():
    parser = argparse.ArgumentParser(description="Analyze prompt comparison results")
    parser.add_argument("--output-dirs", nargs="+", 
                       default=[
                           "outputs_prompt_comparison_baseline",
                           "outputs_prompt_comparison_frequency_based",
                           "outputs_prompt_comparison_short_focused", 
                           "outputs_prompt_comparison_deep_analytical",
                           "outputs_prompt_comparison_high_variance",
                           "outputs_prompt_comparison_opinionated",
                           "outputs_prompt_comparison_base_rate"
                       ],
                       help="Output directories to analyze")
    
    args = parser.parse_args()
    
    print("🎯 Prompt Technique Comparison Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = PromptComparisonAnalyzer(args.output_dirs)
    
    # Load and analyze results
    analyzer.load_experiment_results()
    
    if not analyzer.results:
        print("❌ No results found. Make sure experiments have been run.")
        return
    
    analyzer.extract_forecasts_and_resolutions()
    
    if analyzer.df.empty:
        print("❌ No valid forecast data found.")
        return
    
    # Create analyses
    performance_df = analyzer.create_performance_comparison()
    detailed_df = analyzer.create_detailed_comparison_table()
    analyzer.create_summary_report(performance_df, detailed_df)
    
    print("\\n🎉 Analysis complete!")
    print("📁 Check the 'plots/prompt_comparison/' and 'results/' directories for outputs")

if __name__ == "__main__":
    main()