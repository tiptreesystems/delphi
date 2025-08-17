#!/usr/bin/env python3
"""
Expert Comparison Analysis Script

This script analyzes and compares the results from the expert model comparison study.
It loads results from all successful experiments and creates comprehensive comparisons.
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

class ExpertComparisonAnalyzer:
    def __init__(self, output_dirs: List[str]):
        """Initialize analyzer with output directories from experiments."""
        self.output_dirs = output_dirs
        self.loader = ForecastDataLoader()
        self.results = {}
        self.resolution_date = "2025-07-21"
        
    def load_experiment_results(self):
        """Load results from all experiment directories."""
        print("📊 Loading experiment results...")
        
        for output_dir in self.output_dirs:
            if not os.path.exists(output_dir):
                print(f"⚠️  Directory not found: {output_dir}")
                continue
                
            # Extract model name from directory
            model_name = output_dir.replace("outputs_experts_comparison_", "").replace("_", " ").title()
            
            # Load all JSON files from this experiment
            json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
            
            if not json_files:
                print(f"⚠️  No results found in: {output_dir}")
                continue
                
            print(f"📁 Loading {len(json_files)} results from {model_name}...")
            
            experiment_results = []
            for json_file in json_files:
                try:
                    with open(os.path.join(output_dir, json_file), 'r') as f:
                        result = json.load(f)
                        experiment_results.append(result)
                except Exception as e:
                    print(f"⚠️  Error loading {json_file}: {e}")
                    continue
                    
            self.results[model_name] = experiment_results
            print(f"✅ Loaded {len(experiment_results)} results for {model_name}")
            
    def extract_forecasts_and_resolutions(self):
        """Extract final forecasts and actual resolutions for analysis."""
        print("\n🔍 Extracting forecasts and resolutions...")
        
        all_data = []
        
        for model_name, experiments in self.results.items():
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
                for expert_id, expert_data in experts_data.items():
                    if isinstance(expert_data, dict) and 'prob' in expert_data:
                        prob = expert_data['prob']
                        if prob is not None:
                            try:
                                forecast_values.append(float(prob))
                            except (ValueError, TypeError):
                                continue
                
                if not forecast_values:
                    continue
                    
                ensemble_forecast = np.median(forecast_values)
                
                # Get resolution
                resolution = self.loader.get_resolution(question_id=question_id, resolution_date=self.resolution_date)
                if resolution is None:
                    continue
                    
                resolution_value = 1.0 if resolution.resolved else 0.0
                
                # Calculate Brier score
                brier_score = (ensemble_forecast - resolution_value) ** 2
                
                # Calculate absolute error
                abs_error = abs(ensemble_forecast - resolution_value)
                
                all_data.append({
                    'model': model_name,
                    'question_id': question_id,
                    'question_text': experiment.get('question_text', ''),
                    'ensemble_forecast': ensemble_forecast,
                    'resolution': resolution_value,
                    'brier_score': brier_score,
                    'abs_error': abs_error,
                    'n_experts': len(forecast_values),
                    'n_rounds': len(rounds),
                    'individual_forecasts': forecast_values
                })
                
        self.df = pd.DataFrame(all_data)
        print(f"✅ Extracted {len(self.df)} forecasts across {self.df['model'].nunique()} models")
        
        # Print summary by model
        summary = self.df.groupby('model').agg({
            'brier_score': ['count', 'mean', 'std'],
            'abs_error': ['mean', 'std'],
            'ensemble_forecast': 'mean'
        }).round(4)
        
        print("\n📈 Summary by Model:")
        print(summary)
        
    def create_performance_comparison(self):
        """Create performance comparison plots."""
        print("\n📊 Creating performance comparison plots...")
        
        # Calculate performance metrics by model
        performance = self.df.groupby('model').agg({
            'brier_score': ['mean', 'std', 'count'],
            'abs_error': ['mean', 'std']
        }).round(4)
        
        performance.columns = ['brier_mean', 'brier_std', 'n_questions', 'mae_mean', 'mae_std']
        performance = performance.reset_index()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Expert Model Comparison - Performance Metrics', fontsize=16, fontweight='bold')
        
        # 1. Brier Score Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(performance['model'], performance['brier_mean'], 
                       yerr=performance['brier_std'], capsize=5, alpha=0.7)
        ax1.set_title('Brier Score by Model (Lower is Better)', fontweight='bold')
        ax1.set_ylabel('Brier Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars1, performance['brier_mean']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Mean Absolute Error Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(performance['model'], performance['mae_mean'], 
                       yerr=performance['mae_std'], capsize=5, alpha=0.7, color='orange')
        ax2.set_title('Mean Absolute Error by Model (Lower is Better)', fontweight='bold')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars2, performance['mae_mean']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Brier Score Distribution
        ax3 = axes[1, 0]
        models = self.df['model'].unique()
        brier_data = [self.df[self.df['model'] == model]['brier_score'].values for model in models]
        
        box_plot = ax3.boxplot(brier_data, labels=models, patch_artist=True)
        ax3.set_title('Brier Score Distribution by Model', fontweight='bold')
        ax3.set_ylabel('Brier Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # Color the boxes
        colors = sns.color_palette("husl", len(models))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 4. Forecast vs Resolution Scatter
        ax4 = axes[1, 1]
        for i, model in enumerate(models):
            model_data = self.df[self.df['model'] == model]
            ax4.scatter(model_data['ensemble_forecast'], model_data['resolution'], 
                       alpha=0.6, label=model, s=50)
        
        # Add perfect calibration line
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax4.set_xlabel('Ensemble Forecast')
        ax4.set_ylabel('Actual Resolution')
        ax4.set_title('Forecast vs Resolution', fontweight='bold')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots/expert_comparison', exist_ok=True)
        plt.savefig('plots/expert_comparison/performance_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: plots/expert_comparison/performance_comparison.png")
        
        return performance
        
    def create_calibration_analysis(self):
        """Create calibration analysis plots."""
        print("\n📊 Creating calibration analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Expert Model Comparison - Calibration Analysis', fontsize=16, fontweight='bold')
        
        models = self.df['model'].unique()
        colors = sns.color_palette("husl", len(models))
        
        # 1. Calibration Plot
        ax1 = axes[0, 0]
        for i, model in enumerate(models):
            model_data = self.df[self.df['model'] == model]
            
            # Create bins for calibration
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Calculate empirical frequencies
            digitized = np.digitize(model_data['ensemble_forecast'], bins)
            empirical_freq = []
            
            for j in range(1, len(bins)):
                mask = digitized == j
                if mask.sum() > 0:
                    freq = model_data[mask]['resolution'].mean()
                    empirical_freq.append(freq)
                else:
                    empirical_freq.append(np.nan)
            
            ax1.plot(bin_centers, empirical_freq, 'o-', label=model, color=colors[i], markersize=6)
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax1.set_xlabel('Forecast Probability')
        ax1.set_ylabel('Empirical Frequency')
        ax1.set_title('Calibration Plot', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reliability Diagram
        ax2 = axes[0, 1]
        for i, model in enumerate(models):
            model_data = self.df[self.df['model'] == model]
            
            # Calculate overconfidence (forecast - resolution when forecast > 0.5)
            overconf_mask = model_data['ensemble_forecast'] > 0.5
            if overconf_mask.sum() > 0:
                overconf = (model_data[overconf_mask]['ensemble_forecast'] - 
                           model_data[overconf_mask]['resolution']).mean()
                
                # Calculate underconfidence (resolution - forecast when forecast < 0.5)
                underconf_mask = model_data['ensemble_forecast'] < 0.5
                if underconf_mask.sum() > 0:
                    underconf = (model_data[underconf_mask]['resolution'] - 
                               model_data[underconf_mask]['ensemble_forecast']).mean()
                else:
                    underconf = 0
                    
                ax2.bar(i, overconf, alpha=0.7, color=colors[i], label=f'{model} (Over)')
                ax2.bar(i + 0.4, underconf, alpha=0.7, color=colors[i], 
                       linestyle='--', label=f'{model} (Under)')
        
        ax2.set_title('Over/Under Confidence', fontweight='bold')
        ax2.set_ylabel('Average Deviation')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45)
        
        # 3. Forecast Distribution
        ax3 = axes[1, 0]
        for i, model in enumerate(models):
            model_data = self.df[self.df['model'] == model]
            ax3.hist(model_data['ensemble_forecast'], bins=20, alpha=0.6, 
                    label=model, color=colors[i], density=True)
        
        ax3.set_xlabel('Ensemble Forecast')
        ax3.set_ylabel('Density')
        ax3.set_title('Forecast Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Resolution Analysis
        ax4 = axes[1, 1]
        resolution_rates = []
        for model in models:
            model_data = self.df[self.df['model'] == model]
            res_rate = model_data['resolution'].mean()
            resolution_rates.append(res_rate)
        
        bars = ax4.bar(models, resolution_rates, alpha=0.7, color=colors)
        ax4.set_title('Resolution Rate by Model', fontweight='bold')
        ax4.set_ylabel('Proportion of Positive Resolutions')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, resolution_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig('plots/expert_comparison/calibration_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: plots/expert_comparison/calibration_analysis.png")
        
    def create_detailed_comparison_table(self):
        """Create detailed comparison table."""
        print("\n📋 Creating detailed comparison table...")
        
        # Calculate comprehensive metrics
        detailed_metrics = []
        
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            
            metrics = {
                'Model': model,
                'N_Questions': len(model_data),
                'Brier_Score_Mean': model_data['brier_score'].mean(),
                'Brier_Score_Std': model_data['brier_score'].std(),
                'MAE_Mean': model_data['abs_error'].mean(),
                'MAE_Std': model_data['abs_error'].std(),
                'Avg_Forecast': model_data['ensemble_forecast'].mean(),
                'Resolution_Rate': model_data['resolution'].mean(),
                'Avg_N_Experts': model_data['n_experts'].mean(),
                'Avg_N_Rounds': model_data['n_rounds'].mean(),
            }
            
            # Calculate calibration metrics
            # Brier decomposition
            forecast_mean = model_data['ensemble_forecast'].mean()
            resolution_mean = model_data['resolution'].mean()
            
            reliability = ((model_data['ensemble_forecast'] - model_data['resolution']) ** 2).mean()
            resolution_variance = model_data['resolution'].var()
            uncertainty = resolution_mean * (1 - resolution_mean)
            
            metrics.update({
                'Reliability': reliability,
                'Resolution_Variance': resolution_variance,
                'Uncertainty': uncertainty,
                'Forecast_Variance': model_data['ensemble_forecast'].var(),
            })
            
            detailed_metrics.append(metrics)
        
        # Create DataFrame and save
        detailed_df = pd.DataFrame(detailed_metrics)
        detailed_df = detailed_df.round(4)
        
        # Sort by Brier score (best first)
        detailed_df = detailed_df.sort_values('Brier_Score_Mean')
        
        print("\n📊 Detailed Model Comparison:")
        print(detailed_df.to_string(index=False))
        
        # Save to CSV
        os.makedirs('results', exist_ok=True)
        detailed_df.to_csv('results/expert_comparison_detailed.csv', index=False)
        print("✅ Saved: results/expert_comparison_detailed.csv")
        
        return detailed_df
        
    def create_summary_report(self, performance_df, detailed_df):
        """Create a summary report."""
        print("\n📝 Creating summary report...")
        
        # Find best and worst models
        best_brier = detailed_df.iloc[0]
        worst_brier = detailed_df.iloc[-1]
        
        report = f"""
# Expert Model Comparison Study - Summary Report

## Overview
- **Total Models Tested**: {len(detailed_df)}
- **Total Questions**: {detailed_df['N_Questions'].iloc[0]}
- **Resolution Date**: {self.resolution_date}

## Performance Ranking (by Brier Score)

### 🥇 Best Performing Model
**{best_brier['Model']}**
- Brier Score: {best_brier['Brier_Score_Mean']:.4f} (±{best_brier['Brier_Score_Std']:.4f})
- Mean Absolute Error: {best_brier['MAE_Mean']:.4f} (±{best_brier['MAE_Std']:.4f})
- Average Forecast: {best_brier['Avg_Forecast']:.4f}

### 🥉 Worst Performing Model  
**{worst_brier['Model']}**
- Brier Score: {worst_brier['Brier_Score_Mean']:.4f} (±{worst_brier['Brier_Score_Std']:.4f})
- Mean Absolute Error: {worst_brier['MAE_Mean']:.4f} (±{worst_brier['MAE_Std']:.4f})
- Average Forecast: {worst_brier['Avg_Forecast']:.4f}

## Full Ranking
"""
        
        for i, row in detailed_df.iterrows():
            rank = detailed_df.index.get_loc(i) + 1
            report += f"{rank}. **{row['Model']}** - Brier: {row['Brier_Score_Mean']:.4f}, MAE: {row['MAE_Mean']:.4f}\\n"
        
        report += f"""
## Key Insights
- **Performance Spread**: {(worst_brier['Brier_Score_Mean'] - best_brier['Brier_Score_Mean']):.4f} Brier score difference between best and worst
- **Average Questions per Model**: {detailed_df['N_Questions'].mean():.1f}
- **Average Experts per Panel**: {detailed_df['Avg_N_Experts'].mean():.1f}
- **Average Delphi Rounds**: {detailed_df['Avg_N_Rounds'].mean():.1f}

## Files Generated
- 📊 `plots/expert_comparison/performance_comparison.png` - Performance metrics comparison
- 📊 `plots/expert_comparison/calibration_analysis.png` - Calibration analysis
- 📋 `results/expert_comparison_detailed.csv` - Detailed metrics table
- 📝 `results/expert_comparison_summary.md` - This summary report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        with open('results/expert_comparison_summary.md', 'w') as f:
            f.write(report)
        
        print("✅ Saved: results/expert_comparison_summary.md")
        print("\n📋 Summary Report:")
        print(report)

def main():
    parser = argparse.ArgumentParser(description="Analyze expert comparison results")
    parser.add_argument("--output-dirs", nargs="+", 
                       default=[
                           "outputs_experts_comparison_llama_maverick",
                           "outputs_experts_comparison_gpt_oss_120b", 
                           "outputs_experts_comparison_gpt_oss_20b",
                           "outputs_experts_comparison_qwen3_32b",
                           "outputs_experts_comparison_deepseek_r1"
                       ],
                       help="Output directories to analyze")
    
    args = parser.parse_args()
    
    print("🎯 Expert Model Comparison Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ExpertComparisonAnalyzer(args.output_dirs)
    
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
    analyzer.create_calibration_analysis()
    detailed_df = analyzer.create_detailed_comparison_table()
    analyzer.create_summary_report(performance_df, detailed_df)
    
    print("\n🎉 Analysis complete!")
    print("📁 Check the 'plots/expert_comparison/' and 'results/' directories for outputs")

if __name__ == "__main__":
    main()