#!/usr/bin/env python3
"""
Compare OSS-120B variants: 3k tokens vs expert configuration.
Analyzes aggregate statistical performance from Delphi experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats
import glob
from comparison_plotter import DelphiComparisonPlotter

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class OSS120BVariantComparator:
    """Specialized class for comparing OSS-120B variants."""
    
    def __init__(self):
        """Initialize the comparator."""
        self.base_dir = Path(".")
        self.plotter = DelphiComparisonPlotter("plots/oss_120b_comparison")
        
    def load_delphi_results(self, output_dir: str) -> Dict[str, Any]:
        """
        Load all Delphi results from an output directory.
        
        Args:
            output_dir: Path to output directory containing JSON files
            
        Returns:
            Dictionary with aggregated results
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"Warning: {output_dir} not found")
            return {}
        
        # Find all Delphi log JSON files
        json_files = list(output_path.glob("groq_delphi_log_*.json"))
        
        if not json_files:
            print(f"Warning: No JSON files found in {output_dir}")
            return {}
        
        print(f"Loading {len(json_files)} result files from {output_dir}")
        
        # Aggregate metrics
        all_aggregates = []
        all_individual_forecasts = []
        all_round1_responses = []
        all_round2_responses = []
        all_brier_scores = []
        all_mae_scores = []
        all_questions = []
        
        config_info = None
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract config info from first file
                if config_info is None and 'config' in data:
                    config_info = data['config']
                
                # Extract metrics
                if 'rounds' in data:
                    # Get final round aggregate
                    if data['rounds']:
                        final_round = data['rounds'][-1]
                        if 'aggregate_forecast' in final_round:
                            all_aggregates.append(final_round['aggregate_forecast'])
                        
                        # Extract individual expert forecasts from final round
                        if 'experts' in final_round:
                            round_forecasts = []
                            for expert_id, expert_data in final_round['experts'].items():
                                if 'prob' in expert_data:
                                    round_forecasts.append(expert_data['prob'])
                            if round_forecasts:
                                all_individual_forecasts.append(round_forecasts)
                    
                    # Extract round-specific data
                    if len(data['rounds']) >= 1:
                        round1_forecasts = []
                        if 'experts' in data['rounds'][0]:
                            for expert_id, expert_data in data['rounds'][0]['experts'].items():
                                if 'prob' in expert_data:
                                    round1_forecasts.append(expert_data['prob'])
                        if round1_forecasts:
                            all_round1_responses.extend(round1_forecasts)
                    
                    if len(data['rounds']) >= 2:
                        round2_forecasts = []
                        if 'experts' in data['rounds'][1]:
                            for expert_id, expert_data in data['rounds'][1]['experts'].items():
                                if 'prob' in expert_data:
                                    round2_forecasts.append(expert_data['prob'])
                        if round2_forecasts:
                            all_round2_responses.extend(round2_forecasts)
                
                # Store question info
                if 'question_text' in data:
                    all_questions.append({
                        'id': data.get('question_id', ''),
                        'text': data['question_text'][:100] + '...',  # Truncate for display
                        'file': json_file.name
                    })
                
                # Calculate performance metrics if we have resolution
                if 'resolution' in data and data['resolution'] is not None:
                    resolution = float(data['resolution']['resolved_to'])
                    
                    if data['rounds'] and 'aggregate_forecast' in data['rounds'][-1]:
                        forecast = data['rounds'][-1]['aggregate_forecast']
                        
                        # Brier score
                        brier = (forecast - resolution) ** 2
                        all_brier_scores.append(brier)
                        
                        # Mean Absolute Error
                        mae = abs(forecast - resolution)
                        all_mae_scores.append(mae)
                        
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        # Create summary statistics
        results = {
            'config': config_info,
            'n_questions': len(json_files),
            'n_resolved': len(all_brier_scores),
            'aggregate_forecasts': all_aggregates,
            'individual_forecasts_all': [f for sublist in all_individual_forecasts for f in sublist],
            'round1_responses': all_round1_responses,
            'round2_responses': all_round2_responses,
            'brier_scores': all_brier_scores,
            'mae_scores': all_mae_scores,
            'questions': all_questions
        }
        
        # Calculate derived metrics
        if all_aggregates:
            results['mean_aggregate'] = np.mean(all_aggregates)
            results['std_aggregate'] = np.std(all_aggregates)
        
        if all_brier_scores:
            results['mean_brier'] = np.mean(all_brier_scores)
            results['std_brier'] = np.std(all_brier_scores)
            results['brier'] = all_brier_scores  # For compatibility with plotter
        
        if all_mae_scores:
            results['mean_mae'] = np.mean(all_mae_scores)
            results['std_mae'] = np.std(all_mae_scores)
            results['mae'] = all_mae_scores  # For compatibility with plotter
        
        if all_individual_forecasts:
            # Calculate expert agreement metrics
            individual_stds = [np.std(forecasts) if len(forecasts) > 1 else 0 
                             for forecasts in all_individual_forecasts]
            results['mean_expert_std'] = np.mean(individual_stds)
            results['individual_forecasts'] = all_individual_forecasts[-1] if all_individual_forecasts else []
        
        return results
    
    def compare_variants(self) -> Tuple[Dict, Dict]:
        """
        Load and compare the two OSS-120B variants.
        
        Returns:
            Tuple of (3ktokens_results, expert_results)
        """
        print("Loading OSS-120B 3k tokens variant...")
        results_3k = self.load_delphi_results("outputs_oss_120b_3ktokens")
        
        print("Loading OSS-120B expert variant...")
        results_expert = self.load_delphi_results("outputs_oss_120b_expert")
        
        return results_3k, results_expert
    
    def print_summary_stats(self, results_3k: Dict, results_expert: Dict):
        """Print summary statistics comparison."""
        
        print("\n" + "="*80)
        print("OSS-120B VARIANT COMPARISON SUMMARY")
        print("="*80)
        
        # Configuration comparison
        print("\n📋 CONFIGURATION COMPARISON:")
        if results_3k.get('config') and results_expert.get('config'):
            config_3k = results_3k['config']
            config_expert = results_expert['config']
            
            print(f"{'Metric':<25} {'3k Tokens':<15} {'Expert':<15}")
            print("-" * 55)
            print(f"{'Model':<25} {config_3k.get('model_name', 'N/A'):<15} {config_expert.get('model_name', 'N/A'):<15}")
            print(f"{'N Experts':<25} {config_3k.get('n_experts', 'N/A'):<15} {config_expert.get('n_experts', 'N/A'):<15}")
            print(f"{'N Rounds':<25} {config_3k.get('n_rounds', 'N/A'):<15} {config_expert.get('n_rounds', 'N/A'):<15}")
        
        # Data comparison
        print(f"\n📊 DATA SUMMARY:")
        print(f"{'Metric':<25} {'3k Tokens':<15} {'Expert':<15}")
        print("-" * 55)
        print(f"{'Total Questions':<25} {results_3k.get('n_questions', 0):<15} {results_expert.get('n_questions', 0):<15}")
        print(f"{'Resolved Questions':<25} {results_3k.get('n_resolved', 0):<15} {results_expert.get('n_resolved', 0):<15}")
        
        # Performance comparison
        print(f"\n🎯 PERFORMANCE COMPARISON:")
        print(f"{'Metric':<25} {'3k Tokens':<15} {'Expert':<15} {'Difference':<15}")
        print("-" * 70)
        
        # Brier Score
        if results_3k.get('mean_brier') is not None and results_expert.get('mean_brier') is not None:
            brier_3k = results_3k['mean_brier']
            brier_expert = results_expert['mean_brier']
            brier_diff = brier_expert - brier_3k
            print(f"{'Brier Score (avg)':<25} {brier_3k:<15.4f} {brier_expert:<15.4f} {brier_diff:<+15.4f}")
        
        # MAE
        if results_3k.get('mean_mae') is not None and results_expert.get('mean_mae') is not None:
            mae_3k = results_3k['mean_mae']
            mae_expert = results_expert['mean_mae']
            mae_diff = mae_expert - mae_3k
            print(f"{'MAE (avg)':<25} {mae_3k:<15.4f} {mae_expert:<15.4f} {mae_diff:<+15.4f}")
        
        # Expert Agreement
        if results_3k.get('mean_expert_std') is not None and results_expert.get('mean_expert_std') is not None:
            std_3k = results_3k['mean_expert_std']
            std_expert = results_expert['mean_expert_std']
            std_diff = std_expert - std_3k
            print(f"{'Expert Disagreement':<25} {std_3k:<15.4f} {std_expert:<15.4f} {std_diff:<+15.4f}")
        
        # Statistical significance testing
        print(f"\n📈 STATISTICAL TESTS:")
        if (results_3k.get('brier_scores') and results_expert.get('brier_scores') and 
            len(results_3k['brier_scores']) > 1 and len(results_expert['brier_scores']) > 1):
            
            # T-test for Brier scores
            t_stat, p_value = stats.ttest_ind(results_3k['brier_scores'], results_expert['brier_scores'])
            print(f"Brier Score t-test: t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                winner = "Expert" if np.mean(results_expert['brier_scores']) < np.mean(results_3k['brier_scores']) else "3k Tokens"
                print(f"  → Significant difference (p<0.05): {winner} performs better")
            else:
                print(f"  → No significant difference (p≥0.05)")
        
        if (results_3k.get('mae_scores') and results_expert.get('mae_scores') and 
            len(results_3k['mae_scores']) > 1 and len(results_expert['mae_scores']) > 1):
            
            # T-test for MAE scores
            t_stat, p_value = stats.ttest_ind(results_3k['mae_scores'], results_expert['mae_scores'])
            print(f"MAE t-test: t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                winner = "Expert" if np.mean(results_expert['mae_scores']) < np.mean(results_3k['mae_scores']) else "3k Tokens"
                print(f"  → Significant difference (p<0.05): {winner} performs better")
            else:
                print(f"  → No significant difference (p≥0.05)")
    
    def create_comparison_plots(self, results_3k: Dict, results_expert: Dict):
        """Create comprehensive comparison plots."""
        
        # Prepare data for plotter
        combined_results = {
            "OSS-120B (3k tokens)": results_3k,
            "OSS-120B (Expert)": results_expert
        }
        
        print("\n🎨 Creating comparison plots...")
        
        # 1. Performance comparison
        print("  1. Performance metrics comparison...")
        self.plotter.plot_performance_comparison(
            combined_results,
            metrics=['brier', 'mae'],
            title="OSS-120B Variants: Performance Comparison"
        )
        
        # 2. Expert agreement analysis
        print("  2. Expert agreement analysis...")
        self.plotter.plot_expert_agreement(
            combined_results,
            title="OSS-120B Variants: Expert Agreement Analysis"
        )
        
        # 3. Convergence analysis (if multi-round data available)
        if (results_3k.get('round1_responses') and results_3k.get('round2_responses') and
            results_expert.get('round1_responses') and results_expert.get('round2_responses')):
            print("  3. Convergence analysis...")
            self.plotter.plot_convergence_analysis(
                combined_results,
                title="OSS-120B Variants: Delphi Convergence Analysis"
            )
        
        # 4. Create detailed comparison plot
        print("  4. Creating detailed comparison...")
        self.create_detailed_comparison_plot(results_3k, results_expert)
        
        # 5. Performance distribution analysis
        print("  5. Performance distribution analysis...")
        self.create_performance_distribution_plot(results_3k, results_expert)
        
        print("✅ All plots created successfully!")
    
    def create_detailed_comparison_plot(self, results_3k: Dict, results_expert: Dict):
        """Create a detailed side-by-side comparison plot."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('OSS-120B Detailed Variant Comparison', fontsize=16, fontweight='bold')
        
        # 1. Brier Score Comparison
        ax = axes[0, 0]
        if results_3k.get('brier_scores') and results_expert.get('brier_scores'):
            data = [results_3k['brier_scores'], results_expert['brier_scores']]
            labels = ['3k Tokens', 'Expert']
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('skyblue')
            bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_title('Brier Score Distribution', fontweight='bold')
        ax.set_ylabel('Brier Score')
        ax.grid(True, alpha=0.3)
        
        # 2. MAE Comparison
        ax = axes[0, 1]
        if results_3k.get('mae_scores') and results_expert.get('mae_scores'):
            data = [results_3k['mae_scores'], results_expert['mae_scores']]
            labels = ['3k Tokens', 'Expert']
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('skyblue')
            bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_title('MAE Distribution', fontweight='bold')
        ax.set_ylabel('Mean Absolute Error')
        ax.grid(True, alpha=0.3)
        
        # 3. Aggregate Forecast Distribution
        ax = axes[0, 2]
        if results_3k.get('aggregate_forecasts') and results_expert.get('aggregate_forecasts'):
            ax.hist(results_3k['aggregate_forecasts'], alpha=0.6, label='3k Tokens', 
                   bins=20, color='skyblue', edgecolor='navy')
            ax.hist(results_expert['aggregate_forecasts'], alpha=0.6, label='Expert', 
                   bins=20, color='lightcoral', edgecolor='darkred')
        ax.set_title('Aggregate Forecast Distribution', fontweight='bold')
        ax.set_xlabel('Forecast Probability')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Expert Disagreement
        ax = axes[1, 0]
        if (results_3k.get('individual_forecasts_all') and results_expert.get('individual_forecasts_all')):
            # Calculate variance for each question set
            variance_3k = []
            variance_expert = []
            
            # For 3k tokens
            if 'individual_forecasts' in results_3k:
                variance_3k = [np.var(forecasts) if len(forecasts) > 1 else 0 
                              for forecasts in [results_3k['individual_forecasts']]]
            
            # For expert
            if 'individual_forecasts' in results_expert:
                variance_expert = [np.var(forecasts) if len(forecasts) > 1 else 0 
                                  for forecasts in [results_expert['individual_forecasts']]]
            
            if variance_3k and variance_expert:
                x = ['3k Tokens', 'Expert']
                y = [np.mean(variance_3k), np.mean(variance_expert)]
                colors = ['skyblue', 'lightcoral']
                bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black')
                
                # Add value labels
                for bar, val in zip(bars, y):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title('Expert Disagreement (Variance)', fontweight='bold')
        ax.set_ylabel('Variance')
        ax.grid(True, alpha=0.3)
        
        # 5. Number of Questions and Coverage
        ax = axes[1, 1]
        categories = ['Total Questions', 'Resolved Questions', 'Performance Data']
        counts_3k = [
            results_3k.get('n_questions', 0),
            results_3k.get('n_resolved', 0),
            len(results_3k.get('brier_scores', []))
        ]
        counts_expert = [
            results_expert.get('n_questions', 0),
            results_expert.get('n_resolved', 0),
            len(results_expert.get('brier_scores', []))
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, counts_3k, width, label='3k Tokens', 
                      color='skyblue', alpha=0.7, edgecolor='navy')
        bars2 = ax.bar(x + width/2, counts_expert, width, label='Expert', 
                      color='lightcoral', alpha=0.7, edgecolor='darkred')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Data Coverage Comparison', fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Summary Metrics Table
        ax = axes[1, 2]
        ax.axis('off')
        
        # Create summary table
        summary_data = []
        metrics = ['Mean Brier', 'Mean MAE', 'Expert Std', 'N Questions']
        
        for metric in metrics:
            row = [metric]
            
            # 3k tokens values
            if metric == 'Mean Brier':
                val = f"{results_3k.get('mean_brier', 0):.4f}"
            elif metric == 'Mean MAE':
                val = f"{results_3k.get('mean_mae', 0):.4f}"
            elif metric == 'Expert Std':
                val = f"{results_3k.get('mean_expert_std', 0):.4f}"
            elif metric == 'N Questions':
                val = str(results_3k.get('n_questions', 0))
            row.append(val)
            
            # Expert values
            if metric == 'Mean Brier':
                val = f"{results_expert.get('mean_brier', 0):.4f}"
            elif metric == 'Mean MAE':
                val = f"{results_expert.get('mean_mae', 0):.4f}"
            elif metric == 'Expert Std':
                val = f"{results_expert.get('mean_expert_std', 0):.4f}"
            elif metric == 'N Questions':
                val = str(results_expert.get('n_questions', 0))
            row.append(val)
            
            summary_data.append(row)
        
        table = ax.table(cellText=summary_data, 
                        colLabels=['Metric', '3k Tokens', 'Expert'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.plotter.output_dir / "detailed_variant_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_performance_distribution_plot(self, results_3k: Dict, results_expert: Dict):
        """Create performance distribution comparison plot."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('OSS-120B Performance Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Brier Score Distribution
        ax = axes[0, 0]
        if results_3k.get('brier_scores') and results_expert.get('brier_scores'):
            ax.hist(results_3k['brier_scores'], alpha=0.6, label='3k Tokens', 
                   bins=15, color='skyblue', edgecolor='navy', density=True)
            ax.hist(results_expert['brier_scores'], alpha=0.6, label='Expert', 
                   bins=15, color='lightcoral', edgecolor='darkred', density=True)
            
            # Add mean lines
            ax.axvline(np.mean(results_3k['brier_scores']), color='blue', 
                      linestyle='--', alpha=0.8, label='3k Mean')
            ax.axvline(np.mean(results_expert['brier_scores']), color='red', 
                      linestyle='--', alpha=0.8, label='Expert Mean')
        
        ax.set_title('Brier Score Distribution', fontweight='bold')
        ax.set_xlabel('Brier Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. MAE Distribution
        ax = axes[0, 1]
        if results_3k.get('mae_scores') and results_expert.get('mae_scores'):
            ax.hist(results_3k['mae_scores'], alpha=0.6, label='3k Tokens', 
                   bins=15, color='skyblue', edgecolor='navy', density=True)
            ax.hist(results_expert['mae_scores'], alpha=0.6, label='Expert', 
                   bins=15, color='lightcoral', edgecolor='darkred', density=True)
            
            # Add mean lines
            ax.axvline(np.mean(results_3k['mae_scores']), color='blue', 
                      linestyle='--', alpha=0.8, label='3k Mean')
            ax.axvline(np.mean(results_expert['mae_scores']), color='red', 
                      linestyle='--', alpha=0.8, label='Expert Mean')
        
        ax.set_title('MAE Distribution', fontweight='bold')
        ax.set_xlabel('Mean Absolute Error')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Q-Q Plot for Brier Scores
        ax = axes[1, 0]
        if results_3k.get('brier_scores') and results_expert.get('brier_scores'):
            # Create Q-Q plot
            brier_3k_sorted = np.sort(results_3k['brier_scores'])
            brier_expert_sorted = np.sort(results_expert['brier_scores'])
            
            # Interpolate to same length for comparison
            n_points = min(len(brier_3k_sorted), len(brier_expert_sorted))
            quantiles = np.linspace(0, 1, n_points)
            
            q_3k = np.quantile(brier_3k_sorted, quantiles)
            q_expert = np.quantile(brier_expert_sorted, quantiles)
            
            ax.scatter(q_3k, q_expert, alpha=0.6, color='purple')
            
            # Add diagonal line for reference
            min_val = min(np.min(q_3k), np.min(q_expert))
            max_val = max(np.max(q_3k), np.max(q_expert))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        ax.set_title('Brier Score Q-Q Plot', fontweight='bold')
        ax.set_xlabel('3k Tokens Quantiles')
        ax.set_ylabel('Expert Quantiles')
        ax.grid(True, alpha=0.3)
        
        # 4. Performance correlation plot
        ax = axes[1, 1]
        if (results_3k.get('brier_scores') and results_expert.get('brier_scores') and
            len(results_3k['brier_scores']) == len(results_expert['brier_scores'])):
            
            ax.scatter(results_3k['brier_scores'], results_expert['brier_scores'], 
                      alpha=0.6, color='green')
            
            # Add correlation line
            if len(results_3k['brier_scores']) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    results_3k['brier_scores'], results_expert['brier_scores'])
                
                x_line = np.array(ax.get_xlim())
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'r-', alpha=0.8)
                
                # Add correlation coefficient
                ax.text(0.05, 0.95, f'r = {r_value:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add diagonal line
            min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
            max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
        
        ax.set_title('Performance Correlation', fontweight='bold')
        ax.set_xlabel('3k Tokens Brier Score')
        ax.set_ylabel('Expert Brier Score')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.plotter.output_dir / "performance_distribution_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


def main():
    """Main function to run the comparison analysis."""
    
    print("🔍 OSS-120B Variant Comparison Analysis")
    print("=" * 50)
    
    # Initialize comparator
    comparator = OSS120BVariantComparator()
    
    # Load and compare variants
    results_3k, results_expert = comparator.compare_variants()
    
    if not results_3k and not results_expert:
        print("❌ No results found. Please check the output directories exist.")
        return
    
    # Print summary statistics
    comparator.print_summary_stats(results_3k, results_expert)
    
    # Create comparison plots
    if results_3k or results_expert:
        comparator.create_comparison_plots(results_3k, results_expert)
    
    print(f"\n📁 All plots saved to: {comparator.plotter.output_dir}")
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()