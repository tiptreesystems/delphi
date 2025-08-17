#!/usr/bin/env python3
"""
Comprehensive plotting functions for comparing different experimental settings in Delphi.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from pathlib import Path
from scipy import stats
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DelphiComparisonPlotter:
    """Class for creating comparison plots of Delphi experimental results."""
    
    def __init__(self, output_dir: str = "plots"):
        """Initialize the plotter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_results(self, results_path: str) -> Dict:
        """Load results from JSON file."""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def load_multiple_results(self, results_paths: Dict[str, str]) -> Dict[str, Dict]:
        """Load multiple result files with labels."""
        results = {}
        for label, path in results_paths.items():
            if os.path.exists(path):
                results[label] = self.load_results(path)
            else:
                print(f"Warning: {path} not found")
        return results
    
    def plot_performance_comparison(self, 
                                   results: Dict[str, Dict],
                                   metrics: List[str] = ['brier', 'mae'],
                                   title: str = "Model Performance Comparison"):
        """
        Create a bar chart comparing different models/settings across metrics.
        
        Args:
            results: Dictionary with setting names as keys and result dicts as values
            metrics: List of metrics to compare
            title: Plot title
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            # Extract metric values for each setting
            settings = []
            values = []
            errors = []
            
            for setting_name, setting_results in results.items():
                if metric in setting_results:
                    if isinstance(setting_results[metric], list):
                        vals = setting_results[metric]
                        values.append(np.mean(vals))
                        errors.append(np.std(vals) / np.sqrt(len(vals)))  # SEM
                    else:
                        values.append(setting_results[metric])
                        errors.append(0)
                    settings.append(setting_name)
            
            # Create bar plot
            bars = ax.bar(settings, values, yerr=errors, capsize=5, alpha=0.7)
            
            # Color bars by performance (lower is better for these metrics)
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_title(f'{metric.upper()} Score', fontsize=14, fontweight='bold')
            ax.set_xlabel('Setting', fontsize=12)
            ax.set_ylabel(f'{metric.upper()}', fontsize=12)
            ax.set_ylim(0, max(values) * 1.2 if values else 1)
            
            # Add value labels on bars
            for bar, value, error in zip(bars, values, errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + error,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Rotate x-labels if needed
            if len(settings) > 3:
                ax.set_xticklabels(settings, rotation=45, ha='right')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"performance_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_convergence_analysis(self,
                                 results: Dict[str, Dict],
                                 title: str = "Delphi Convergence Analysis"):
        """
        Plot how forecasts converge across Delphi rounds.
        
        Args:
            results: Dictionary with results containing round-by-round data
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        for setting_name, setting_results in results.items():
            if 'round1_responses' in setting_results and 'round2_responses' in setting_results:
                round1 = setting_results.get('round1_responses', [])
                round2 = setting_results.get('round2_responses', [])
                
                if round1 and round2:
                    # Calculate variance reduction
                    var1 = np.var(round1) if len(round1) > 1 else 0
                    var2 = np.var(round2) if len(round2) > 1 else 0
                    
                    # Plot variance by round
                    ax1.plot([1, 2], [var1, var2], marker='o', label=setting_name, linewidth=2)
                    
                    # Plot distribution change
                    positions = [1, 2]
                    data = [round1, round2]
                    parts = ax2.violinplot(data, positions=positions, widths=0.7,
                                          showmeans=True, showextrema=True)
                    
                    # Color violin plots
                    for pc in parts['bodies']:
                        pc.set_alpha(0.5)
        
        ax1.set_xlabel('Delphi Round', fontsize=12)
        ax1.set_ylabel('Variance', fontsize=12)
        ax1.set_title('Variance Reduction Across Rounds', fontsize=14, fontweight='bold')
        ax1.set_xticks([1, 2])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Delphi Round', fontsize=12)
        ax2.set_ylabel('Forecast Value', fontsize=12)
        ax2.set_title('Forecast Distribution by Round', fontsize=14, fontweight='bold')
        ax2.set_xticks([1, 2])
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"convergence_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_calibration_curves(self,
                               results: Dict[str, Dict],
                               n_bins: int = 10,
                               title: str = "Calibration Analysis"):
        """
        Plot calibration curves for different settings.
        
        Args:
            results: Dictionary with forecasts and outcomes
            n_bins: Number of bins for calibration
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        for setting_name, setting_results in results.items():
            if 'forecasts' in setting_results and 'outcomes' in setting_results:
                forecasts = np.array(setting_results['forecasts'])
                outcomes = np.array(setting_results['outcomes'])
                
                # Calculate calibration
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                bin_means = []
                bin_counts = []
                
                for i in range(n_bins):
                    mask = (forecasts >= bin_edges[i]) & (forecasts < bin_edges[i+1])
                    if np.sum(mask) > 0:
                        bin_means.append(np.mean(outcomes[mask]))
                        bin_counts.append(np.sum(mask))
                    else:
                        bin_means.append(np.nan)
                        bin_counts.append(0)
                
                # Plot calibration curve
                valid_bins = ~np.isnan(bin_means)
                ax1.plot(bin_centers[valid_bins], np.array(bin_means)[valid_bins],
                        marker='o', label=setting_name, linewidth=2)
                
                # Plot histogram of predictions
                ax2.hist(forecasts, bins=bin_edges, alpha=0.5, label=setting_name,
                        edgecolor='black', linewidth=1)
        
        # Perfect calibration line
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        
        ax1.set_xlabel('Forecast Probability', fontsize=12)
        ax1.set_ylabel('Observed Frequency', fontsize=12)
        ax1.set_title('Calibration Curves', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        ax2.set_xlabel('Forecast Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Forecasts', fontsize=14, fontweight='bold')
        ax2.legend()
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"calibration_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_expert_agreement(self,
                             results: Dict[str, Dict],
                             title: str = "Expert Agreement Analysis"):
        """
        Plot expert agreement metrics across different settings.
        
        Args:
            results: Dictionary with expert forecasts
            title: Plot title
        """
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        agreement_data = []
        
        for setting_name, setting_results in results.items():
            if 'individual_forecasts' in setting_results:
                forecasts = setting_results['individual_forecasts']
                
                if isinstance(forecasts, list) and len(forecasts) > 1:
                    # Calculate agreement metrics
                    mean_forecast = np.mean(forecasts)
                    std_forecast = np.std(forecasts)
                    range_forecast = np.max(forecasts) - np.min(forecasts)
                    
                    agreement_data.append({
                        'Setting': setting_name,
                        'Mean': mean_forecast,
                        'Std Dev': std_forecast,
                        'Range': range_forecast,
                        'Forecasts': forecasts
                    })
        
        if agreement_data:
            df = pd.DataFrame(agreement_data)
            
            # Plot 1: Standard deviation by setting
            ax1.bar(df['Setting'], df['Std Dev'], alpha=0.7, color='skyblue', edgecolor='navy')
            ax1.set_xlabel('Setting', fontsize=12)
            ax1.set_ylabel('Standard Deviation', fontsize=12)
            ax1.set_title('Forecast Variability', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Range by setting
            ax2.bar(df['Setting'], df['Range'], alpha=0.7, color='lightcoral', edgecolor='darkred')
            ax2.set_xlabel('Setting', fontsize=12)
            ax2.set_ylabel('Range (Max - Min)', fontsize=12)
            ax2.set_title('Forecast Range', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Box plot of all forecasts
            all_forecasts = []
            labels = []
            for _, row in df.iterrows():
                all_forecasts.append(row['Forecasts'])
                labels.append(row['Setting'])
            
            bp = ax3.boxplot(all_forecasts, labels=labels, patch_artist=True)
            
            # Color the box plots
            colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.set_xlabel('Setting', fontsize=12)
            ax3.set_ylabel('Forecast Value', fontsize=12)
            ax3.set_title('Distribution of Individual Expert Forecasts', fontsize=14, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"expert_agreement.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_model_comparison_matrix(self,
                                    results: Dict[str, Dict],
                                    metrics: List[str] = ['brier', 'mae', 'calibration'],
                                    title: str = "Model Comparison Matrix"):
        """
        Create a heatmap matrix comparing multiple models across multiple metrics.
        
        Args:
            results: Dictionary with model results
            metrics: List of metrics to compare
            title: Plot title
        """
        # Prepare data matrix
        models = list(results.keys())
        data_matrix = []
        
        for model in models:
            row = []
            for metric in metrics:
                if metric in results[model]:
                    value = results[model][metric]
                    if isinstance(value, list):
                        value = np.mean(value)
                    row.append(value)
                else:
                    row.append(np.nan)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Normalize each metric to 0-1 range for comparison
        for j in range(data_matrix.shape[1]):
            col = data_matrix[:, j]
            if not np.all(np.isnan(col)):
                min_val = np.nanmin(col)
                max_val = np.nanmax(col)
                if max_val > min_val:
                    data_matrix[:, j] = (col - min_val) / (max_val - min_val)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # For metrics where lower is better, invert the colormap
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(models)
        
        # Rotate the tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Score (lower is better)', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(metrics)):
                if not np.isnan(data_matrix[i, j]):
                    text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"model_comparison_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_panel_size_analysis(self,
                                results_by_size: Dict[int, Dict],
                                metrics: List[str] = ['brier', 'mae'],
                                title: str = "Panel Size Impact Analysis"):
        """
        Analyze how panel size affects performance.
        
        Args:
            results_by_size: Dictionary with panel sizes as keys
            metrics: Metrics to analyze
            title: Plot title
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(7*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            sizes = sorted(results_by_size.keys())
            means = []
            stds = []
            
            for size in sizes:
                if metric in results_by_size[size]:
                    values = results_by_size[size][metric]
                    if isinstance(values, list):
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                    else:
                        means.append(values)
                        stds.append(0)
                else:
                    means.append(np.nan)
                    stds.append(0)
            
            # Plot with error bars
            ax.errorbar(sizes, means, yerr=stds, marker='o', markersize=8,
                       linewidth=2, capsize=5, capthick=2)
            
            # Add trend line
            valid_idx = ~np.isnan(means)
            if np.sum(valid_idx) > 1:
                z = np.polyfit(np.array(sizes)[valid_idx], np.array(means)[valid_idx], 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(min(sizes), max(sizes), 100)
                ax.plot(x_smooth, p(x_smooth), '--', alpha=0.5, color='red',
                       label='Trend')
            
            ax.set_xlabel('Number of Experts', fontsize=12)
            ax.set_ylabel(f'{metric.upper()}', fontsize=12)
            ax.set_title(f'{metric.upper()} vs Panel Size', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Mark optimal size
            if means:
                optimal_idx = np.nanargmin(means)
                optimal_size = sizes[optimal_idx]
                ax.axvline(x=optimal_size, color='green', linestyle=':', alpha=0.5)
                ax.text(optimal_size, ax.get_ylim()[1]*0.95, f'Optimal: {optimal_size}',
                       ha='center', fontsize=10, color='green')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"panel_size_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_comprehensive_report(self,
                                  results_dict: Dict[str, Dict],
                                  title: str = "Delphi Experiment Comprehensive Report"):
        """
        Create a comprehensive multi-panel report with all analyses.
        
        Args:
            results_dict: Dictionary of all results to compare
            title: Overall report title
        """
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Performance comparison
        ax1 = fig.add_subplot(gs[0, :])
        self._add_performance_subplot(ax1, results_dict)
        
        # Calibration
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        self._add_calibration_subplots(ax2, ax3, results_dict)
        
        # Expert agreement
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])
        self._add_agreement_subplots(ax4, ax5, results_dict)
        
        # Summary statistics
        ax6 = fig.add_subplot(gs[3, :])
        self._add_summary_table(ax6, results_dict)
        
        plt.suptitle(title, fontsize=20, fontweight='bold', y=0.995)
        
        # Save figure
        save_path = self.output_dir / f"comprehensive_report.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def _add_performance_subplot(self, ax, results):
        """Helper to add performance comparison to subplot."""
        metrics = ['brier', 'mae']
        settings = list(results.keys())
        
        x = np.arange(len(settings))
        width = 0.35
        
        for i, metric in enumerate(metrics):
            values = []
            for setting in settings:
                if metric in results[setting]:
                    val = results[setting][metric]
                    if isinstance(val, list):
                        val = np.mean(val)
                    values.append(val)
                else:
                    values.append(0)
            
            offset = width * (i - 0.5)
            bars = ax.bar(x + offset, values, width, label=metric.upper())
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Setting', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(settings, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _add_calibration_subplots(self, ax1, ax2, results):
        """Helper to add calibration plots to subplots."""
        # Implement calibration curve logic
        ax1.set_title('Calibration Curves', fontsize=12, fontweight='bold')
        ax2.set_title('Forecast Distributions', fontsize=12, fontweight='bold')
        
        # Add perfect calibration line to ax1
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax1.set_xlabel('Forecast')
        ax1.set_ylabel('Observed')
        ax1.legend()
        
    def _add_agreement_subplots(self, ax1, ax2, results):
        """Helper to add agreement analysis to subplots."""
        ax1.set_title('Expert Variance', fontsize=12, fontweight='bold')
        ax2.set_title('Convergence Rate', fontsize=12, fontweight='bold')
        
    def _add_summary_table(self, ax, results):
        """Helper to add summary statistics table."""
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary data
        summary_data = []
        for setting, data in results.items():
            row = [setting]
            for metric in ['brier', 'mae', 'n_experts', 'n_questions']:
                if metric in data:
                    val = data[metric]
                    if isinstance(val, list):
                        val = f"{np.mean(val):.3f} ± {np.std(val):.3f}"
                    elif isinstance(val, float):
                        val = f"{val:.3f}"
                    else:
                        val = str(val)
                    row.append(val)
                else:
                    row.append('-')
            summary_data.append(row)
        
        # Create table
        columns = ['Setting', 'Brier Score', 'MAE', 'N Experts', 'N Questions']
        table = ax.table(cellText=summary_data, colLabels=columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)


def generate_example_usage():
    """Generate example usage with synthetic data."""
    
    # Create synthetic results for demonstration
    np.random.seed(42)
    
    results = {
        "GPT-4": {
            "brier": list(np.random.beta(2, 5, 20) * 0.3),
            "mae": list(np.random.beta(2, 5, 20) * 0.4),
            "forecasts": list(np.random.beta(2, 2, 100)),
            "outcomes": list(np.random.binomial(1, 0.5, 100)),
            "round1_responses": list(np.random.beta(2, 2, 10)),
            "round2_responses": list(np.random.beta(3, 3, 10)),
            "individual_forecasts": list(np.random.beta(2, 2, 5)),
            "n_experts": 5,
            "n_questions": 20
        },
        "Claude-Sonnet": {
            "brier": list(np.random.beta(2, 6, 20) * 0.25),
            "mae": list(np.random.beta(2, 6, 20) * 0.35),
            "forecasts": list(np.random.beta(3, 3, 100)),
            "outcomes": list(np.random.binomial(1, 0.5, 100)),
            "round1_responses": list(np.random.beta(2, 3, 10)),
            "round2_responses": list(np.random.beta(4, 4, 10)),
            "individual_forecasts": list(np.random.beta(3, 3, 5)),
            "n_experts": 5,
            "n_questions": 20
        },
        "Llama-70B": {
            "brier": list(np.random.beta(2, 4, 20) * 0.35),
            "mae": list(np.random.beta(2, 4, 20) * 0.45),
            "forecasts": list(np.random.beta(2, 3, 100)),
            "outcomes": list(np.random.binomial(1, 0.5, 100)),
            "round1_responses": list(np.random.beta(3, 2, 10)),
            "round2_responses": list(np.random.beta(3, 3, 10)),
            "individual_forecasts": list(np.random.beta(2, 3, 7)),
            "n_experts": 7,
            "n_questions": 20
        }
    }
    
    # Panel size analysis data
    panel_size_results = {}
    for size in [1, 3, 5, 7, 10, 15]:
        panel_size_results[size] = {
            "brier": list(np.random.beta(2, 5, 10) * (0.3 + 0.02 * np.log(size))),
            "mae": list(np.random.beta(2, 5, 10) * (0.4 + 0.01 * np.log(size)))
        }
    
    return results, panel_size_results


if __name__ == "__main__":
    # Create plotter instance
    plotter = DelphiComparisonPlotter(output_dir="plots/comparison_plots")
    
    # Generate example data
    print("Generating example data...")
    results, panel_size_results = generate_example_usage()
    
    # Create various plots
    print("\n1. Creating performance comparison plot...")
    plotter.plot_performance_comparison(results, 
                                       metrics=['brier', 'mae'],
                                       title="Model Performance Comparison")
    
    print("\n2. Creating convergence analysis plot...")
    plotter.plot_convergence_analysis(results,
                                     title="Delphi Convergence Analysis")
    
    print("\n3. Creating calibration curves...")
    plotter.plot_calibration_curves(results,
                                   title="Model Calibration Analysis")
    
    print("\n4. Creating expert agreement analysis...")
    plotter.plot_expert_agreement(results,
                                 title="Expert Agreement Patterns")
    
    print("\n5. Creating model comparison matrix...")
    plotter.plot_model_comparison_matrix(results,
                                        metrics=['brier', 'mae'],
                                        title="Model Performance Matrix")
    
    print("\n6. Creating panel size analysis...")
    plotter.plot_panel_size_analysis(panel_size_results,
                                    metrics=['brier', 'mae'],
                                    title="Optimal Panel Size Analysis")
    
    print("\n7. Creating comprehensive report...")
    plotter.create_comprehensive_report(results,
                                      title="Delphi Experiment Report - All Models")
    
    print("\n✅ All plots created successfully!")
    print(f"Plots saved to: {plotter.output_dir}")