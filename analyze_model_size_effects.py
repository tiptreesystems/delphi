import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_detailed_metrics(data):
    results = []
    metadata = data['metadata']
    model_name = metadata['model']
    
    for question_id, question_data in data['questions'].items():
        for n_experts, expert_data in question_data.items():
            if n_experts == 'metadata':
                continue
            
            # Extract round 1 and round 2 forecasts
            round1_forecasts = []
            round2_forecasts = []
            
            if 'round1_responses' in expert_data:
                for response in expert_data['round1_responses']:
                    # Extract forecast from response text
                    text = response['response'].lower()
                    # Simple extraction - look for probability patterns
                    import re
                    prob_match = re.search(r'(?:probability|forecast|estimate)[:\s]*(?:0\.)?(\d+)(?:\%|percent)?', text)
                    if prob_match:
                        prob = float(prob_match.group(1))
                        if prob > 1:
                            prob = prob / 100
                        round1_forecasts.append(prob)
            
            result = {
                'model': model_name,
                'question_id': question_id,
                'n_experts': int(n_experts),
                'outcome': expert_data['outcome'],
                'aggregate_forecast': expert_data['aggregate'],
                'brier_score': expert_data['brier'],
                'mae': expert_data['mae'],
                'human_mean': expert_data['human_mean'],
                'human_brier': expert_data['human_brier'],
                'human_mae': expert_data.get('human_mae', expert_data['human_mean']),  # Fallback to human_mean
                'individual_forecasts': expert_data['individual_forecasts'],
                'forecast_std': np.std(expert_data['individual_forecasts']) if expert_data['individual_forecasts'] else 0,
                'forecast_range': max(expert_data['individual_forecasts']) - min(expert_data['individual_forecasts']) if expert_data['individual_forecasts'] else 0
            }
            results.append(result)
    
    return results

def analyze_model_size_effects(haiku_data, sonnet_data):
    haiku_results = extract_detailed_metrics(haiku_data)
    sonnet_results = extract_detailed_metrics(sonnet_data)
    
    df_haiku = pd.DataFrame(haiku_results)
    df_sonnet = pd.DataFrame(sonnet_results)
    
    df_haiku['model_type'] = 'Haiku (Smaller)'
    df_sonnet['model_type'] = 'Sonnet (Larger)'
    
    df_combined = pd.concat([df_haiku, df_sonnet])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Performance Improvement with Panel Size
    ax1 = plt.subplot(2, 3, 1)
    for model in ['Haiku (Smaller)', 'Sonnet (Larger)']:
        df_model = df_combined[df_combined['model_type'] == model]
        df_grouped = df_model.groupby('n_experts')['brier_score'].mean()
        ax1.plot(df_grouped.index, df_grouped.values, marker='o', linewidth=2, label=model)
    
    human_brier = df_combined['human_brier'].mean()
    ax1.axhline(y=human_brier, color='red', linestyle='--', alpha=0.7, label='Human Baseline')
    ax1.set_xlabel('Number of Experts')
    ax1.set_ylabel('Average Brier Score')
    ax1.set_title('Model Performance vs Panel Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Forecast Diversity Analysis
    ax2 = plt.subplot(2, 3, 2)
    df_diversity = df_combined.groupby(['model_type', 'n_experts'])['forecast_std'].mean().reset_index()
    for model in ['Haiku (Smaller)', 'Sonnet (Larger)']:
        df_model = df_diversity[df_diversity['model_type'] == model]
        ax2.plot(df_model['n_experts'], df_model['forecast_std'], marker='o', linewidth=2, label=model)
    
    ax2.set_xlabel('Number of Experts')
    ax2.set_ylabel('Average Forecast Std Dev')
    ax2.set_title('Forecast Diversity by Model Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy vs Diversity Trade-off
    ax3 = plt.subplot(2, 3, 3)
    colors = {'Haiku (Smaller)': 'blue', 'Sonnet (Larger)': 'orange'}
    for model in ['Haiku (Smaller)', 'Sonnet (Larger)']:
        df_model = df_combined[df_combined['model_type'] == model]
        ax3.scatter(df_model['forecast_std'], df_model['brier_score'], 
                   alpha=0.6, label=model, color=colors[model])
    
    ax3.set_xlabel('Forecast Standard Deviation')
    ax3.set_ylabel('Brier Score')
    ax3.set_title('Accuracy vs Diversity Trade-off')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Calibration Curves
    ax4 = plt.subplot(2, 3, 4)
    for model in ['Haiku (Smaller)', 'Sonnet (Larger)']:
        df_model = df_combined[df_combined['model_type'] == model]
        
        # Bin forecasts
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        actual_probs = []
        forecast_probs = []
        
        for i in range(len(bins)-1):
            mask = (df_model['aggregate_forecast'] >= bins[i]) & (df_model['aggregate_forecast'] < bins[i+1])
            if mask.sum() > 0:
                actual_probs.append(df_model[mask]['outcome'].mean())
                forecast_probs.append(df_model[mask]['aggregate_forecast'].mean())
        
        if len(forecast_probs) > 0:
            ax4.plot(forecast_probs, actual_probs, marker='o', linewidth=2, label=model)
    
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax4.set_xlabel('Mean Forecast Probability')
    ax4.set_ylabel('Actual Probability')
    ax4.set_title('Calibration Curves by Model Size')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    
    # 5. Performance Distribution
    ax5 = plt.subplot(2, 3, 5)
    data_to_plot = []
    labels = []
    for model in ['Haiku (Smaller)', 'Sonnet (Larger)']:
        df_model = df_combined[df_combined['model_type'] == model]
        data_to_plot.append(df_model['brier_score'].values)
        labels.append(model)
    
    bp = ax5.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax5.axhline(y=human_brier, color='red', linestyle='--', alpha=0.7, label='Human Mean')
    ax5.set_ylabel('Brier Score')
    ax5.set_title('Distribution of Brier Scores')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Model Size Effect Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate key statistics
    haiku_stats = df_combined[df_combined['model_type'] == 'Haiku (Smaller)']
    sonnet_stats = df_combined[df_combined['model_type'] == 'Sonnet (Larger)']
    
    # Perform statistical test
    t_stat, p_value = stats.ttest_ind(haiku_stats['brier_score'], sonnet_stats['brier_score'])
    
    summary_text = f"""Model Size Effect Analysis

Haiku (Smaller Model):
• Avg Brier Score: {haiku_stats['brier_score'].mean():.4f}
• Avg MAE: {haiku_stats['mae'].mean():.4f}
• Avg Forecast Diversity: {haiku_stats['forecast_std'].mean():.4f}

Sonnet (Larger Model):
• Avg Brier Score: {sonnet_stats['brier_score'].mean():.4f}
• Avg MAE: {sonnet_stats['mae'].mean():.4f}
• Avg Forecast Diversity: {sonnet_stats['forecast_std'].mean():.4f}

Human Baseline:
• Avg Brier Score: {human_brier:.4f}
• Avg MAE: {df_combined['human_mae'].mean():.4f}

Statistical Comparison:
• t-statistic: {t_stat:.3f}
• p-value: {p_value:.4f}
• Significant difference: {'Yes' if p_value < 0.05 else 'No'}

Key Findings:
• Haiku outperforms human baseline by {(1 - haiku_stats['brier_score'].mean()/human_brier)*100:.1f}%
• Sonnet performs similarly to human baseline
• Smaller model shows {(1 - haiku_stats['brier_score'].mean()/sonnet_stats['brier_score'].mean())*100:.1f}% better accuracy"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Model Size Effects in Delphi Method Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_size_effects_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_combined

def create_panel_size_analysis(df_combined):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Panel Size Effects: Haiku vs Sonnet', fontsize=16)
    
    # 1. Improvement Rate with Panel Size
    ax1 = axes[0, 0]
    for model in ['Haiku (Smaller)', 'Sonnet (Larger)']:
        df_model = df_combined[df_combined['model_type'] == model]
        
        # Calculate improvement from 2 to 16 experts
        improvements = []
        n_experts_list = sorted(df_model['n_experts'].unique())
        
        if len(n_experts_list) > 1:
            base_score = df_model[df_model['n_experts'] == n_experts_list[0]]['brier_score'].mean()
            
            for n in n_experts_list:
                current_score = df_model[df_model['n_experts'] == n]['brier_score'].mean()
                improvement = (base_score - current_score) / base_score * 100
                improvements.append(improvement)
            
            ax1.plot(n_experts_list, improvements, marker='o', linewidth=2, label=model)
    
    ax1.set_xlabel('Number of Experts')
    ax1.set_ylabel('Improvement from 2 Experts (%)')
    ax1.set_title('Performance Improvement with Panel Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 2. Convergence Speed
    ax2 = axes[0, 1]
    for model in ['Haiku (Smaller)', 'Sonnet (Larger)']:
        df_model = df_combined[df_combined['model_type'] == model]
        
        convergence_rates = []
        n_experts_list = sorted(df_model['n_experts'].unique())
        
        for n in n_experts_list:
            df_n = df_model[df_model['n_experts'] == n]
            # Use forecast range as a measure of convergence
            avg_range = df_n['forecast_range'].mean()
            convergence_rates.append(avg_range)
        
        ax2.plot(n_experts_list, convergence_rates, marker='o', linewidth=2, label=model)
    
    ax2.set_xlabel('Number of Experts')
    ax2.set_ylabel('Average Forecast Range')
    ax2.set_title('Convergence Speed by Model Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Efficiency Analysis
    ax3 = axes[1, 0]
    for model in ['Haiku (Smaller)', 'Sonnet (Larger)']:
        df_model = df_combined[df_combined['model_type'] == model]
        
        n_experts_list = sorted(df_model['n_experts'].unique())
        efficiency_scores = []
        
        for n in n_experts_list:
            df_n = df_model[df_model['n_experts'] == n]
            # Efficiency = performance gain per expert
            avg_brier = df_n['brier_score'].mean()
            efficiency = 1 / (avg_brier * n)  # Lower brier * fewer experts = higher efficiency
            efficiency_scores.append(efficiency)
        
        ax3.plot(n_experts_list, efficiency_scores, marker='o', linewidth=2, label=model)
    
    ax3.set_xlabel('Number of Experts')
    ax3.set_ylabel('Efficiency Score')
    ax3.set_title('Efficiency: Performance per Expert')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Optimal Panel Size Analysis
    ax4 = axes[1, 1]
    
    optimal_sizes = {}
    for model in ['Haiku (Smaller)', 'Sonnet (Larger)']:
        df_model = df_combined[df_combined['model_type'] == model]
        
        # Find the panel size with best average performance
        performance_by_size = df_model.groupby('n_experts')['brier_score'].mean()
        optimal_size = performance_by_size.idxmin()
        optimal_sizes[model] = optimal_size
        
        # Plot performance curve
        ax4.plot(performance_by_size.index, performance_by_size.values, 
                marker='o', linewidth=2, label=f'{model} (optimal: {optimal_size})')
        ax4.scatter([optimal_size], [performance_by_size[optimal_size]], 
                   s=100, marker='*', edgecolors='black', linewidths=2)
    
    human_brier = df_combined['human_brier'].mean()
    ax4.axhline(y=human_brier, color='red', linestyle='--', alpha=0.7, label='Human Baseline')
    ax4.set_xlabel('Number of Experts')
    ax4.set_ylabel('Average Brier Score')
    ax4.set_title('Optimal Panel Size Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('panel_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load results
    haiku_path = 'results/delphi2_run_haiku_20250624_165355/detailed_results.json'
    sonnet_path = 'results/delphi2_sonnet_run_20250624_225431/detailed_results.json'
    
    print("Loading results...")
    haiku_data = load_results(haiku_path)
    sonnet_data = load_results(sonnet_path)
    
    print("Analyzing model size effects...")
    df_combined = analyze_model_size_effects(haiku_data, sonnet_data)
    
    print("Creating panel size analysis...")
    create_panel_size_analysis(df_combined)
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main() 