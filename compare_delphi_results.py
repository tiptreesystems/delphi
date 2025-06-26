import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(data):
    results = []
    metadata = data['metadata']
    model_name = metadata['model']
    
    for question_id, question_data in data['questions'].items():
        for n_experts, expert_data in question_data.items():
            if n_experts == 'metadata':
                continue
            
            result = {
                'model': model_name,
                'question_id': question_id,
                'question_text': expert_data['question_text'][:50] + '...',
                'n_experts': int(n_experts),
                'outcome': expert_data['outcome'],
                'aggregate_forecast': expert_data['aggregate'],
                'brier_score': expert_data['brier'],
                'mae': expert_data['mae'],
                'human_mean': expert_data['human_mean'],
                'human_brier': expert_data['human_brier'],
                'human_mae': expert_data['human_mae'],
                'individual_forecasts': expert_data['individual_forecasts']
            }
            results.append(result)
    
    return results

def create_comparison_plots(haiku_data, sonnet_data):
    haiku_results = extract_metrics(haiku_data)
    sonnet_results = extract_metrics(sonnet_data)
    
    df_haiku = pd.DataFrame(haiku_results)
    df_sonnet = pd.DataFrame(sonnet_results)
    
    df_haiku['model_type'] = 'Haiku'
    df_sonnet['model_type'] = 'Sonnet'
    
    df_combined = pd.concat([df_haiku, df_sonnet])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Delphi Method Performance: Haiku vs Sonnet Models', fontsize=16)
    
    # 1. Brier Score by Number of Experts
    ax1 = axes[0, 0]
    df_pivot = df_combined.pivot_table(values='brier_score', index='n_experts', 
                                      columns='model_type', aggfunc='mean')
    df_pivot.plot(kind='bar', ax=ax1)
    
    # Add human baseline
    human_brier = df_combined['human_brier'].mean()
    ax1.axhline(y=human_brier, color='red', linestyle='--', label=f'Human Mean: {human_brier:.4f}')
    ax1.set_xlabel('Number of Experts')
    ax1.set_ylabel('Brier Score (lower is better)')
    ax1.set_title('Average Brier Score by Number of Experts')
    ax1.legend()
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    
    # 2. MAE by Number of Experts
    ax2 = axes[0, 1]
    df_pivot_mae = df_combined.pivot_table(values='mae', index='n_experts', 
                                          columns='model_type', aggfunc='mean')
    df_pivot_mae.plot(kind='bar', ax=ax2)
    
    # Add human baseline
    human_mae = df_combined['human_mae'].mean()
    ax2.axhline(y=human_mae, color='red', linestyle='--', label=f'Human Mean: {human_mae:.4f}')
    ax2.set_xlabel('Number of Experts')
    ax2.set_ylabel('MAE (lower is better)')
    ax2.set_title('Average MAE by Number of Experts')
    ax2.legend()
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    
    # 3. Distribution of Brier Scores
    ax3 = axes[0, 2]
    df_combined.boxplot(column='brier_score', by='model_type', ax=ax3)
    ax3.axhline(y=human_brier, color='red', linestyle='--', label='Human Mean')
    ax3.set_xlabel('Model Type')
    ax3.set_ylabel('Brier Score')
    ax3.set_title('Distribution of Brier Scores')
    ax3.legend()
    
    # 4. Forecast vs Outcome Scatter
    ax4 = axes[1, 0]
    for model in ['Haiku', 'Sonnet']:
        df_model = df_combined[df_combined['model_type'] == model]
        ax4.scatter(df_model['aggregate_forecast'], df_model['outcome'], 
                   label=model, alpha=0.6, s=50)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    ax4.set_xlabel('Aggregate Forecast')
    ax4.set_ylabel('Actual Outcome')
    ax4.set_title('Calibration: Forecast vs Outcome')
    ax4.legend()
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    
    # 5. Performance vs Human Baseline
    ax5 = axes[1, 1]
    models = ['Haiku', 'Sonnet']
    metrics = ['Brier Score', 'MAE']
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model in enumerate(models):
        df_model = df_combined[df_combined['model_type'] == model]
        model_values = [df_model['brier_score'].mean(), df_model['mae'].mean()]
        ax5.bar(x + i*width, model_values, width, label=model)
    
    human_values = [human_brier, human_mae]
    ax5.bar(x + 2*width, human_values, width, label='Human', color='red', alpha=0.7)
    
    ax5.set_xlabel('Metric')
    ax5.set_ylabel('Value (lower is better)')
    ax5.set_title('Model Performance vs Human Baseline')
    ax5.set_xticks(x + width)
    ax5.set_xticklabels(metrics)
    ax5.legend()
    
    # 6. Convergence Analysis
    ax6 = axes[1, 2]
    for model in ['Haiku', 'Sonnet']:
        df_model = df_combined[df_combined['model_type'] == model]
        
        # Calculate standard deviation of individual forecasts for each prediction
        std_devs = []
        n_experts_list = sorted(df_model['n_experts'].unique())
        
        for n in n_experts_list:
            df_n = df_model[df_model['n_experts'] == n]
            all_stds = []
            for _, row in df_n.iterrows():
                if row['individual_forecasts']:
                    std = np.std(row['individual_forecasts'])
                    all_stds.append(std)
            if all_stds:
                std_devs.append(np.mean(all_stds))
            else:
                std_devs.append(0)
        
        ax6.plot(n_experts_list, std_devs, marker='o', label=model)
    
    ax6.set_xlabel('Number of Experts')
    ax6.set_ylabel('Average Std Dev of Forecasts')
    ax6.set_title('Forecast Convergence by Number of Experts')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('delphi_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_combined

def create_detailed_analysis(df_combined):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detailed Delphi Analysis: Model Size Effects', fontsize=16)
    
    # 1. Relative Performance to Human
    ax1 = axes[0, 0]
    df_summary = df_combined.groupby('model_type').agg({
        'brier_score': 'mean',
        'mae': 'mean',
        'human_brier': 'mean',
        'human_mae': 'mean'
    })
    
    df_summary['brier_relative'] = df_summary['brier_score'] / df_summary['human_brier']
    df_summary['mae_relative'] = df_summary['mae'] / df_summary['human_mae']
    
    metrics = ['brier_relative', 'mae_relative']
    df_summary[metrics].plot(kind='bar', ax=ax1)
    ax1.axhline(y=1, color='red', linestyle='--', label='Human Performance')
    ax1.set_ylabel('Relative to Human (lower is better)')
    ax1.set_title('Performance Relative to Human Baseline')
    ax1.legend()
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # 2. Performance by Question
    ax2 = axes[0, 1]
    questions = df_combined['question_text'].unique()[:5]  # Top 5 questions
    
    for i, q in enumerate(questions):
        df_q = df_combined[df_combined['question_text'] == q]
        df_q_summary = df_q.groupby('model_type')['brier_score'].mean()
        ax2.bar([i-0.2, i+0.2], df_q_summary.values, width=0.4, 
               label=df_q_summary.index if i == 0 else "")
    
    ax2.set_xticks(range(len(questions)))
    ax2.set_xticklabels([q[:20] + '...' for q in questions], rotation=45, ha='right')
    ax2.set_ylabel('Brier Score')
    ax2.set_title('Performance by Question')
    if len(questions) > 0:
        ax2.legend()
    
    # 3. Expert Agreement Analysis
    ax3 = axes[1, 0]
    for model in ['Haiku', 'Sonnet']:
        df_model = df_combined[df_combined['model_type'] == model]
        
        agreement_scores = []
        n_experts_list = sorted(df_model['n_experts'].unique())
        
        for n in n_experts_list:
            df_n = df_model[df_model['n_experts'] == n]
            agreements = []
            for _, row in df_n.iterrows():
                if row['individual_forecasts'] and len(row['individual_forecasts']) > 1:
                    # Calculate agreement as 1 - coefficient of variation
                    mean_forecast = np.mean(row['individual_forecasts'])
                    if mean_forecast > 0:
                        cv = np.std(row['individual_forecasts']) / mean_forecast
                        agreement = 1 - min(cv, 1)
                        agreements.append(agreement)
            if agreements:
                agreement_scores.append(np.mean(agreements))
            else:
                agreement_scores.append(0)
        
        ax3.plot(n_experts_list, agreement_scores, marker='o', label=model, linewidth=2)
    
    ax3.set_xlabel('Number of Experts')
    ax3.set_ylabel('Expert Agreement Score')
    ax3.set_title('Expert Agreement by Panel Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistical Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_data = []
    for model in ['Haiku', 'Sonnet']:
        df_model = df_combined[df_combined['model_type'] == model]
        summary_data.append([
            model,
            f"{df_model['brier_score'].mean():.4f}",
            f"{df_model['mae'].mean():.4f}",
            f"{len(df_model)}",
            f"{df_model['brier_score'].std():.4f}"
        ])
    
    # Add human baseline
    summary_data.append([
        'Human',
        f"{df_combined['human_brier'].mean():.4f}",
        f"{df_combined['human_mae'].mean():.4f}",
        'N/A',
        'N/A'
    ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Model', 'Avg Brier', 'Avg MAE', 'N Predictions', 'Brier Std'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Statistical Summary', pad=20)
    
    plt.tight_layout()
    plt.savefig('delphi_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load results
    haiku_path = 'results/delphi2_run_haiku_20250624_165355/detailed_results.json'
    sonnet_path = 'results/delphi2_sonnet_run_20250624_225431/detailed_results.json'
    
    print("Loading results...")
    haiku_data = load_results(haiku_path)
    sonnet_data = load_results(sonnet_path)
    
    print("Creating comparison plots...")
    df_combined = create_comparison_plots(haiku_data, sonnet_data)
    
    print("Creating detailed analysis...")
    create_detailed_analysis(df_combined)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for model in ['Haiku', 'Sonnet']:
        df_model = df_combined[df_combined['model_type'] == model]
        print(f"\n{model} Model:")
        print(f"  Average Brier Score: {df_model['brier_score'].mean():.4f}")
        print(f"  Average MAE: {df_model['mae'].mean():.4f}")
        print(f"  Brier Score Std Dev: {df_model['brier_score'].std():.4f}")
        print(f"  Best Brier Score: {df_model['brier_score'].min():.4f}")
        print(f"  Worst Brier Score: {df_model['brier_score'].max():.4f}")
    
    print(f"\nHuman Baseline:")
    print(f"  Average Brier Score: {df_combined['human_brier'].mean():.4f}")
    print(f"  Average MAE: {df_combined['human_mae'].mean():.4f}")
    
    # Calculate relative performance
    haiku_brier = df_combined[df_combined['model_type'] == 'Haiku']['brier_score'].mean()
    sonnet_brier = df_combined[df_combined['model_type'] == 'Sonnet']['brier_score'].mean()
    human_brier = df_combined['human_brier'].mean()
    
    print(f"\nRelative Performance (vs Human):")
    print(f"  Haiku: {haiku_brier/human_brier:.2f}x human Brier score")
    print(f"  Sonnet: {sonnet_brier/human_brier:.2f}x human Brier score")
    print(f"  Sonnet vs Haiku: {(haiku_brier-sonnet_brier)/sonnet_brier*100:.1f}% improvement")

if __name__ == "__main__":
    main() 