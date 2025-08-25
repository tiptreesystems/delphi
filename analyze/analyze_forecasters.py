from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import ForecastDataLoader
from analyze.user_profiler import UserProfiler
import json


class ForecasterAnalyzer:
    def __init__(self, loader: ForecastDataLoader):
        self.loader = loader
        self.profiler = UserProfiler(loader)

    def create_forecaster_dataframe(self) -> pd.DataFrame:
        """Create a comprehensive dataframe of all forecasters and their predictions"""
        rows = []

        for question_id, forecasts in self.loader.super_forecasts.items():
            question = self.loader.get_question(question_id)
            if not question:
                continue

            resolution = self.loader.get_resolution(question_id)
            all_predictions = [f.forecast for f in forecasts]
            mean_prediction = np.mean(all_predictions)

            for forecast in forecasts:
                row = {
                    'user_id': forecast.user_id,
                    'question_id': question_id,
                    'question': question.question[:100] + '...',
                    'source': question.source,
                    'prediction': forecast.forecast,
                    'community_mean': mean_prediction,
                    'deviation_from_mean': forecast.forecast - mean_prediction,
                    'reasoning_length': len(forecast.reasoning),
                    'used_searches': bool(forecast.searches),
                    'num_searches': len(forecast.searches) if forecast.searches else 0,
                    'consulted_urls': bool(forecast.consulted_urls),
                    'num_urls': len(forecast.consulted_urls) if forecast.consulted_urls else 0
                }

                if resolution and resolution.resolved:
                    actual = 1.0 if resolution.resolved_to else 0.0
                    row['actual'] = actual
                    row['error'] = abs(forecast.forecast - actual)
                    row['squared_error'] = (forecast.forecast - actual) ** 2
                    row['correct_direction'] = (forecast.forecast > 0.5) == (actual > 0.5)

                rows.append(row)

        return pd.DataFrame(rows)

    def analyze_top_performers(self, df: pd.DataFrame, min_predictions: int = 10) -> pd.DataFrame:
        """Identify and analyze top performing forecasters"""
        resolved_df = df[df['actual'].notna()]

        user_stats = resolved_df.groupby('user_id').agg({
            'error': ['mean', 'std', 'count'],
            'squared_error': 'mean',
            'correct_direction': 'mean',
            'prediction': 'count'
        }).round(3)

        user_stats.columns = ['avg_error', 'std_error', 'resolved_count', 'brier_score', 'accuracy', 'total_predictions']
        user_stats = user_stats[user_stats['resolved_count'] >= min_predictions]
        user_stats = user_stats.sort_values('avg_error')

        return user_stats

    def plot_user_calibration(self, user_id: str, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot calibration curve for a specific user"""
        user_df = df[(df['user_id'] == user_id) & (df['actual'].notna())]

        if len(user_df) < 10:
            print(f"Not enough resolved predictions for user {user_id}")
            return

        bins = np.linspace(0, 1, 11)
        calibration_data = []

        for i in range(len(bins) - 1):
            mask = (user_df['prediction'] >= bins[i]) & (user_df['prediction'] < bins[i+1])
            bin_data = user_df[mask]

            if len(bin_data) > 0:
                calibration_data.append({
                    'predicted_prob': bin_data['prediction'].mean(),
                    'actual_freq': bin_data['actual'].mean(),
                    'count': len(bin_data)
                })

        if not calibration_data:
            return

        cal_df = pd.DataFrame(calibration_data)

        plt.figure(figsize=(8, 8))
        plt.scatter(cal_df['predicted_prob'], cal_df['actual_freq'],
                   s=cal_df['count']*50, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Frequency')
        plt.title(f'Calibration Plot for User {user_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def analyze_expertise_domains(self, df: pd.DataFrame) -> Dict:
        """Analyze which users are best at which types of questions"""
        resolved_df = df[df['actual'].notna()]

        expertise_matrix = defaultdict(dict)

        for source in resolved_df['source'].unique():
            source_df = resolved_df[resolved_df['source'] == source]

            user_performance = source_df.groupby('user_id').agg({
                'error': ['mean', 'count']
            })
            user_performance.columns = ['avg_error', 'count']

            # Only consider users with at least 3 predictions in this domain
            qualified_users = user_performance[user_performance['count'] >= 3]

            if len(qualified_users) > 0:
                top_performers = qualified_users.nsmallest(5, 'avg_error')
                expertise_matrix[source] = top_performers.to_dict('index')

        return dict(expertise_matrix)

    def generate_comparative_report(self, output_file: str = "forecaster_analysis_report.json"):
        """Generate a comprehensive analysis report"""
        df = self.create_forecaster_dataframe()

        report = {
            'overview': {
                'total_users': df['user_id'].nunique(),
                'total_predictions': len(df),
                'total_questions': df['question_id'].nunique(),
                'resolved_questions': df[df['actual'].notna()]['question_id'].nunique()
            },
            'top_performers': self.analyze_top_performers(df).head(20).to_dict('index'),
            'expertise_domains': self.analyze_expertise_domains(df),
            'research_habits': {
                'search_usage_rate': df['used_searches'].mean(),
                'url_consultation_rate': df['consulted_urls'].mean(),
                'avg_searches_when_used': df[df['used_searches']]['num_searches'].mean(),
                'avg_urls_when_used': df[df['consulted_urls']]['num_urls'].mean()
            }
        }

        # Add distribution statistics
        if 'error' in df.columns:
            resolved_df = df[df['actual'].notna()]
            report['performance_distribution'] = {
                'error_percentiles': {
                    '10th': resolved_df['error'].quantile(0.1),
                    '25th': resolved_df['error'].quantile(0.25),
                    '50th': resolved_df['error'].quantile(0.5),
                    '75th': resolved_df['error'].quantile(0.75),
                    '90th': resolved_df['error'].quantile(0.9)
                },
                'avg_brier_score': resolved_df['squared_error'].mean(),
                'overall_accuracy': resolved_df['correct_direction'].mean()
            }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Analysis report saved to {output_file}")
        return report

    def plot_performance_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot distribution of forecaster performance"""
        user_performance = df[df['actual'].notna()].groupby('user_id')['error'].mean()

        plt.figure(figsize=(10, 6))
        plt.hist(user_performance, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(user_performance.mean(), color='red', linestyle='--',
                   label=f'Mean: {user_performance.mean():.3f}')
        plt.xlabel('Average Prediction Error')
        plt.ylabel('Number of Users')
        plt.title('Distribution of Forecaster Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    # Fix path for data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    loader = ForecastDataLoader(data_dir=data_dir)
    analyzer = ForecasterAnalyzer(loader)

    # Generate comprehensive report
    report = analyzer.generate_comparative_report()

    # Create dataframe for further analysis
    df = analyzer.create_forecaster_dataframe()

    # Show top performers
    print("\nTop 10 Performers (by average error):")
    top_performers = analyzer.analyze_top_performers(df)
    print(top_performers.head(10))

    # Plot performance distribution
    if len(df[df['actual'].notna()]) > 0:
        analyzer.plot_performance_distribution(df)

    # Example: Plot calibration for top performer
    if len(top_performers) > 0:
        top_user = top_performers.index[0]
        analyzer.plot_user_calibration(top_user, df)