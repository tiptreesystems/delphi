from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import ForecastDataLoader, Forecast, Question
from models import LLMFactory, LLMProvider, LLMModel, BaseLLM
import json


class UserProfiler:
    def __init__(self, loader: ForecastDataLoader, llm: Optional[BaseLLM] = None):
        self.loader = loader
        print(os.getenv("ANTHROPIC_API_KEY"))
        self.llm = llm or LLMFactory.create_llm(LLMProvider.CLAUDE, LLMModel.CLAUDE_4_OPUS, api_key=os.getenv("ANTHROPIC_API_KEY"))
        
    def get_all_user_forecasts(self) -> Dict[str, List[Tuple[str, Question, Forecast, float, float]]]:
        """Get all forecasts organized by user_id with question info and comparative statistics"""
        user_forecasts = defaultdict(list)
        
        for question_id, forecasts in self.loader.forecasts.items():
            question = self.loader.get_question(question_id)
            if not question or not forecasts:
                continue
                
            # Calculate statistics for this question
            all_predictions = [f.forecast for f in forecasts]
            mean_prediction = np.mean(all_predictions)
            std_prediction = np.std(all_predictions)
            
            for forecast in forecasts:
                # Calculate percentile excluding the user's own prediction
                other_predictions = [f.forecast for f in forecasts if f.user_id != forecast.user_id]
                if other_predictions:
                    # Count how many predictions are strictly less than this user's prediction
                    num_below = sum(1 for p in other_predictions if p < forecast.forecast)
                    # Percentile = (number of values below) / (total number of other values) * 100
                    percentile = (num_below / len(other_predictions)) * 100
                else:
                    percentile = 50.0  # Default to median if no other predictions
                user_forecasts[forecast.user_id].append((
                    question_id,
                    question,
                    forecast,
                    mean_prediction,
                    percentile
                ))
                
        return dict(user_forecasts)
    
    def analyze_user_performance(self, user_id: str, user_data: List[Tuple[str, Question, Forecast, float, float]]) -> Dict:
        """Analyze a single user's forecasting patterns and performance"""
        resolved_predictions = []
        all_predictions = []
        topic_performance = defaultdict(list)
        
        for question_id, question, forecast, mean_pred, percentile in user_data:
            all_predictions.append({
                'question_id': question_id,
                'prediction': forecast.forecast,
                'mean_prediction': mean_pred,
                'percentile': percentile,
                'reasoning_length': len(forecast.reasoning),
                'used_searches': bool(forecast.searches),
                'consulted_urls': bool(forecast.consulted_urls)
            })
            
            # Check if resolved
            resolution = self.loader.get_resolution(question_id)
            if resolution and resolution.resolved:
                actual = 1.0 if resolution.resolved_to else 0.0
                error = abs(forecast.forecast - actual)
                resolved_predictions.append({
                    'question_id': question_id,
                    'prediction': forecast.forecast,
                    'actual': actual,
                    'error': error,
                    'brier_score': (forecast.forecast - actual) ** 2
                })
                
                # Categorize by topic
                topic = question.source
                topic_performance[topic].append(error)
        
        # Calculate aggregate statistics
        stats = {
            'total_predictions': len(all_predictions),
            'resolved_predictions': len(resolved_predictions),
            'avg_percentile': np.mean([p['percentile'] for p in all_predictions]),
            'avg_reasoning_length': np.mean([p['reasoning_length'] for p in all_predictions]),
            'search_usage_rate': np.mean([p['used_searches'] for p in all_predictions]),
            'url_consultation_rate': np.mean([p['consulted_urls'] for p in all_predictions])
        }
        
        if resolved_predictions:
            stats['avg_error'] = np.mean([p['error'] for p in resolved_predictions])
            stats['avg_brier_score'] = np.mean([p['brier_score'] for p in resolved_predictions])
            stats['calibration_score'] = self._calculate_calibration(resolved_predictions)
        
        stats['topic_performance'] = {topic: np.mean(errors) for topic, errors in topic_performance.items()}
        
        return stats
    
    def _calculate_calibration(self, resolved_predictions: List[Dict]) -> float:
        """Calculate calibration score (lower is better)"""
        bins = np.linspace(0, 1, 11)
        calibration_error = 0
        total_count = 0
        
        for i in range(len(bins) - 1):
            bin_mask = [(bins[i] <= p['prediction'] < bins[i+1]) for p in resolved_predictions]
            bin_predictions = [p for p, mask in zip(resolved_predictions, bin_mask) if mask]
            
            if bin_predictions:
                predicted_prob = np.mean([p['prediction'] for p in bin_predictions])
                actual_freq = np.mean([p['actual'] for p in bin_predictions])
                count = len(bin_predictions)
                calibration_error += count * abs(predicted_prob - actual_freq)
                total_count += count
        
        return calibration_error / total_count if total_count > 0 else 0
    
    def generate_expertise_profile(self, user_id: str, user_data: List[Tuple[str, Question, Forecast, float, float]], stats: Dict) -> str:
        """Generate an expertise profile using LLM based on user's forecasting history"""
        # Prepare context for LLM
        question_samples = []
        for i, (q_id, question, forecast, mean_pred, percentile) in enumerate(user_data[:10]):  # Sample first 10
            question_samples.append(f"""
Question {i+1}: {question.question}
User's Prediction: {forecast.forecast:.3f} (Community Average: {mean_pred:.3f}, Percentile: {percentile:.1f}%)
Reasoning: {forecast.reasoning[:200]}...
""")
        
        prompt = f"""Based on the following forecasting data, create a hypothetical expertise profile for this user:

OVERALL STATISTICS:
- Total Predictions: {stats['total_predictions']}
- Average Percentile (vs other forecasters): {stats['avg_percentile']:.1f}%
- Search Usage Rate: {stats['search_usage_rate']:.1%}
- URL Consultation Rate: {stats['url_consultation_rate']:.1%}
- Average Reasoning Length: {stats['avg_reasoning_length']:.0f} characters

PERFORMANCE METRICS (if available):
{f"- Average Error: {stats.get('avg_error', 'N/A'):.3f}" if 'avg_error' in stats else '- No resolved predictions yet'}
{f"- Calibration Score: {stats.get('calibration_score', 'N/A'):.3f}" if 'calibration_score' in stats else ''}

TOPIC PERFORMANCE:
{chr(10).join([f"- {topic}: {perf:.3f} avg error" for topic, perf in stats.get('topic_performance', {}).items()])}

SAMPLE PREDICTIONS:
{''.join(question_samples)}

Based on this data, provide a 1 paragraph expertise profile that includes:
1. The user's name (use a real name)
2. the user's professional background
3. The user's areas of expertise and knowledge domains
4. Their forecasting style (conservative/aggressive, research-heavy, intuitive, etc.)
5. Strengths and potential blind spots
"""

        response = self.llm.generate(prompt, max_tokens=500, temperature=0.4)
        return response.strip()
    
    def generate_all_profiles(self, output_file: str = "user_profiles.json", save_incrementally: bool = True):
        """Generate profiles for all users and save to file"""
        all_user_forecasts = self.get_all_user_forecasts()
        profiles = {}
        
        print(f"Found {len(all_user_forecasts)} unique users")
        
        # Load existing profiles if file exists and we're saving incrementally
        if save_incrementally and os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    profiles = json.load(f)
                print(f"Loaded {len(profiles)} existing profiles from {output_file}")
            except:
                print("Could not load existing profiles, starting fresh")
        
        for i, (user_id, user_data) in enumerate(all_user_forecasts.items()):
            # Skip if already processed
            if user_id in profiles and save_incrementally:
                print(f"Skipping user {i+1}/{len(all_user_forecasts)}: {user_id} (already processed)")
                continue
                
            print(f"Processing user {i+1}/{len(all_user_forecasts)}: {user_id}")
            
            try:
                # Analyze performance
                stats = self.analyze_user_performance(user_id, user_data)
                
                # Generate expertise profile
                profile = self.generate_expertise_profile(user_id, user_data, stats)
                
                profiles[user_id] = {
                    'user_id': user_id,
                    'statistics': stats,
                    'expertise_profile': profile,
                    'total_questions_answered': len(user_data)
                }
                
                # Save incrementally after each profile
                if save_incrementally:
                    with open(output_file, 'w') as f:
                        json.dump(profiles, f, indent=2)
                    print(f"  Saved {len(profiles)} profiles to {output_file}")
                    
            except Exception as e:
                print(f"  Error processing user {user_id}: {str(e)}")
                continue
        
        # Final save (in case save_incrementally is False)
        with open(output_file, 'w') as f:
            json.dump(profiles, f, indent=2)
        
        print(f"\nCompleted! Saved {len(profiles)} profiles to {output_file}")
        return profiles


if __name__ == "__main__":
    # Fix path for data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    loader = ForecastDataLoader(data_dir=data_dir)
    profiler = UserProfiler(loader)
    
    # Generate profiles for all users with incremental saving
    profiles = profiler.generate_all_profiles(save_incrementally=True)
    
    # Print a sample profile
    if profiles:
        sample_user = list(profiles.keys())[0]
        print(f"\nSample Profile for User {sample_user}:")
        print(f"Statistics: {profiles[sample_user]['statistics']}")
        print(f"\nExpertise Profile:\n{profiles[sample_user]['expertise_profile']}") 