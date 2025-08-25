#!/usr/bin/env python3
"""
Analyze the evolution of learned prompts over sequential predictions.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import pandas as pd


def load_results(file_path: str) -> Dict:
    """Load results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_prompt_evolution(results: Dict) -> pd.DataFrame:
    """Analyze how the learned prompt evolves over predictions."""
    
    data = []
    prompt_evolution = results.get('prompt_evolution', [])
    
    for i, result in enumerate(results['results']):
        # Get the prompt used at this step
        prompt_at_step = prompt_evolution[i] if i < len(prompt_evolution) else ""
        
        data.append({
            'step': i + 1,
            'question': result['question_text'][:50] + "...",
            'topic': result['topic'],
            'predicted': result['predicted_prob'],
            'actual': result['actual_outcome'],
            'error': result['error'],
            'prompt_length': len(result.get('learned_prompt_used', '')),
            'prompt_sections': count_sections(result.get('learned_prompt_used', ''))
        })
    
    return pd.DataFrame(data)


def count_sections(prompt: str) -> int:
    """Count the number of sections in a markdown prompt."""
    return prompt.count('##')


def extract_strategies(prompt: str) -> Dict[str, int]:
    """Extract and count different types of strategies mentioned."""
    strategies = {
        'mistakes_to_avoid': prompt.count('Mistake') + prompt.count('Avoid'),
        'successful_strategies': prompt.count('Success') + prompt.count('Work'),
        'domain_insights': prompt.count('Domain') + prompt.count('Topic'),
        'calibration': prompt.count('Calibrat') + prompt.count('Adjust'),
        'information': prompt.count('Information') + prompt.count('Source')
    }
    return strategies


def plot_evolution(df: pd.DataFrame, results: Dict, output_dir: Path):
    """Create visualizations of prompt evolution."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Error over time
    ax = axes[0, 0]
    ax.plot(df['step'], df['error'].abs(), marker='o', color='red', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Prediction Error Over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Prompt length evolution
    ax = axes[0, 1]
    ax.plot(df['step'], df['prompt_length'], marker='s', color='blue', alpha=0.7)
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Prompt Length (chars)')
    ax.set_title('Learned Prompt Growth')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Number of sections
    ax = axes[0, 2]
    ax.plot(df['step'], df['prompt_sections'], marker='^', color='green', alpha=0.7)
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Number of Sections')
    ax.set_title('Prompt Complexity (Sections)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative average error
    ax = axes[1, 0]
    cumulative_avg = df['error'].abs().expanding().mean()
    ax.plot(df['step'], cumulative_avg, marker='o', color='purple', alpha=0.7)
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Cumulative Avg Error')
    ax.set_title('Learning Progress')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Error by topic
    ax = axes[1, 1]
    topic_errors = df.groupby('topic')['error'].apply(lambda x: x.abs().mean())
    if not topic_errors.empty:
        topic_errors.plot(kind='bar', ax=ax, color='orange', alpha=0.7)
        ax.set_xlabel('Topic')
        ax.set_ylabel('Average Absolute Error')
        ax.set_title('Error by Topic')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 6: Strategy evolution
    ax = axes[1, 2]
    if results.get('prompt_evolution'):
        strategy_counts = []
        for prompt in results['prompt_evolution']:
            strategies = extract_strategies(prompt)
            strategy_counts.append(sum(strategies.values()))
        
        ax.plot(range(1, len(strategy_counts) + 1), strategy_counts, 
                marker='d', color='teal', alpha=0.7)
        ax.set_xlabel('Prediction Step')
        ax.set_ylabel('Total Strategy Mentions')
        ax.set_title('Strategy Accumulation')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'prompt_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved evolution plot to: {output_path}")
    plt.close()


def print_prompt_diff(results: Dict):
    """Print how the prompt changed between steps."""
    
    evolution = results.get('prompt_evolution', [])
    
    if len(evolution) < 2:
        print("Not enough evolution steps to show differences")
        return
    
    print("\nPROMPT EVOLUTION HIGHLIGHTS")
    print("="*60)
    
    for i in range(1, len(evolution)):
        prev_prompt = evolution[i-1] if i > 0 else ""
        curr_prompt = evolution[i]
        
        # Find new content (simple approach)
        prev_lines = set(prev_prompt.split('\n'))
        curr_lines = set(curr_prompt.split('\n'))
        new_lines = curr_lines - prev_lines
        
        if new_lines:
            print(f"\nStep {i} → {i+1} additions:")
            for line in list(new_lines)[:3]:  # Show first 3 new lines
                if line.strip():
                    print(f"  + {line[:80]}...")


def print_final_summary(df: pd.DataFrame, results: Dict):
    """Print comprehensive summary of learning."""
    
    print("\n" + "="*60)
    print("PROMPT LEARNING SUMMARY")
    print("="*60)
    
    # Error statistics
    print("\nPREDICTION PERFORMANCE:")
    print(f"  Initial error: {abs(df.iloc[0]['error']):.3f}")
    print(f"  Final error: {abs(df.iloc[-1]['error']):.3f}")
    print(f"  Average error: {df['error'].abs().mean():.3f}")
    print(f"  Best prediction: {df['error'].abs().min():.3f}")
    print(f"  Worst prediction: {df['error'].abs().max():.3f}")
    
    # Prompt evolution
    if results.get('prompt_evolution'):
        evolution = results['prompt_evolution']
        print(f"\nPROMPT EVOLUTION:")
        print(f"  Evolution steps: {len(evolution)}")
        print(f"  Initial length: {len(evolution[0]) if evolution else 0} chars")
        print(f"  Final length: {len(evolution[-1]) if evolution else 0} chars")
        print(f"  Growth rate: {(len(evolution[-1]) - len(evolution[0])) / len(evolution[0]) * 100:.1f}%" if evolution and len(evolution[0]) > 0 else "N/A")
        
        # Extract final strategies
        if evolution:
            final_strategies = extract_strategies(evolution[-1])
            print(f"\nFINAL STRATEGY COUNTS:")
            for strategy, count in final_strategies.items():
                print(f"  {strategy.replace('_', ' ').title()}: {count}")
    
    # Learning effectiveness
    if len(df) >= 4:
        first_half_error = df.iloc[:len(df)//2]['error'].abs().mean()
        second_half_error = df.iloc[len(df)//2:]['error'].abs().mean()
        improvement = (first_half_error - second_half_error) / first_half_error * 100
        
        print(f"\nLEARNING EFFECTIVENESS:")
        print(f"  First half avg error: {first_half_error:.3f}")
        print(f"  Second half avg error: {second_half_error:.3f}")
        print(f"  Improvement: {improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Analyze prompt learning evolution')
    parser.add_argument('results_file', type=str,
                       help='Path to the results JSON file')
    parser.add_argument('--output-dir', type=str, default='results/analysis',
                       help='Directory to save analysis outputs')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze evolution
    df = analyze_prompt_evolution(results)
    
    # Print summaries
    print_final_summary(df, results)
    print_prompt_diff(results)
    
    # Generate plots
    if not args.no_plots and not df.empty:
        plot_evolution(df, results, output_dir)
    
    # Save analyzed data
    csv_path = output_dir / 'prompt_evolution_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved evolution data to: {csv_path}")
    
    # Save final prompt separately
    if results.get('final_learned_prompt'):
        prompt_path = output_dir / 'final_learned_prompt.md'
        with open(prompt_path, 'w') as f:
            f.write(results['final_learned_prompt'])
        print(f"Saved final prompt to: {prompt_path}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()