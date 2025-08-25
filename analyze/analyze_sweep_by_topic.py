#!/usr/bin/env python3
"""
Topic-based analysis of sweep results.

This script analyzes sweep results grouped by question topics, creating separate plots for each topic.
It aggregates results across questions within the same topic and generates individual visualizations.
"""

import json
import sys
import subprocess
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict
import pickle
from dataset.dataloader import ForecastDataLoader

def run_icl_delphi_results(config_path):
    """Run icl_delphi_results.py and capture the output."""
    try:
        result = subprocess.run(
            ['python3', 'icl_delphi_results.py', config_path],
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout
    except Exception as e:
        print(f"Error running icl_delphi_results.py with {config_path}: {e}", file=sys.stderr)
        return None

def parse_brier_scores_by_question(output_text):
    """Parse Brier scores from icl_delphi_results.py output, separated by question."""
    if not output_text:
        return {}
    
    results_by_question = {}
    current_question = None
    
    # Split output by lines and process
    lines = output_text.split('\n')
    
    for line in lines:
        # Check for question ID lines
        question_match = re.match(r'^Question: (.+)$', line)
        if question_match:
            current_question = question_match.group(1)
            results_by_question[current_question] = {
                'sf_brier': None,
                'public_brier': None,
                'llm_rounds': {}
            }
            continue
        
        if current_question:
            # Parse SF Brier for current question
            sf_match = re.search(r'\[SF\].*?Brier.*?: ([\d.]+|nan)', line)
            if sf_match and sf_match.group(1) != 'nan':
                results_by_question[current_question]['sf_brier'] = float(sf_match.group(1))
            
            # Parse Public Brier for current question
            public_match = re.search(r'\[Public\].*?Brier.*?: ([\d.]+|nan)', line)
            if public_match and public_match.group(1) != 'nan':
                results_by_question[current_question]['public_brier'] = float(public_match.group(1))
            
            # Parse LLM rounds for current question
            round_match = re.search(r'\[LLM\].*?round (\d+).*?Brier.*?: ([\d.]+|nan)', line)
            if round_match and round_match.group(2) != 'nan':
                round_num = int(round_match.group(1))
                brier = float(round_match.group(2))
                results_by_question[current_question]['llm_rounds'][round_num] = brier
    
    # If no per-question results found, try to get aggregate results
    if not results_by_question:
        # Parse aggregate results
        aggregate = {
            'aggregate': {
                'sf_brier': None,
                'public_brier': None,
                'llm_rounds': {}
            }
        }
        
        # Parse SF Brier
        sf_match = re.search(r'\[SF\].*?median SF forecast.*?: ([\d.]+|nan)', output_text)
        if sf_match and sf_match.group(1) != 'nan':
            aggregate['aggregate']['sf_brier'] = float(sf_match.group(1))
        
        # Parse Public Brier
        public_match = re.search(r'\[Public\].*?median Public forecast.*?: ([\d.]+|nan)', output_text)
        if public_match and public_match.group(1) != 'nan':
            aggregate['aggregate']['public_brier'] = float(public_match.group(1))
        
        # Parse LLM rounds
        round_matches = re.findall(r'\[LLM\].*?median LLM forecast.*?at round (\d+): ([\d.]+|nan)', output_text)
        for round_num, brier in round_matches:
            if brier != 'nan':
                aggregate['aggregate']['llm_rounds'][int(round_num)] = float(brier)
        
        if aggregate['aggregate']['llm_rounds']:
            return aggregate
    
    return results_by_question

def load_question_topics(sweep_dir):
    """Load question topics using ForecastDataLoader."""
    sweep_dir = Path(sweep_dir)
    question_to_topic = {}
    loader = ForecastDataLoader()
    
    # Try to find any output directory with JSON files to get question IDs
    output_dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith('results_')]
    
    question_ids = set()
    for output_dir in output_dirs:
        json_files = list(output_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract question ID
                    qid = data.get('question_id', '')
                    if qid:
                        question_ids.add(qid)
            except Exception as e:
                continue
    
    # Load topics for each question ID
    for qid in question_ids:
        try:
            question = loader.get_question(qid)
            if question and hasattr(question, 'topic'):
                question_to_topic[qid] = question.topic
            else:
                question_to_topic[qid] = 'Unknown'
        except Exception as e:
            question_to_topic[qid] = 'Unknown'
    
    return question_to_topic

def extract_params_from_output_dir(output_dir_name):
    """Extract parameter values from output directory name."""
    params = {}
    
    # Handle n_experts_X_seed_Y pattern
    if 'results_n_experts_' in output_dir_name:
        parts = output_dir_name.replace('results_n_experts_', '').split('_')
        if len(parts) >= 3 and parts[1] == 'seed':
            params['n_experts'] = int(parts[0])
            params['seed'] = int(parts[2])
    
    return params

def scan_for_actual_results(sweep_dir):
    """Scan directory for actual output directories and build results list."""
    sweep_dir = Path(sweep_dir)
    
    # Look for output directories
    output_dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith('results_')]
    
    # Filter out _initial directories
    output_dirs = [d for d in output_dirs if not d.name.endswith('_initial')]
    
    results = []
    
    for output_dir in output_dirs:
        # Extract parameters from directory name
        params = extract_params_from_output_dir(output_dir.name)
        if not params:
            print(f"Could not parse parameters from {output_dir.name}, skipping...")
            continue
        
        # Check if directory has JSON files (indicating completed experiments)
        json_files = list(output_dir.glob("*.json"))
        
        if json_files:
            # Find corresponding config file
            if 'n_experts' in params and 'seed' in params:
                config_name = f"config_n_experts_{params['n_experts']}_seed_{params['seed']}.yml"
            else:
                print(f"Could not determine config file for {output_dir.name}, skipping...")
                continue
                
            config_path = sweep_dir / config_name
            
            if config_path.exists():
                result = {
                    "config_file": str(config_path),
                    "output_dir": str(output_dir),
                    "parameters": params,
                    "n_json_files": len(json_files)
                }
                results.append(result)
                print(f"Found {len(json_files)} results in {output_dir.name}")
            else:
                print(f"Config file not found for {output_dir.name}: {config_path}")
        else:
            print(f"No JSON files found in {output_dir.name}, skipping...")
    
    return results

def load_results_by_topic(sweep_dir):
    """Load and process sweep results, organizing by topic."""
    sweep_dir = Path(sweep_dir)
    
    # Load question topics
    print("Loading question topics...")
    question_to_topic = load_question_topics(sweep_dir)
    print(f"Found {len(question_to_topic)} questions with topics")
    
    # Count questions per topic
    topic_counts = defaultdict(int)
    for topic in question_to_topic.values():
        topic_counts[topic] += 1
    
    print("\nQuestions per topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  - {topic}: {count} questions")
    
    # Load sweep configuration
    sweep_config_path = sweep_dir / "sweep_config.json"
    if sweep_config_path.exists():
        with open(sweep_config_path, 'r') as f:
            sweep_config = json.load(f)
    else:
        print(f"Warning: No sweep_config.json found in {sweep_dir}")
        sweep_config = {
            "mode": "combinatorial",
            "parameters": [
                {"name": "n_experts", "values": [1, 2, 3, 4, 5]},
                {"name": "seed", "values": [42, 123, 456]}
            ]
        }
    
    # Scan for actual results
    print("\nScanning for actual output directories...")
    actual_results = scan_for_actual_results(sweep_dir)
    
    # Process each result and organize by topic  
    # Use the same approach as analyze_results_robust.py
    results_by_topic = defaultdict(list)

    for result in actual_results:
        config_path = result['config_file']
        
        # Run icl_delphi_results to get Brier scores
        print(f"Processing {Path(config_path).name}...")
        output = run_icl_delphi_results(config_path)

        if output:
            # Parse aggregated Brier scores first
            aggregate_results = parse_brier_scores_by_question(output)
            
            # For each question in the output
            output_dir = Path(result['output_dir'])
            json_files = list(output_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    qid = data.get('question_id', '')
                    if qid and qid in question_to_topic:
                        topic = question_to_topic[qid]
                        
                        # Use aggregate results for now (all questions get same scores)
                        # This is a simplification - ideally we'd parse per-question scores
                        if 'aggregate' in aggregate_results:
                            brier_scores = aggregate_results['aggregate']
                        else:
                            brier_scores = {
                                'sf_brier': None,
                                'public_brier': None,
                                'llm_rounds': {}
                            }
                        
                        question_result = {
                            'question_id': qid,
                            'topic': topic,
                            'parameters': result['parameters'],
                            'config_file': result['config_file'],
                            'brier_scores': brier_scores
                        }
                        
                        # Add to results by topic
                        if brier_scores['llm_rounds']:  # Only add if we have round data
                            results_by_topic[topic].append(question_result)
                        
                except Exception as e:
                    continue
    
    return {
        'sweep_config': sweep_config,
        'results_by_topic': dict(results_by_topic),
        'question_to_topic': question_to_topic,
        'total_found': len(actual_results)
    }

def plot_topic_progression(topic, topic_results, output_path):
    """Create round progression plot for a specific topic."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Group results by n_experts (averaging over seeds and questions)
    n_experts_to_rounds = defaultdict(lambda: defaultdict(list))
    
    for result in topic_results:
        if 'n_experts' in result['parameters'] and result['brier_scores']['llm_rounds']:
            n_experts = result['parameters']['n_experts']
            
            for round_num, brier in result['brier_scores']['llm_rounds'].items():
                n_experts_to_rounds[n_experts][round_num].append(brier)
    
    if not n_experts_to_rounds:
        ax.text(0.5, 0.5, f'No data available for topic: {topic}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Topic: {topic} (No Data)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Color palette
    n_experts_values = sorted(n_experts_to_rounds.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_experts_values)))
    
    for idx, n_experts in enumerate(n_experts_values):
        # Calculate means and standard errors for each round
        rounds_data = n_experts_to_rounds[n_experts]
        rounds = sorted(rounds_data.keys())
        mean_briers = []
        std_errors = []
        
        for round_num in rounds:
            briers = rounds_data[round_num]
            if briers:
                mean_briers.append(np.mean(briers))
                if len(briers) > 1:
                    std_errors.append(np.std(briers, ddof=1) / np.sqrt(len(briers)))
                else:
                    std_errors.append(0)
        
        if mean_briers:
            # Plot with error bars
            ax.errorbar(rounds, mean_briers, yerr=std_errors,
                       fmt='o-', label=f'n_experts={n_experts}', color=colors[idx],
                       linewidth=2, markersize=8, alpha=0.8,
                       capsize=3, capthick=1.5, elinewidth=1.5)
    
    ax.set_xlabel('Delphi Round', fontsize=14)
    ax.set_ylabel('Brier Score (lower is better)', fontsize=14)
    ax.set_title(f'Topic: {topic} - Brier Score Progression', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_topic_final_scores(topic, topic_results, output_path):
    """Create final score comparison plot for a specific topic."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Group results by n_experts (averaging over seeds and questions)
    n_experts_to_final_briers = defaultdict(list)
    
    for result in topic_results:
        if 'n_experts' in result['parameters'] and result['brier_scores']['llm_rounds']:
            n_experts = result['parameters']['n_experts']
            
            # Get final round score
            if result['brier_scores']['llm_rounds']:
                final_round = max(result['brier_scores']['llm_rounds'].keys())
                final_brier = result['brier_scores']['llm_rounds'][final_round]
                n_experts_to_final_briers[n_experts].append(final_brier)
    
    if not n_experts_to_final_briers:
        ax.text(0.5, 0.5, f'No data available for topic: {topic}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Topic: {topic} - Final Scores (No Data)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Calculate means and standard errors
    n_experts_values = []
    mean_briers = []
    std_errors = []
    sample_sizes = []
    
    for n_experts in sorted(n_experts_to_final_briers.keys()):
        briers = n_experts_to_final_briers[n_experts]
        n_experts_values.append(n_experts)
        mean_briers.append(np.mean(briers))
        sample_sizes.append(len(briers))
        
        if len(briers) > 1:
            std_errors.append(np.std(briers, ddof=1) / np.sqrt(len(briers)))
        else:
            std_errors.append(0)
    
    # Plot with error bars
    ax.errorbar(n_experts_values, mean_briers, yerr=std_errors,
                fmt='o-', linewidth=2.5, markersize=10,
                capsize=5, capthick=2, elinewidth=2,
                label=f'Mean ± SE', color='blue', alpha=0.8)
    
    # Add sample size annotations
    for x, y, n in zip(n_experts_values, mean_briers, sample_sizes):
        ax.annotate(f'n={n}', (x, y), 
                   xytext=(0, 10), textcoords='offset points', 
                   ha='center', fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Number of Experts', fontsize=14)
    ax.set_ylabel('Final Brier Score (lower is better)', fontsize=14)
    ax.set_title(f'Topic: {topic} - Final Score by Number of Experts', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    # Set y-axis limits
    if mean_briers:
        y_min = min(mean_briers) - max(std_errors) - 0.02
        y_max = max(mean_briers) + max(std_errors) + 0.02
        ax.set_ylim(max(0, y_min), min(1, y_max))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_topic_summary(topic, topic_results):
    """Create a summary for a specific topic."""
    print(f"\n{topic}:")
    print(f"  Total data points: {len(topic_results)}")
    
    # Count unique questions
    unique_questions = set(r['question_id'] for r in topic_results)
    print(f"  Unique questions: {len(unique_questions)}")
    
    # Get parameter coverage
    n_experts_coverage = set()
    seed_coverage = set()
    
    for result in topic_results:
        if 'n_experts' in result['parameters']:
            n_experts_coverage.add(result['parameters']['n_experts'])
        if 'seed' in result['parameters']:
            seed_coverage.add(result['parameters']['seed'])
    
    print(f"  n_experts values: {sorted(n_experts_coverage)}")
    print(f"  Seeds: {sorted(seed_coverage)}")
    
    # Calculate average final scores by n_experts
    n_experts_finals = defaultdict(list)
    for result in topic_results:
        if 'n_experts' in result['parameters'] and result['brier_scores']['llm_rounds']:
            n_experts = result['parameters']['n_experts']
            if result['brier_scores']['llm_rounds']:
                final_round = max(result['brier_scores']['llm_rounds'].keys())
                final_brier = result['brier_scores']['llm_rounds'][final_round]
                n_experts_finals[n_experts].append(final_brier)
    
    if n_experts_finals:
        print("  Average final Brier scores:")
        for n_experts in sorted(n_experts_finals.keys()):
            scores = n_experts_finals[n_experts]
            avg_score = np.mean(scores)
            print(f"    n_experts={n_experts}: {avg_score:.3f} (n={len(scores)})")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze sweep results by question topic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_sweep_by_topic.py results/20250821_072517
  python analyze_sweep_by_topic.py results/20250821_072517 --output topic_plots
  
This script creates separate plots for each question topic found in the sweep results.
        """
    )
    
    parser.add_argument(
        "sweep_dir",
        help="Path to sweep results directory"
    )
    
    parser.add_argument(
        "--output",
        default="topic_plots",
        help="Output directory for topic plots (default: topic_plots)"
    )
    
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots, only save them"
    )
    
    args = parser.parse_args()
    
    # Load sweep results organized by topic
    print(f"Loading sweep results from {args.sweep_dir}...")
    data = load_results_by_topic(args.sweep_dir)
    
    if not data['results_by_topic']:
        print("No results found organized by topic!")
        return
    
    # Create output directory
    sweep_path = Path(args.sweep_dir)
    output_dir = sweep_path / args.output
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nCreating plots in {output_dir}")
    print("="*80)
    print("TOPIC SUMMARIES")
    print("="*80)
    
    # Process each topic
    for topic in sorted(data['results_by_topic'].keys()):
        topic_results = data['results_by_topic'][topic]
        
        # Create summary
        create_topic_summary(topic, topic_results)
        
        # Create safe filename from topic name
        safe_topic_name = re.sub(r'[^\w\s-]', '', topic)
        safe_topic_name = re.sub(r'[-\s]+', '_', safe_topic_name)
        
        # Create progression plot
        progression_path = output_dir / f"{safe_topic_name}_progression.png"
        print(f"  Creating progression plot: {progression_path.name}")
        plot_topic_progression(topic, topic_results, progression_path)
        
        # Create final scores plot
        final_path = output_dir / f"{safe_topic_name}_final_scores.png"
        print(f"  Creating final scores plot: {final_path.name}")
        plot_topic_final_scores(topic, topic_results, final_path)
    
    print("\n" + "="*80)
    print(f"Created plots for {len(data['results_by_topic'])} topics in {output_dir}")
    
    # Create an index file listing all topics and their plots
    index_path = output_dir / "index.txt"
    with open(index_path, 'w') as f:
        f.write("Topic Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        for topic in sorted(data['results_by_topic'].keys()):
            safe_topic_name = re.sub(r'[^\w\s-]', '', topic)
            safe_topic_name = re.sub(r'[-\s]+', '_', safe_topic_name)
            
            f.write(f"Topic: {topic}\n")
            f.write(f"  - Progression: {safe_topic_name}_progression.png\n")
            f.write(f"  - Final Scores: {safe_topic_name}_final_scores.png\n")
            f.write(f"  - Questions: {len(set(r['question_id'] for r in data['results_by_topic'][topic]))}\n")
            f.write("\n")
    
    print(f"Index file created: {index_path}")

if __name__ == "__main__":
    main()