import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from delphi import DelphiPanel, ForecastDataLoader
from models import LLMProvider, LLMModel
from tqdm import tqdm
import dotenv
import json
import pandas as pd
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import random
from loguru import logger

dotenv.load_dotenv()

def load_config(config_path: str = "configs/config.yml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_single_question(
    question,
    question_id: str,
    delphi_panel: DelphiPanel,
    loader: ForecastDataLoader,
) -> Optional[Dict]:
    """Process a single question and return metrics."""
    # Get human forecasts
    human_forecasts = loader.get_super_forecasts(question_id)
    human_values = [f.forecast for f in human_forecasts]
    
    # Get resolution
    resolution = loader.get_resolution(question_id)
    if not resolution:
        return None
    
    outcome = float(resolution.resolved_to)
    
    # Calculate human statistics
    human_mean = np.mean(human_values) if human_values else None
    human_std = np.std(human_values) if human_values else None
    human_brier = (human_mean - outcome) ** 2 if human_mean is not None else None
    human_mae = abs(human_mean - outcome) if human_mean is not None else None
    
    # AI forecasts 
    results = delphi_panel.forecast_question(question_id)
    
    # Calculate metrics
    brier = (results['aggregate'] - outcome) ** 2
    mae = abs(results['aggregate'] - outcome)
    
    return {
        'question_id': question_id,
        'question_text': question.question,
        'outcome': outcome,
        'human_values': human_values,
        'human_mean': human_mean,
        'human_std': human_std,
        'human_brier': human_brier,
        'human_mae': human_mae,
        'sampled_expert_ids': results['selected_experts'],
        'num_experts': len(results['selected_experts']),
        'round1_responses': results['round1_responses'],
        'round2_responses': results['round2_responses'],
        'individual_forecasts': results['individual_forecasts'],
        'aggregate': results['aggregate'],
        'brier': brier,
        'mae': mae,
        # Add human group performance metrics
        'human_group_mean': results.get('human_group_mean'),
        'human_group_std': results.get('human_group_std'),
        'human_group_brier': results.get('human_group_brier'),
        'human_group_mae': results.get('human_group_mae')
    }

def run_comprehensive_evaluation(
    config: Optional[Dict] = None,
    n_experts_range: Optional[List[int]] = None,
    n_questions: Optional[int] = None,
    provider: Optional[LLMProvider] = None,
    model: Optional[LLMModel] = None,
    max_workers: Optional[int] = None
) -> Dict:
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Use config values with backwards compatibility
    n_experts_range = n_experts_range or config['evaluation']['n_experts_range']
    n_questions = n_questions or config['evaluation']['n_questions']
    max_workers = max_workers or config['evaluation']['max_workers']
    n_groups = config['evaluation'].get('n_groups', 1)  # Default to 1 group if not specified
    
    # Parse provider and model from config
    if provider is None:
        provider_str = config['model']['provider'].upper()
        provider = LLMProvider[provider_str]
    
    if model is None:
        model_name = config['model']['name'].upper().replace('-', '_')
        model = LLMModel[model_name]
    
    # Set random seed for reproducible question sampling
    random_seed = config['evaluation'].get('random_seed', None)
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        print(f"Random seed set to: {random_seed}")
    
    loader = ForecastDataLoader()
    resolved_questions = loader.get_resolved_questions()[:n_questions]
    
    # Sort questions by ID for consistent ordering
    resolved_questions = sorted(resolved_questions, key=lambda q: q.id)[:n_questions]
    
    if len(resolved_questions) < n_questions:
        print(f"Warning: Only {len(resolved_questions)} resolved questions available")
    
    timestamp = datetime.now().strftime(config['experiment']['timestamp_format'])
    results_dir = config['experiment']['results_dir'].format(timestamp=timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize logging to both file and console for robust run tracking
    log_file = os.path.join(results_dir, "run.log")
    # Set up loguru to log to file and stdout
    logger.add(log_file, level="INFO", enqueue=True)
    logger.info(f"Logs will be written to {log_file}")
    
    # Check for existing partial results to enable resuming
    partial_path = os.path.join(results_dir, "partial_results.json")
    if os.path.exists(partial_path):
        with open(partial_path, "r") as pf:
            results_by_expert = json.load(pf)
        # Keys loaded as strings; convert to int
        results_by_expert = {int(k): v for k, v in results_by_expert.items()}
        logger.info(f"Resuming from partial results at {partial_path}")
    else:
        # Initialize with groups structure
        results_by_expert = {}
        for n in n_experts_range:
            results_by_expert[n] = {
                'groups': {str(g): {'questions': [], 'metrics': {'brier': [], 'mae': []}} for g in range(n_groups)},
                'questions': [],  # Keep for backwards compatibility
                'metrics': {'brier': [], 'mae': []}  # Keep for backwards compatibility
            }
    
    all_results = {
        'metadata': {
            'timestamp': timestamp,
            'n_questions': len(resolved_questions),
            'provider': provider.value,
            'model': model.value,
            'n_experts_range': n_experts_range,
            'n_groups': n_groups,
            'config': config  # Store config in results
        },
        'questions': {},
        'aggregate_metrics': {}
    }
    
    # Collect all data in parallel
    print(f"Collecting forecasts with {n_groups} group(s) per expert count...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for n_experts in n_experts_range:
            for group_idx in range(n_groups):
                # Create panels for each group
                delphi_panel = DelphiPanel(
                    loader, 
                    provider=provider,
                    model=model,
                    n_experts=n_experts,
                    condition_on_data=True,
                    system_prompt=config['model']['system_prompt'],
                    config=config
                )
                
                # Determine already processed question IDs to skip when resuming
                if 'groups' in results_by_expert[n_experts]:
                    processed_qids = {q['question_id'] for q in results_by_expert[n_experts]['groups'][str(group_idx)]['questions']}
                else:
                    processed_qids = set()

                for question in resolved_questions:
                    if question.id in processed_qids:
                        continue  # skip already done
                    future = executor.submit(
                        process_single_question,
                        question,
                        question.id,
                        delphi_panel,
                        loader,
                    )
                    futures.append((future, n_experts, group_idx, question.id))
        
        # Collect results
        for future, n_experts, group_idx, qid in tqdm(futures, desc="Processing forecasts"):
            try:
                result = future.result()
            except Exception as e:
                logger.exception(f"Exception processing question {qid} with {n_experts} experts (group {group_idx}): {e}")
                continue
            if result:
                # Add group index to result
                result['group_idx'] = group_idx
                
                # Store in groups structure
                if 'groups' not in results_by_expert[n_experts]:
                    results_by_expert[n_experts]['groups'] = {str(g): {'questions': [], 'metrics': {'brier': [], 'mae': []}} for g in range(n_groups)}
                
                results_by_expert[n_experts]['groups'][str(group_idx)]['questions'].append(result)
                results_by_expert[n_experts]['groups'][str(group_idx)]['metrics']['brier'].append(result['brier'])
                results_by_expert[n_experts]['groups'][str(group_idx)]['metrics']['mae'].append(result['mae'])
                
                # Also add to legacy structure for backwards compatibility
                results_by_expert[n_experts]['questions'].append(result)
                results_by_expert[n_experts]['metrics']['brier'].append(result['brier'])
                results_by_expert[n_experts]['metrics']['mae'].append(result['mae'])
                
                # Save incremental checkpoint after each successful result
                with open(f"{results_dir}/partial_results.json", 'w') as pf:
                    json.dump(results_by_expert, pf)
    
    # Now create all plots sequentially
    print("Creating plots...")
    for n_experts in tqdm(n_experts_range, desc="Creating plots"):
        expert_results = results_by_expert[n_experts]
        
        # Calculate aggregate metrics across all groups
        all_briers = []
        all_maes = []
        
        if 'groups' in expert_results:
            for group_data in expert_results['groups'].values():
                all_briers.extend(group_data['metrics']['brier'])
                all_maes.extend(group_data['metrics']['mae'])
        else:
            # Fallback to legacy structure
            all_briers = expert_results['metrics']['brier']
            all_maes = expert_results['metrics']['mae']
        
        all_results['aggregate_metrics'][n_experts] = {
            'mean_brier': np.mean(all_briers) if all_briers else 0,
            'std_brier': np.std(all_briers) if all_briers else 0,
            'mean_mae': np.mean(all_maes) if all_maes else 0,
            'std_mae': np.std(all_maes) if all_maes else 0,
            'n_groups': n_groups,
            'n_samples': len(all_briers)
        }
        
        # Store individual question results with group information
        if 'groups' in expert_results:
            for group_idx, group_data in expert_results['groups'].items():
                for question_result in group_data['questions']:
                    qid = question_result['question_id']
                    if qid not in all_results['questions']:
                        all_results['questions'][qid] = {}
                    if n_experts not in all_results['questions'][qid]:
                        all_results['questions'][qid][n_experts] = {'groups': {}}
                    all_results['questions'][qid][n_experts]['groups'][group_idx] = question_result
        else:
            # Fallback to legacy structure
            for question_result in expert_results['questions']:
                qid = question_result['question_id']
                if qid not in all_results['questions']:
                    all_results['questions'][qid] = {}
                all_results['questions'][qid][n_experts] = question_result
    
    # Save detailed results
    with open(f"{results_dir}/detailed_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create CSV summary
    create_csv_summary(all_results, results_dir)
    
    
    return all_results

def create_csv_summary(results: Dict, results_dir: str):
    rows = []
    
    # Iterate through questions and their results for different expert counts
    for question_id, expert_data in results['questions'].items():
        for n_experts, question_data in expert_data.items():
            if isinstance(question_data, dict) and 'groups' in question_data:
                # New structure with groups - calculate averages across groups
                group_aggregates = []
                group_briers = []
                group_maes = []
                outcomes = []
                human_means = []
                human_stds = []
                human_briers = []
                human_maes = []
                num_human_forecasts = []
                # Add human group metrics
                human_group_means = []
                human_group_stds = []
                human_group_briers = []
                human_group_maes = []
                
                for group_idx, group_result in question_data['groups'].items():
                    group_aggregates.append(group_result['aggregate'])
                    group_briers.append(group_result['brier'])
                    group_maes.append(group_result['mae'])
                    outcomes.append(group_result['outcome'])
                    if group_result['human_mean'] is not None:
                        human_means.append(group_result['human_mean'])
                        human_stds.append(group_result['human_std'])
                        human_briers.append(group_result['human_brier'])
                        human_maes.append(group_result['human_mae'])
                        num_human_forecasts.append(len(group_result['human_values']))
                    # Add human group metrics
                    if group_result.get('human_group_mean') is not None:
                        human_group_means.append(group_result['human_group_mean'])
                        human_group_stds.append(group_result['human_group_std'])
                        human_group_briers.append(group_result['human_group_brier'])
                        human_group_maes.append(group_result['human_group_mae'])
                
                # Create averaged row
                row = {
                    'question_id': question_id,
                    'n_experts': n_experts,
                    'n_groups': len(question_data['groups']),
                    'outcome': outcomes[0] if outcomes else None,  # Should be same for all groups
                    'ai_aggregate': np.mean(group_aggregates),
                    'ai_aggregate_std': np.std(group_aggregates),
                    'ai_brier': np.mean(group_briers),
                    'ai_brier_std': np.std(group_briers),
                    'ai_mae': np.mean(group_maes),
                    'ai_mae_std': np.std(group_maes),
                    'human_mean': np.mean(human_means) if human_means else None,
                    'human_std': np.mean(human_stds) if human_stds else None,
                    'human_brier': np.mean(human_briers) if human_briers else None,
                    'human_mae': np.mean(human_maes) if human_maes else None,
                    'num_human_forecasts': int(np.mean(num_human_forecasts)) if num_human_forecasts else 0,
                    # Add human group metrics
                    'human_group_mean': np.mean(human_group_means) if human_group_means else None,
                    'human_group_mean_std': np.std(human_group_means) if human_group_means else None,
                    'human_group_brier': np.mean(human_group_briers) if human_group_briers else None,
                    'human_group_brier_std': np.std(human_group_briers) if human_group_briers else None,
                    'human_group_mae': np.mean(human_group_maes) if human_group_maes else None,
                    'human_group_mae_std': np.std(human_group_maes) if human_group_maes else None
                }
                rows.append(row)
            else:
                # Legacy structure without groups
                question_result = question_data
                row = {
                    'question_id': question_id,
                    'n_experts': n_experts,
                    'n_groups': 1,
                    'outcome': question_result['outcome'],
                    'ai_aggregate': question_result['aggregate'],
                    'ai_aggregate_std': 0,
                    'ai_brier': question_result['brier'],
                    'ai_brier_std': 0,
                    'ai_mae': question_result['mae'],
                    'ai_mae_std': 0,
                    'human_mean': question_result['human_mean'],
                    'human_std': question_result['human_std'],
                    'human_brier': question_result['human_brier'],
                    'human_mae': question_result['human_mae'],
                    'num_human_forecasts': len(question_result['human_values'])
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/summary_results.csv", index=False)
    
    # Print summary statistics from aggregate metrics
    print("\nResults Summary by Number of Experts:")
    print("-" * 80)
    for n_experts, metrics in sorted(results['aggregate_metrics'].items()):
        print(f"\nN={n_experts} experts:")
        print(f"  Brier: {metrics['mean_brier']:.4f} (±{metrics['std_brier']:.4f})")
        print(f"  MAE: {metrics['mean_mae']:.4f} (±{metrics['std_mae']:.4f})")
        if 'n_groups' in metrics:
            print(f"  Groups: {metrics['n_groups']}, Total samples: {metrics['n_samples']}")

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    print("Running comprehensive evaluation with parallel processing...")
    print(f"Configuration loaded from config.yml")
    print(f"Model: {config['model']['provider']} - {config['model']['name']}")
    print(f"Questions: {config['evaluation']['n_questions']}")
    print(f"Expert range: {config['evaluation']['n_experts_range']}")
    
    results = run_comprehensive_evaluation(config=config)
    
    print(f"\nResults saved to: {results['metadata']['config']['experiment']['results_dir'].format(timestamp=results['metadata']['timestamp'])}/") 