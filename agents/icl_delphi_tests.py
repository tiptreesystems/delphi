import argparse
import os

from dotenv import load_dotenv

from dataset.dataloader import ForecastDataLoader
from delphi_runner import initialize_experts, run_delphi_rounds, select_experts
from utils.forecast_loader import (
    load_forecasts
)
from utils.logs import save_delphi_log
from utils.llm_config import get_llm_from_config
from utils.utils import load_experiment_config, setup_environment

load_dotenv()


async def run_delphi_experiment(config: dict):
    """Run the Delphi experiment based on configuration."""
    # Setup environment
    setup_environment(config)
    
    # Initialize components
    llm = get_llm_from_config(config, role='expert')
    loader = ForecastDataLoader()
    
    # Load forecasts
    questions, llmcasts_by_qid_sfid, example_pairs_by_qid_sfid = await load_forecasts(config, loader, llm)
    
    # Get configuration values
    output_dir = config['experiment']['output_dir']
    selected_resolution_date = config['data']['resolution_date']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each question
    for question in questions:
        output_pattern = config['output']['file_pattern']
        output_file = os.path.join(
            output_dir, 
            output_pattern.format(
                question_id=question.id,
                resolution_date=selected_resolution_date
            )
        )
        
        if os.path.exists(output_file) and config['processing']['skip_existing']:
            print(f"Skipping {output_file} (already exists)")
            continue
        
        llmcasts_by_sfid = llmcasts_by_qid_sfid.get(question.id, {})
        example_pairs = example_pairs_by_qid_sfid.get(question.id, {})
        
        if not llmcasts_by_sfid:
            print(f"No forecasts found for question {question.id}, skipping")
            continue
        
        # Initialize and select experts
        experts = initialize_experts(llmcasts_by_sfid, config, llm)
        experts = select_experts(experts, config)
        
        print(f"Running Delphi for question {question.id} with {len(experts)} experts")
        
        # Run Delphi rounds
        delphi_log = await run_delphi_rounds(question, experts, config, example_pairs)
        
        # Save the log
        save_delphi_log(delphi_log, output_file)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Delphi experiment")
    parser.add_argument("config_path", help="Path to experiment configuration YAML file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_experiment_config(args.config_path)
    
    # Run experiment
    import asyncio
    asyncio.run(run_delphi_experiment(config))