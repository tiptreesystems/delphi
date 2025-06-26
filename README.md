# Delphi

A Python implementation of the Delphi Method for probabilistic forecasting using LLMs. This system creates panels of AI experts that deliberate on forecasting questions, comparing their performance against human forecasters.

## Features

- **Multi-round Delphi Method**: Supports both single and two-round expert deliberation
- **Multiple LLM Support**: Works with Claude (Haiku, Sonnet, Opus) and OpenAI models
- **Expert Personas**: Uses real human forecaster profiles to create diverse expert panels
- **Parallel Processing**: Efficient evaluation across multiple questions and panel sizes

## Installation

```bash
pip install -r requirements.txt
```

Set up API keys in `.env`:
```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## Usage

Run experiments with different configurations:

```bash
# Default configuration
python run_experiment.py

# Specific configuration
python run_experiment.py config_delphi_2round.yml

# Quick test with fewer questions
python run_experiment.py config_quick_test.yml
```

## Key Components

- `delphi.py`: Core Delphi panel implementation
- `eval.py`: Evaluation framework and metrics
- `models.py`: LLM abstraction layer
- `configs/`: Experiment configurations
- `visualize_delphi_results.py`: Generate analysis plots

