# Genetic Algorithm Prompt Evolution

This module implements a sophisticated genetic algorithm system for evolving prompts to maximize forecasting performance. The system replaces the diff-based prompt learning approach with an evolutionary strategy that maintains a constant population of eight prompts and uses LLM-based mutations.

## Key Features

### Genetic Algorithm Components
- **Population Size**: Constant population of 8 prompt candidates
- **Elitism**: Top 2 prompts are copied unchanged to next generation  
- **Tournament Selection**: Size 3 tournaments for parent selection
- **Crossover**: Single-point crossover between parent prompts (50% probability)
- **Mutation**: LLM-based mutations including rephrase, add_detail, simplify, change_focus (50% probability)
- **Adaptive Mutation Rate**: Starts at 0.5, increases by 0.1 when stagnant (max 0.9)

### Fitness Evaluation
- **Task Performance**: Primary fitness based on forecasting accuracy
- **Length Penalty**: Optional penalty for overly long/short prompts
- **Batch Evaluation**: All candidates evaluated on same validation batch
- **Calibrated Scoring**: Uses metrics like Brier score for proper calibration

### Superforecaster Integration
- **Reasoning Templates**: Incorporates superforecaster methodology
- **ICL Examples**: Can include in-context learning examples from expert forecasters
- **Topic Filtering**: Examples can be filtered by topic relevance
- **Flexible Enhancement**: Configurable reasoning and example inclusion

### Termination Conditions
- **Max Generations**: 40 generations maximum
- **Stagnation**: Stops after 8 generations without improvement
- **Adaptive Exploration**: Increases mutation rate after 4 stagnant generations

## Architecture

### Core Classes

#### `GeneticPromptOptimizer`
Main orchestrator class that manages the evolution process:
- Configures population and fitness evaluation
- Handles LLM-based mutations
- Manages logging and progress tracking
- Integrates with superforecaster utilities

#### `PromptPopulation` 
Manages the population of prompt candidates:
- Implements elitism and generational replacement
- Tracks fitness statistics and evolution history
- Handles adaptive mutation rate adjustment
- Supports both LLM and fallback mutations

#### `PromptCandidate`
Represents individual prompts in the population:
- Stores prompt text and fitness score
- Tracks generation and parent information
- Supports genealogy tracking for analysis

### Genetic Operators

#### `tournament_selection()`
Selects parents using tournament selection with configurable size.

#### `single_point_crossover()`
Combines two parent prompts at a random crossover point using tokenization.

#### `mutate()` (async, LLM-based)
Uses the LLM to generate sophisticated mutations:
- **Rephrase**: Rewrite for better effectiveness while keeping meaning
- **Add Detail**: Include helpful instructions or details
- **Simplify**: Remove unnecessary complexity 
- **Change Focus**: Emphasize different but important aspects

#### `mutate_sync()` (fallback)
Simple text-based mutations when LLM is unavailable.

### Utilities Integration

#### `SuperforecasterManager`
Located in `utils/superforecaster_utils.py`:
- Loads examples from JSON files
- Provides reasoning templates
- Enhances prompts with expert context
- Supports topic-based example filtering

## Usage

### Basic Evolution

```python
from genetic_evolution import GeneticPromptOptimizer, FitnessConfig
from utils.superforecaster_utils import get_superforecaster_manager

# Configure fitness evaluation
superforecaster_manager = get_superforecaster_manager("examples.json")
fitness_config = FitnessConfig(
    length_penalty_weight=0.1,
    superforecaster_manager=superforecaster_manager,
    include_reasoning=True,
    include_examples=True,
    n_examples=3
)

# Create optimizer
optimizer = GeneticPromptOptimizer(
    llm=your_llm,
    fitness_config=fitness_config
)

# Set evaluation function
optimizer.set_evaluation_function(your_evaluation_function)

# Run evolution
results = await optimizer.evolve(
    seed_prompts=["Forecast the probability.", "Predict the outcome."],
    validation_batch=your_validation_data,
    max_generations=40
)
```

### Custom Fitness Evaluation

The evaluation function should accept prompts and return fitness scores:

```python
async def evaluate_fitness(prompts: List[str], validation_batch) -> List[float]:
    scores = []
    for prompt in prompts:
        # Test prompt on validation tasks
        # Calculate performance metric (e.g., Brier score)
        # Return score where higher = better
        score = calculate_performance(prompt, validation_batch)
        scores.append(score)
    return scores

optimizer.set_evaluation_function(evaluate_fitness)
```

### Superforecaster Examples

Create a JSON file with expert examples:

```json
{
  "examples": [
    {
      "question": "Will the S&P 500 close higher next month?",
      "reasoning": "I examine historical base rates (58% positive months)...",
      "forecast": 0.65,
      "base_rate": 0.58,
      "key_factors": ["volatility", "Fed policy", "earnings"],
      "topic": "finance",
      "resolution": true
    }
  ]
}
```

## Configuration Options

### Population Configuration
```python
population_config = {
    'population_size': 8,           # Number of prompts in population
    'elitism_size': 2,              # Top prompts to preserve
    'tournament_size': 3,           # Tournament selection size
    'initial_mutation_rate': 0.5,   # Starting mutation probability
    'mutation_rate_increase': 0.1,  # Increase when stagnant
    'max_mutation_rate': 0.9,       # Maximum mutation rate
    'stagnation_threshold': 4,      # Generations before increasing mutation
    'max_stagnation': 8            # Generations before termination
}
```

### Fitness Configuration
```python
fitness_config = FitnessConfig(
    length_penalty_weight=0.1,      # Weight for length penalty
    target_length=50,               # Target prompt length in tokens
    superforecaster_manager=manager, # Superforecaster integration
    include_reasoning=True,         # Include reasoning template
    include_examples=True,          # Include ICL examples
    n_examples=3                    # Number of examples to include
)
```

## Testing

Run the test suite to verify functionality:

```bash
PYTHONPATH=. python3 genetic_evolution/test_genetic_evolution.py
```

The test includes:
- Mock LLM for testing without API costs
- Mock fitness evaluator with realistic scoring
- Verification of evolution dynamics
- Integration testing with superforecaster utilities

## Logging and Results

The system provides comprehensive logging:
- Generation-by-generation progress
- Fitness statistics and trends  
- Mutation rate adaptations
- Best prompts and improvements
- Detailed timing information

Results are saved in JSON format including:
- Evolution summary and statistics
- Final population with genealogy
- Configuration parameters
- Fitness history over generations

## Integration with Existing System

This genetic evolution system is designed to integrate with the existing Delphi forecasting pipeline:

- Uses existing `utils/models.py` LLM infrastructure
- Integrates with `utils/prompt_loader.py` system
- Leverages existing forecasting evaluation metrics
- Compatible with current configuration and data loading systems
- Extends the `prompt_learning/` module capabilities

The system provides a more sophisticated alternative to diff-based prompt learning while maintaining compatibility with existing forecasting workflows.