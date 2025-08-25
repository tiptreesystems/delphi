"""
Test script to verify parallel mutations are working correctly.
"""

import asyncio
import time
from genetic_evolution.prompt_population import PromptPopulation
from genetic_evolution.operators import PromptCandidate


class MockLLM:
    """Mock LLM that simulates API delay to test parallelism."""
    
    def __init__(self, delay=0.5):
        self.delay = delay
        self.call_count = 0
        
    async def generate_response(self, prompt, max_tokens=200, temperature=0.7):
        self.call_count += 1
        # Simulate API delay
        await asyncio.sleep(self.delay)
        # Return a simple mutation
        return f"Improved version of the prompt (call #{self.call_count})"


async def test_parallel_vs_sequential():
    """Compare parallel vs sequential mutation performance."""
    
    print("Testing Parallel Mutations Performance")
    print("=" * 50)
    
    # Create test population
    population = PromptPopulation(
        population_size=6,
        elitism_size=1,
        tournament_size=2,
        initial_mutation_rate=1.0  # Force all mutations
    )
    
    # Initialize with test prompts
    seed_prompts = [
        "Forecast the probability carefully",
        "Predict the outcome systematically", 
        "Analyze the question thoroughly",
        "Consider all relevant factors"
    ]
    
    population.initialize_population(seed_prompts)
    
    # Set fitness scores (need 6 scores for population size 6)
    population.evaluate_fitness([0.8, 0.6, 0.7, 0.9, 0.5, 0.3])
    
    # Test with different concurrency settings
    concurrency_levels = [1, 3, 5]
    delay = 0.3  # Simulate 300ms API calls
    
    for max_concurrent in concurrency_levels:
        print(f"\n--- Testing with max_concurrent = {max_concurrent} ---")
        
        # Create fresh population copy
        test_population = PromptPopulation(
            population_size=6,
            elitism_size=1, 
            tournament_size=2,
            initial_mutation_rate=1.0
        )
        test_population.initialize_population(seed_prompts)
        test_population.evaluate_fitness([0.8, 0.6, 0.7, 0.9, 0.5, 0.3])
        test_population.max_concurrent_mutations = max_concurrent
        
        # Create mock LLM with delay
        mock_llm = MockLLM(delay=delay)
        
        # Time the evolution
        start_time = time.time()
        await test_population.evolve_generation(llm=mock_llm)
        elapsed = time.time() - start_time
        
        expected_sequential_time = mock_llm.call_count * delay
        speedup = expected_sequential_time / elapsed if elapsed > 0 else 0
        
        print(f"  Mutations performed: {mock_llm.call_count}")
        print(f"  Time elapsed: {elapsed:.2f}s")
        print(f"  Expected sequential time: {expected_sequential_time:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")
        
        if max_concurrent > 1 and speedup > 1.5:
            print(f"  ✅ Parallel execution working!")
        elif max_concurrent == 1:
            print(f"  📝 Sequential baseline")
        else:
            print(f"  ⚠️  Limited parallelism detected")


async def test_error_handling():
    """Test that parallel mutations handle errors gracefully."""
    
    print("\n" + "=" * 50)
    print("Testing Error Handling in Parallel Mutations")
    print("=" * 50)
    
    class FailingMockLLM:
        def __init__(self, failure_rate=0.5):
            self.failure_rate = failure_rate
            self.call_count = 0
            
        async def generate_response(self, prompt, max_tokens=200, temperature=0.7):
            self.call_count += 1
            if self.call_count % 2 == 0:  # Fail every other call
                raise Exception("Mock API failure")
            return f"Success call #{self.call_count}"
    
    # Test with failing LLM
    population = PromptPopulation(
        population_size=6,
        elitism_size=1,
        tournament_size=2, 
        initial_mutation_rate=1.0
    )
    
    population.initialize_population([
        "Test prompt 1", "Test prompt 2", "Test prompt 3", "Test prompt 4"
    ])
    population.evaluate_fitness([0.8, 0.6, 0.7, 0.9, 0.5, 0.3])
    population.max_concurrent_mutations = 3
    
    failing_llm = FailingMockLLM()
    
    try:
        await population.evolve_generation(llm=failing_llm)
        print(f"✅ Evolution completed despite {failing_llm.call_count//2} API failures")
        print(f"   Total API calls: {failing_llm.call_count}")
        print(f"   Final population size: {len(population.population)}")
    except Exception as e:
        print(f"❌ Evolution failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_parallel_vs_sequential())
    asyncio.run(test_error_handling())