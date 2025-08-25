"""
Prompt loader utility for loading versioned prompts from markdown files.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any


class PromptLoader:
    """Loads prompts from markdown files with version support."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the prompt loader.
        
        Args:
            prompts_dir: Directory containing the prompt files
        """
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}
    
    def load_prompt(self, prompt_name: str, version: str = "v1") -> str:
        """
        Load a prompt from a markdown file.
        
        Args:
            prompt_name: Name of the prompt (directory name)
            version: Version of the prompt (filename without .md)
        
        Returns:
            The prompt content as a string
        
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        cache_key = f"{prompt_name}:{version}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        prompt_path = self.prompts_dir / prompt_name / f"{version}.md"
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        self._cache[cache_key] = content
        return content
    
    def list_prompts(self) -> Dict[str, list]:
        """
        List all available prompts and their versions.
        
        Returns:
            Dictionary mapping prompt names to list of available versions
        """
        prompts = {}
        
        if not self.prompts_dir.exists():
            return prompts
        
        for prompt_dir in self.prompts_dir.iterdir():
            if prompt_dir.is_dir():
                versions = []
                for file in prompt_dir.iterdir():
                    if file.suffix == '.md':
                        versions.append(file.stem)
                if versions:
                    prompts[prompt_dir.name] = sorted(versions)
        
        return prompts
    
    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()


# Singleton instance
_prompt_loader = None


def get_prompt_loader(prompts_dir: str = "prompts") -> PromptLoader:
    """
    Get or create the singleton prompt loader instance.
    
    Args:
        prompts_dir: Directory containing the prompt files
    
    Returns:
        The prompt loader instance
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader(prompts_dir)
    return _prompt_loader


def load_prompt(prompt_name: str, version: str = "v1", **kwargs) -> str:
    """
    Convenience function to load and format a prompt.
    
    Args:
        prompt_name: Name of the prompt
        version: Version of the prompt
        **kwargs: Variables to format into the prompt
    
    Returns:
        The formatted prompt string
    """
    loader = get_prompt_loader()
    prompt = loader.load_prompt(prompt_name, version)
    
    if kwargs:
        # Only format if variables are provided
        prompt = prompt.format(**kwargs)
    
    return prompt


# Preload commonly used prompts as constants for backward compatibility
def load_all_prompts() -> Dict[str, str]:
    """Load all v1 prompts for backward compatibility."""
    loader = get_prompt_loader()
    prompts = {}
    
    prompt_mapping = {
        'EXPERT_FORECAST_PROMPT': 'expert_forecast',
        'EXPERT_FORECAST_WITH_EXAMPLES_PROMPT': 'expert_forecast/v1_with_examples',
        'EXPERT_ROUND1_PROMPT': 'expert_round1',
        'EXPERT_ROUND2_PROMPT': 'expert_round2',
        'DELPHI_ROUND_1_SURVEY': 'delphi_survey',
        'MEDIATOR_SYSTEM_PROMPT': 'mediator_system',
        'MEDIATOR_FEEDBACK_REQUEST': 'mediator_feedback',
        'EXPERT_SYSTEM_PROMPT_TEMPLATE': 'expert_system'
    }
    
    for const_name, prompt_path in prompt_mapping.items():
        try:
            if '/' in prompt_path:
                # Handle special case for v1_with_examples
                parts = prompt_path.split('/')
                prompts[const_name] = loader.load_prompt(parts[0], parts[1])
            else:
                prompts[const_name] = loader.load_prompt(prompt_path, 'v1')
        except FileNotFoundError:
            print(f"Warning: Could not load prompt {prompt_path}")
            prompts[const_name] = ""
    
    return prompts