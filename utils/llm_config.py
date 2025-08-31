"""
Shared LLM configuration utilities to avoid circular imports.
"""

from utils.models import LLMProvider, LLMModel, LLMFactory
from utils.prompt_loader import load_prompt


def get_llm_from_config(config: dict, role: str = None):
    """
    Create LLM instance from configuration.

    Args:
        config: Configuration dictionary
        role: Optional role ('expert', 'mediator', or 'learner') to get role-specific config
    """
    model_config = config['model'].copy()

    # If a role is specified, use role-specific config (now required)
    if role:
        if role not in model_config:
            raise ValueError(f"Role '{role}' not found in model configuration. Must specify configuration for each component.")
        role_config = model_config[role]
        # Role config must have provider and model
        if 'provider' not in role_config:
            raise ValueError(f"'provider' is required in model.{role} configuration")
        if 'model' not in role_config and 'name' not in role_config:
            raise ValueError(f"'model' or 'name' is required in model.{role} configuration")

        model_config['provider'] = role_config['provider']
        model_config['name'] = role_config.get('model', role_config.get('name'))

        # Handle system prompt references
        if 'system_prompt' in role_config:
            # For backward compatibility, warn if inline prompt is used
            import warnings
            warnings.warn(
                f"Inline system_prompt in config for {role} is deprecated. "
                "Use 'system_prompt_name' and 'system_prompt_version' instead.",
                DeprecationWarning
            )
            model_config['system_prompt'] = role_config['system_prompt']
        elif 'system_prompt_name' in role_config:
            # Load prompt from reference
            prompt_name = role_config['system_prompt_name']
            prompt_version = role_config.get('system_prompt_version', 'v1')
            model_config['system_prompt'] = load_prompt(prompt_name, prompt_version)
    else:
        # No role specified - this should not be used anymore
        raise ValueError("Role must be specified when getting LLM from config. Use 'expert', 'mediator', or 'learner'.")

    # Map string provider to enum
    provider_map = {
        'openai': LLMProvider.OPENAI,
        'claude': LLMProvider.CLAUDE,
        'anthropic': LLMProvider.CLAUDE,
        'groq': LLMProvider.GROQ,
    }
    provider = provider_map.get(model_config['provider'].lower())

    # Map model name to enum (you may need to extend this)
    model_map = {
        'gpt-4o-2024-05-13': LLMModel.GPT_4O_2024_05_13,
        'gpt-4o': LLMModel.GPT_4O,
        'openai/gpt-oss-20b': LLMModel.GROQ_GPT_OSS_20B,
        'openai/gpt-oss-120b': LLMModel.GROQ_GPT_OSS_120B,
        'moonshotai/kimi-k2-instruct': LLMModel.GROQ_KIMI_K2_INSTRUCT,
        'llama-3.3-70b-versatile': LLMModel.GROQ_LLAMA_3_3_70B,
        'llama-3.1-70b-versatile': LLMModel.GROQ_LLAMA_3_1_70B,
        'meta-llama/llama-4-maverick-17b-128e-instruct': LLMModel.GROQ_LLAMA_4_MAVERICK_17B,
        'qwen/qwen3-32b': LLMModel.GROQ_QWEN3_32B,
        'deepseek-r1-distill-llama-70b': LLMModel.GROQ_DEEPSEEK_R1_DISTILL_70B,
        'o3-2025-04-16': LLMModel.O3_2025_04_16,
        'gpt-5-2025-08-07': LLMModel.GPT_5_2025_08_07,
        'o1-2024-12-17': LLMModel.O1_2024_12_17,
        'claude-3-7-sonnet-20250219': LLMModel.CLAUDE_3_7_SONNET,
    }
    model = model_map.get(model_config['name'].lower())

    system_prompt = model_config.get('system_prompt', '')

    return LLMFactory.create_llm(provider, model, system_prompt=system_prompt)