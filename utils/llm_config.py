"""
Shared LLM configuration utilities to avoid circular imports.
"""

from typing import Optional
from utils.models import LLMProvider, LLMModel, LLMFactory
from utils.prompt_loader import load_prompt
from utils.config_types import RootConfig


def get_llm_from_config(config: RootConfig, role: str):
    """
    Create LLM instance from typed configuration for a specific role.
    """
    if not isinstance(config, RootConfig):
        raise TypeError("get_llm_from_config expects a typed RootConfig")

    # Extract role-specific model settings
    role_cfg = getattr(config.model, role, None)
    if role_cfg is None:
        raise ValueError(f"Role '{role}' not configured under model.*")

    if not role_cfg.provider:
        raise ValueError(f"'provider' is required in model.{role} configuration")
    name: Optional[str] = role_cfg.model or getattr(role_cfg, "name", None)
    if not name:
        raise ValueError(f"'model' (or 'name') is required in model.{role}")

    # Provider mapping
    provider_map = {
        "openai": LLMProvider.OPENAI,
        "claude": LLMProvider.CLAUDE,
        "anthropic": LLMProvider.CLAUDE,
        "groq": LLMProvider.GROQ,
    }
    provider = provider_map.get(role_cfg.provider.lower())

    # Model mapping to enum if known; else keep string
    model_map = {
        "gpt-4o-2024-05-13": LLMModel.GPT_4O_2024_05_13,
        "gpt-4o": LLMModel.GPT_4O,
        "openai/gpt-oss-20b": LLMModel.GROQ_GPT_OSS_20B,
        "openai/gpt-oss-120b": LLMModel.GROQ_GPT_OSS_120B,
        "moonshotai/kimi-k2-instruct": LLMModel.GROQ_KIMI_K2_INSTRUCT,
        "llama-3.3-70b-versatile": LLMModel.GROQ_LLAMA_3_3_70B,
        "llama-3.1-70b-versatile": LLMModel.GROQ_LLAMA_3_1_70B,
        "meta-llama/llama-4-maverick-17b-128e-instruct": LLMModel.GROQ_LLAMA_4_MAVERICK_17B,
        "qwen/qwen3-32b": LLMModel.GROQ_QWEN3_32B,
        "deepseek-r1-distill-llama-70b": LLMModel.GROQ_DEEPSEEK_R1_DISTILL_70B,
        "o3-2025-04-16": LLMModel.O3_2025_04_16,
        "gpt-5-2025-08-07": LLMModel.GPT_5_2025_08_07,
        "o1-2024-12-17": LLMModel.O1_2024_12_17,
        "claude-3-7-sonnet-20250219": LLMModel.CLAUDE_3_7_SONNET,
    }
    model = model_map.get(name.lower(), name)

    # System prompt resolution
    system_prompt = role_cfg.system_prompt or ""
    if not system_prompt and role_cfg.system_prompt_name:
        prompt_version = role_cfg.system_prompt_version or "v1"
        system_prompt = load_prompt(role_cfg.system_prompt_name, prompt_version)

    return LLMFactory.create_llm(provider, model, system_prompt=system_prompt)
