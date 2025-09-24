"""
Typed configuration loader for Delphi experiments.

This module provides tolerant dataclass wrappers around the current YAML
configs so callers can opt into structured access without breaking the
existing dict-based flow. Unknown fields are preserved in an `extras` map
so we can evolve the schema safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ReuseInitialForecasts:
    enabled: bool = False
    source_dir: Optional[str] = None
    with_examples: Optional[bool] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "ReuseInitialForecasts":
        d = d or {}
        known = {k: d.get(k) for k in ("enabled", "source_dir", "with_examples")}
        extras = {k: v for k, v in d.items() if k not in known}
        return ReuseInitialForecasts(
            enabled=bool(known.get("enabled", False)),
            source_dir=known.get("source_dir"),
            with_examples=known.get("with_examples"),
            extras=extras,
        )


@dataclass
class ExperimentConfig:
    seed: int = 42
    output_dir: Optional[str] = None
    initial_forecasts_dir: Optional[str] = None
    reuse_initial_forecasts: ReuseInitialForecasts = field(
        default_factory=ReuseInitialForecasts
    )
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "ExperimentConfig":
        d = d or {}
        extras = {
            k: v
            for k, v in d.items()
            if k
            not in {
                "seed",
                "output_dir",
                "initial_forecasts_dir",
                "reuse_initial_forecasts",
            }
        }
        return ExperimentConfig(
            seed=int(d.get("seed", 42)),
            output_dir=d.get("output_dir") or d.get("output_path"),
            initial_forecasts_dir=d.get("initial_forecasts_dir"),
            reuse_initial_forecasts=ReuseInitialForecasts.from_dict(
                d.get("reuse_initial_forecasts")
            ),
            extras=extras,
        )


@dataclass
class ModelRoleConfig:
    provider: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    system_prompt_name: Optional[str] = None
    system_prompt_version: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    feedback_max_tokens: Optional[int] = None
    feedback_temperature: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "ModelRoleConfig":
        d = d or {}
        known_keys = {
            "provider",
            "model",
            "name",
            "system_prompt",
            "system_prompt_name",
            "system_prompt_version",
            "temperature",
            "max_tokens",
            "feedback_max_tokens",
            "feedback_temperature",
        }
        extras = {k: v for k, v in d.items() if k not in known_keys}
        model = d.get("model", d.get("name"))
        return ModelRoleConfig(
            provider=d.get("provider"),
            model=model,
            system_prompt=d.get("system_prompt"),
            system_prompt_name=d.get("system_prompt_name"),
            system_prompt_version=d.get("system_prompt_version"),
            temperature=d.get("temperature"),
            max_tokens=d.get("max_tokens"),
            feedback_max_tokens=d.get("feedback_max_tokens"),
            feedback_temperature=d.get("feedback_temperature"),
            extras=extras,
        )


@dataclass
class ModelConfig:
    expert: ModelRoleConfig = field(default_factory=ModelRoleConfig)
    mediator: ModelRoleConfig = field(default_factory=ModelRoleConfig)
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "ModelConfig":
        d = d or {}
        # Backward compatibility: allow root-level provider/name to be interpreted
        # as expert/mediator defaults, but keep structured roles when present.
        expert_cfg = d.get("expert") or d
        mediator_cfg = d.get("mediator") or d
        # If only one role provided, we still parse both to avoid KeyErrors.
        extras = {k: v for k, v in d.items() if k not in {"expert", "mediator"}}
        return ModelConfig(
            expert=ModelRoleConfig.from_dict(expert_cfg),
            mediator=ModelRoleConfig.from_dict(mediator_cfg),
            extras=extras,
        )


@dataclass
class SamplingConfig:
    method: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "SamplingConfig":
        d = d or {}
        extras = {k: v for k, v in d.items() if k != "method"}
        return SamplingConfig(method=d.get("method"), extras=extras)


@dataclass
class DataConfig:
    resolution_date: str = ""
    forecast_due_date: Optional[str] = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    filters: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "DataConfig":
        d = d or {}
        extras = {
            k: v
            for k, v in d.items()
            if k not in {"resolution_date", "sampling", "filters"}
        }
        return DataConfig(
            resolution_date=str(d.get("resolution_date", "")),
            forecast_due_date=d.get("forecast_due_date"),
            sampling=SamplingConfig.from_dict(d.get("sampling")),
            filters=d.get("filters") or {},
            extras=extras,
        )


@dataclass
class ProcessingConfig:
    skip_existing: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "ProcessingConfig":
        d = d or {}
        extras = {k: v for k, v in d.items() if k != "skip_existing"}
        return ProcessingConfig(
            skip_existing=bool(d.get("skip_existing", True)), extras=extras
        )


@dataclass
class OutputSaveConfig:
    conversation_histories: bool = True
    example_pairs: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "OutputSaveConfig":
        d = d or {}
        extras = {
            k: v
            for k, v in d.items()
            if k not in {"conversation_histories", "example_pairs"}
        }
        return OutputSaveConfig(
            conversation_histories=bool(d.get("conversation_histories", True)),
            example_pairs=bool(d.get("example_pairs", True)),
            extras=extras,
        )


@dataclass
class OutputConfig:
    file_pattern: str = "delphi_eval_{question_id}_{resolution_date}.json"
    save: OutputSaveConfig = field(default_factory=OutputSaveConfig)
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "OutputConfig":
        d = d or {}
        extras = {k: v for k, v in d.items() if k not in {"file_pattern", "save"}}
        return OutputConfig(
            file_pattern=str(
                d.get(
                    "file_pattern", "delphi_eval_{question_id}_{resolution_date}.json"
                )
            ),
            save=OutputSaveConfig.from_dict(d.get("save")),
            extras=extras,
        )


@dataclass
class ApiProviderConfig:
    api_key_env: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "ApiProviderConfig":
        d = d or {}
        extras = {k: v for k, v in d.items() if k != "api_key_env"}
        return ApiProviderConfig(api_key_env=d.get("api_key_env"), extras=extras)


@dataclass
class ApiConfig:
    openai: Optional[ApiProviderConfig] = None
    groq: Optional[ApiProviderConfig] = None
    anthropic: Optional[ApiProviderConfig] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "ApiConfig":
        d = d or {}
        return ApiConfig(
            openai=ApiProviderConfig.from_dict(d.get("openai"))
            if d.get("openai")
            else None,
            groq=ApiProviderConfig.from_dict(d.get("groq")) if d.get("groq") else None,
            anthropic=ApiProviderConfig.from_dict(d.get("anthropic"))
            if d.get("anthropic")
            else None,
            extras={
                k: v for k, v in d.items() if k not in {"openai", "groq", "anthropic"}
            },
        )


@dataclass
class DebugConfig:
    enabled: bool = False
    breakpoint_on_start: bool = False
    port: int = 5679
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "DebugConfig":
        d = d or {}
        extras = {
            k: v
            for k, v in d.items()
            if k not in {"enabled", "breakpoint_on_start", "port"}
        }
        return DebugConfig(
            enabled=bool(d.get("enabled", False)),
            breakpoint_on_start=bool(d.get("breakpoint_on_start", False)),
            port=int(d.get("port", 5679)),
            extras=extras,
        )


@dataclass
class RootConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    delphi: Dict[str, Any] = field(default_factory=dict)  # keep dict for now
    data: DataConfig = field(default_factory=DataConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    initial_forecasts: Dict[str, Any] = field(default_factory=dict)
    debug: DebugConfig = field(default_factory=DebugConfig)
    extras: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RootConfig":
        d = d or {}
        known = {
            "experiment",
            "model",
            "delphi",
            "data",
            "processing",
            "api",
            "output",
            "initial_forecasts",
            "debug",
        }
        return RootConfig(
            experiment=ExperimentConfig.from_dict(d.get("experiment")),
            model=ModelConfig.from_dict(d.get("model")),
            delphi=d.get("delphi") or {},
            data=DataConfig.from_dict(d.get("data")),
            processing=ProcessingConfig.from_dict(d.get("processing")),
            api=ApiConfig.from_dict(d.get("api")),
            output=OutputConfig.from_dict(d.get("output")),
            initial_forecasts=d.get("initial_forecasts") or {},
            debug=DebugConfig.from_dict(d.get("debug")),
            extras={k: v for k, v in d.items() if k not in known},
        )


def load_typed_experiment_config(path: str) -> RootConfig:
    """Load YAML config into typed RootConfig without altering existing loaders."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RootConfig.from_dict(raw)


def _compact(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys with None values (shallow)."""
    return {k: v for k, v in d.items() if v is not None}


def _role_to_dict(role: ModelRoleConfig) -> Dict[str, Any]:
    if role is None:
        return {}
    base = {
        "provider": role.provider,
        "model": role.model,
        "system_prompt": role.system_prompt,
        "system_prompt_name": role.system_prompt_name,
        "system_prompt_version": role.system_prompt_version,
        "temperature": role.temperature,
        "max_tokens": role.max_tokens,
        "feedback_max_tokens": role.feedback_max_tokens,
        "feedback_temperature": role.feedback_temperature,
    }
    out = _compact(base)
    out.update(role.extras or {})
    return out


def _api_to_dict(api: ApiConfig) -> Dict[str, Any]:
    def prov_to_dict(p: Optional[ApiProviderConfig]):
        if not p:
            return None
        out = {"api_key_env": p.api_key_env}
        out.update(p.extras or {})
        return _compact(out)

    data: Dict[str, Any] = {}
    if api.openai:
        data["openai"] = prov_to_dict(api.openai)
    if api.groq:
        data["groq"] = prov_to_dict(api.groq)
    if api.anthropic:
        data["anthropic"] = prov_to_dict(api.anthropic)
    data.update(api.extras or {})
    return data


def to_legacy_dict(cfg: RootConfig) -> Dict[str, Any]:
    """Convert typed RootConfig to the legacy nested dict layout."""
    model_dict = {
        "expert": _role_to_dict(cfg.model.expert),
        "mediator": _role_to_dict(cfg.model.mediator),
    }
    model_dict.update(cfg.model.extras or {})

    out: Dict[str, Any] = {
        "experiment": _compact(
            {
                "seed": cfg.experiment.seed,
                "output_dir": cfg.experiment.output_dir,
                "initial_forecasts_dir": cfg.experiment.initial_forecasts_dir,
            }
        )
    }
    # reuse_initial_forecasts
    rif = cfg.experiment.reuse_initial_forecasts
    out["experiment"]["reuse_initial_forecasts"] = _compact(
        {
            "enabled": rif.enabled,
            "source_dir": rif.source_dir,
            "with_examples": rif.with_examples,
            **(rif.extras or {}),
        }
    )

    out["model"] = model_dict
    out["delphi"] = cfg.delphi or {}
    out["data"] = {
        "resolution_date": cfg.data.resolution_date,
        "sampling": {
            "method": cfg.data.sampling.method,
            **(cfg.data.sampling.extras or {}),
        },
        "filters": cfg.data.filters or {},
        **(cfg.data.extras or {}),
    }
    out["processing"] = {
        "skip_existing": cfg.processing.skip_existing,
        **(cfg.processing.extras or {}),
    }
    out["api"] = _api_to_dict(cfg.api)
    out["output"] = {
        "file_pattern": cfg.output.file_pattern,
        "save": {
            "conversation_histories": cfg.output.save.conversation_histories,
            "example_pairs": cfg.output.save.example_pairs,
            **(cfg.output.save.extras or {}),
        },
        **(cfg.output.extras or {}),
    }
    out["initial_forecasts"] = cfg.initial_forecasts or {}
    out["debug"] = cfg.debug or {}
    out.update(cfg.extras or {})
    return out
