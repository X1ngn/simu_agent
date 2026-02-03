from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import os
import yaml
from dotenv import load_dotenv


LLMKind = Literal["openai", "ollama", "openai_compatible", "vllm_community"]


@dataclass(frozen=True)
class ProviderConfig:
    kind: LLMKind
    model: str
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None  # for openai_compatible

    # vllm community extras
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    max_new_tokens: Optional[int] = None


@dataclass(frozen=True)
class AgentConfig:
    provider_ref: str
    system_prompt: str
    response_style: Dict[str, Any]


@dataclass(frozen=True)
class AppConfig:
    llm_providers: Dict[str, ProviderConfig]
    agents: Dict[str, AgentConfig]


def load_app_config(llm_path: str, agent_path: str) -> AppConfig:

    with open(llm_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    providers: Dict[str, ProviderConfig] = {}
    for name, cfg in raw.get("llm_providers", {}).items():
        providers[name] = ProviderConfig(**cfg)

    with open(agent_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    agents: Dict[str, AgentConfig] = {}
    for name, cfg in raw.get("agents", {}).items():
        agents[name] = AgentConfig(**cfg)

    return AppConfig(llm_providers=providers, agents=agents)


def get_api_key() -> str:
    # You asked "API_KEY from .env" → compatible with both OPENAI_API_KEY and API_KEY
    load_dotenv(override=True)
    key = os.getenv("API_KEY")
    if not key:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY or API_KEY in your .env"
        )
    return key
