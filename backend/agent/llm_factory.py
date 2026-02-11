from __future__ import annotations

from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .config import ProviderConfig, get_api_key
from .chat_adapters import LLMAsChatModel

from pathlib import Path

from backend.agent.config import load_app_config  # 你放哪都行

# ✅ token usage callback（可选）
try:
    from backend.agent.llm_usage import TokenUsageCallback  # type: ignore
except Exception:  # noqa: BLE001
    TokenUsageCallback = None  # type: ignore


def create_chat_model(cfg: ProviderConfig) -> BaseChatModel:
    if cfg.kind == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=get_api_key(),
            base_url=cfg.base_url,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    if cfg.kind == "openai_compatible":
        # vLLM OpenAI-compatible server, or any OpenAI-compatible endpoint
        from langchain_openai import ChatOpenAI

        if not cfg.base_url:
            raise ValueError("openai_compatible requires base_url")

        return ChatOpenAI(
            api_key=get_api_key(),  # 对部分本地 server 可写任意值，但这里统一从 .env 取
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            base_url=cfg.base_url,
        )

    if cfg.kind == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg.model,
            temperature=cfg.temperature,
        )

    if cfg.kind == "vllm_community":
        # Community wrapper: langchain_community.llms.VLLM (LLM, not chat)
        from langchain_community.llms import VLLM

        llm = VLLM(
            model=cfg.model,
            tensor_parallel_size=cfg.tensor_parallel_size or 1,
            gpu_memory_utilization=cfg.gpu_memory_utilization or 0.90,
            max_new_tokens=cfg.max_new_tokens or cfg.max_tokens or 1024,
            temperature=cfg.temperature,
        )
        return LLMAsChatModel(llm=llm, model_name=cfg.model)

    raise ValueError(f"Unknown LLM kind: {cfg.kind}")


_LLM_CACHE: dict[str, BaseChatModel] = {}


def build_agent_llms(providers: Dict[str, ProviderConfig]) -> Dict[str, BaseChatModel]:
    return {name: create_chat_model(cfg) for name, cfg in providers.items()}


def _attach_usage_callback(llm: BaseChatModel, agent_name: str) -> BaseChatModel:
    """
    给 LLM 绑定默认 callbacks（无需每次 invoke/ainvoke 传 callbacks）。
    - 使用 Runnable.with_config 合并默认 config
    - 若 llm_usage 未安装/导入失败，则原样返回
    """
    if TokenUsageCallback is None:
        return llm

    try:
        cb = TokenUsageCallback(label=agent_name)  # type: ignore
        # ✅ 关键：给这个 runnable 增加默认 callbacks
        return llm.with_config({"callbacks": [cb]})
    except Exception:
        # 不要让统计影响主流程
        return llm


def get_llm_for_agent(agent_name: str) -> BaseChatModel:
    """
    进程级缓存：第一次创建，后续复用
    """
    if agent_name in _LLM_CACHE:
        return _LLM_CACHE[agent_name]

    # 这里 path 你按自己项目实际位置改
    llm_cfg_path = Path(__file__).resolve().parents[2] / "backend" / "agent" / "configs" / "llm_config.yaml"
    agent_cfg_path = Path(__file__).resolve().parents[2] / "backend" / "agent" / "configs" / "agent_config.yaml"
    app_cfg = load_app_config(str(llm_cfg_path), str(agent_cfg_path))

    agent_cfg = app_cfg.agents[agent_name]
    provider_cfg = app_cfg.llm_providers[agent_cfg.provider_ref]

    llm = create_chat_model(provider_cfg)

    # ✅ 绑定 token usage callback（默认启用）
    llm = _attach_usage_callback(llm, agent_name)

    _LLM_CACHE[agent_name] = llm
    return llm
