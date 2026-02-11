# backend/agent/llm_usage.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class UsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TokenUsageCallback(BaseCallbackHandler):
    """
    兼容多数 provider 的 usage 返回形式：
    - OpenAI/兼容：response.llm_output["token_usage"] 或 response.usage_metadata
    - 一些实现会放在 generations[0][0].generation_info
    """

    def __init__(self, label: str = "llm") -> None:
        self.label = label
        self.last: Optional[UsageRecord] = None

    def on_llm_end(self, response, **kwargs: Any) -> None:
        usage = None

        # 1) LangChain 常见：llm_output.token_usage
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage") or llm_output.get("usage")

        # 2) LangChain v0.2+：usage_metadata（很多 chat model 会放这）
        if usage is None:
            usage_meta = getattr(response, "usage_metadata", None)
            if isinstance(usage_meta, dict):
                usage = usage_meta

        # 3) generation_info（少数 provider）
        if usage is None:
            try:
                gi = response.generations[0][0].generation_info
                if isinstance(gi, dict):
                    usage = gi.get("token_usage") or gi.get("usage")
            except Exception:
                pass

        rec = UsageRecord()
        if isinstance(usage, dict):
            # 不同 provider 字段名略有差异，做个兼容
            rec.prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            rec.completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            rec.total_tokens = int(usage.get("total_tokens") or (rec.prompt_tokens + rec.completion_tokens))

        self.last = rec
        print(
            f"[token][{self.label}] "
            f"prompt={rec.prompt_tokens} completion={rec.completion_tokens} total={rec.total_tokens}"
        )
