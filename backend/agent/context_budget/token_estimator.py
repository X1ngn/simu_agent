from __future__ import annotations
from typing import List
from langchain_core.messages import BaseMessage

def rough_tokens(text: str) -> int:
    # 粗估：宁可偏大
    return max(1, len(text) // 3)

def estimate_messages_tokens(messages: List[BaseMessage]) -> int:
    # 每条 message 有一些开销
    overhead = 10 * len(messages)
    return overhead + sum(rough_tokens(getattr(m, "content", "") or "") for m in messages)

def estimate_json_tokens(obj: object) -> int:
    import json
    s = json.dumps(obj, ensure_ascii=False)
    return rough_tokens(s) + 20
