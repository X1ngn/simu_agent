from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage

from .schemas import build_summary_from_state
from .token_estimator import estimate_messages_tokens, estimate_json_tokens
from .logger_sqlite import SQLiteCompressionLogger, CompressionEvent, _msg_to_dict

SUMMARY_SYSTEM_TAG = "ROLLING_SUMMARY_JSON="

def _upsert_summary_system_message(messages: List[BaseMessage], summary: Dict[str, Any]) -> List[BaseMessage]:
    content = SUMMARY_SYSTEM_TAG + __import__("json").dumps(summary, ensure_ascii=False)
    # 替换旧的 summary system message；没有就插到最前
    out: List[BaseMessage] = []
    replaced = False
    for m in messages:
        if getattr(m, "type", "") == "system" and (getattr(m, "content", "") or "").startswith(SUMMARY_SYSTEM_TAG):
            out.append(SystemMessage(content=content))
            replaced = True
        else:
            out.append(m)
    if not replaced:
        out = [SystemMessage(content=content)] + out
    return out

def maybe_truncate_and_summarize(
    *,
    state: Dict[str, Any],
    logger: Optional[SQLiteCompressionLogger],
    max_tokens: int,
    reserve_tokens: int = 1200,
    keep_last_n: int = 20,
    thread_id_key: str = "run_id",
) -> Tuple[List[BaseMessage], Dict[str, Any], bool]:
    """
    返回 (new_messages, rolling_summary, truncated)
    - rolling_summary 只含 intent/config/confirmed/pending
    - 超预算时删除最早 messages，只保留 last_n，并注入 summary system message
    """
    messages: List[BaseMessage] = list(state.get("messages") or [])
    thread_id = str(state.get(thread_id_key) or "unknown")

    # 构造（或更新）rolling summary：不依赖 LLM，完全从你现有结构化 state 派生
    summary = build_summary_from_state(state)
    state["rolling_summary"] = summary

    total_before = estimate_messages_tokens(messages) + estimate_json_tokens(summary)
    budget = max_tokens - reserve_tokens
    if total_before <= budget:
        messages = _upsert_summary_system_message(messages, summary)
        return messages, summary, False

    # 超预算：删到只剩最后 keep_last_n 条
    cut = max(0, len(messages) - keep_last_n)
    raw_window = messages[:cut]
    remaining = messages[cut:]

    # 注入 summary
    remaining = _upsert_summary_system_message(remaining, summary)

    total_after = estimate_messages_tokens(remaining) + estimate_json_tokens(summary)

    # 落库追溯
    if logger is not None and raw_window:
        try:
            ev = CompressionEvent(
                thread_id=thread_id,
                start_idx=0,
                end_idx=cut - 1,
                raw_messages=[_msg_to_dict(m, i) for i, m in enumerate(raw_window)],
                summary_after=summary,
                token_before=total_before,
                token_after=total_after,
            )
            logger.log(ev)
        except Exception:
            pass

    # 写回 state.messages（关键！）
    state["messages"] = remaining
    return remaining, summary, True
