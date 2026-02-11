from __future__ import annotations
from typing import Any, Dict, List, TypedDict

class ConfirmedItem(TypedDict):
    k: str
    v: Any

class PendingItem(TypedDict, total=False):
    q: str
    need: List[str]
    opts: List[Any]

class RollingSummary(TypedDict):
    v: int
    intent: str
    config: Dict[str, Any]
    confirmed: List[ConfirmedItem]
    pending: List[PendingItem]

def build_summary_from_state(state: Dict[str, Any]) -> RollingSummary:
    # intent
    intent = str(state.get("user_intent") or "")

    # current config：优先 experiments（最终）> experiments_plan（计划）> clarify_answers（如果你想也可以）
    cfg = {}
    if isinstance(state.get("experiments"), list) and state.get("experiments"):
        cfg = {"experiments": state.get("experiments")}
    elif isinstance(state.get("experiments_plan"), list) and state.get("experiments_plan"):
        cfg = {"experiments_plan": state.get("experiments_plan")}
    else:
        cfg = {}

    # confirmed：直接用 clarify_answers（你已经结构化了）
    confirmed: List[ConfirmedItem] = []
    ans = state.get("clarify_answers") or {}
    if isinstance(ans, dict):
        for k, v in ans.items():
            confirmed.append({"k": str(k), "v": v})

    # pending：直接用 clarify_questions（你已经结构化了）
    pending: List[PendingItem] = []
    qs = state.get("clarify_questions") or []
    if isinstance(qs, list):
        for q in qs:
            if not isinstance(q, dict):
                continue
            key = str(q.get("key") or "").strip()
            question = str(q.get("question") or "").strip()
            if not key or not question:
                continue
            item: PendingItem = {"q": question, "need": [key]}
            if q.get("type") == "choice" and isinstance(q.get("choices"), list):
                item["opts"] = q.get("choices")  # type: ignore[assignment]
            pending.append(item)

    return {"v": 1, "intent": intent, "config": cfg, "confirmed": confirmed, "pending": pending}
