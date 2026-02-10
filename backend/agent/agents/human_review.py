from __future__ import annotations

import asyncio
from typing import Any, Dict

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from backend.agent.state import GlobalState
from backend.agent.memory import _get_mem_store


async def human_review_node(state: GlobalState, config: RunnableConfig) -> Dict[str, Any]:
    """
    只处理 stage=need_human 的 HITL：
      - accept/edit -> stage=dispatch_ready
      - reject -> stage=design（回到 designer 的“澄清/重设计”入口）
    并在此处写 mem0（best-effort，不影响主链路）。
    """
    stage = state.get("stage")
    if stage != "need_human":
        return {}

    payload = {
        "title": "Human Review Required",
        "stage": stage,
        "user_intent": state.get("user_intent"),
        "experiments_plan": state.get("experiments_plan"),
        "note": "请返回 {action: 'accept'|'edit'|'reject', feedback?: str, experiments?: list}. edit 时 experiments 为修改后的最终版本。",
    }

    resp = interrupt(payload) or {}
    if not isinstance(resp, dict):
        resp = {"action": "reject", "feedback": "invalid response type"}

    action = str(resp.get("action") or "").strip().lower()
    feedback = str(resp.get("feedback") or "").strip()

    if action not in ("accept", "edit", "reject"):
        action = "reject"
        feedback = feedback or "invalid action"

    updates: Dict[str, Any] = {}

    # edit 允许直接带最终 experiments
    if "experiments" in resp:
        updates["experiments"] = resp["experiments"]
    if feedback:
        updates["human_feedback"] = feedback

    # mem0 record (best-effort)
    store = _get_mem_store(config)
    if store is not None:
        try:
            await asyncio.to_thread(
                store.record_from_human_review,
                state=dict(state),
                action=action,
                human_feedback=feedback,
                experiments_plan=state.get("experiments_plan"),
                experiments=updates.get("experiments"),
            )
        except Exception as e:
            updates["messages"] = [AIMessage(content=f"[mem0][error] record failed: {e}")]

    # route
    if action in ("accept", "edit"):
        updates["stage"] = "dispatch_ready"
        updates.setdefault("messages", [])
        updates["messages"].append(AIMessage(content=f"[HITL][{action.upper()}] proceed to dispatch"))
        return updates

    # reject：回到 designer 第一阶段（澄清/重设计）
    updates["stage"] = "design"
    updates["last_rejection"] = {
        "action": "reject",
        "feedback": feedback,
        "experiments_plan": state.get("experiments_plan"),
    }
    updates.setdefault("messages", [])
    updates["messages"].append(AIMessage(content=f"[HITL][REJECT] {feedback} -> back to redesign"))
    return updates
