from typing import Any, Dict, Optional

from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from backend.agent.memory import _get_mem_store
from backend.agent.state import GlobalState


def human_review_node(state: GlobalState, config: RunnableConfig) -> Dict[str, Any]:
    """
    HITL：人类确认/修改/拒绝实验设计
    - accept/edit/reject 时写入 mem0 记忆（best-effort）
    """
    review_payload = {
        "title": "Human Review Required",
        "stage": state.get("stage"),
        "experiments_plan": state.get("experiments_plan"),
        "note": "请返回 {action: str, feedback: str, experiments: ...}；edit 时 experiments 为你修改后的正确结果",
    }

    resp = interrupt(review_payload) or {}

    action = resp.get("action")
    feedback = resp.get("feedback") or ""

    updates: Dict[str, Any] = {}

    if "experiments" in resp:
        updates["experiments"] = resp["experiments"]
    updates["human_feedback"] = feedback

    if not action:
        raise ValueError("Invalid human action")

    # --- write memory (DI from config) ---
    store = _get_mem_store(config)
    if store is None:
        # 这里建议直接 raise，避免“以为写了其实没写”
        raise RuntimeError("mem_store missing in RunnableConfig.configurable.mem_store")

    try:
        store.record_from_human_review(
            state=dict(state),
            action=str(action),
            human_feedback=str(feedback),
            experiments_plan=state.get("experiments_plan"),
            experiments=resp.get("experiments"),
        )
    except Exception as e:
        if bool(state.get("debug", False)):
            print(f"[human_review][mem0] write failed: {e}")

    # --- stage transition ---
    if action in ("accept", "edit"):
        updates["stage"] = "dispatch_ready"
    elif action == "reject":
        updates["stage"] = "done"
    else:
        updates["stage"] = "done"

    return updates
