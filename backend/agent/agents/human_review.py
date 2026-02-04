

from typing import Any, Dict
from backend.agent.state import GlobalState
from langgraph.types import interrupt

def human_review_node(state: GlobalState) -> Dict[str, Any]:
    """
    人在回路节点：
    - 暂停图执行，把 designer 的“待分发方案”交给人类确认/修改
    - 人类批准后：把 stage 置为 dispatch，回到 designer 让其 Send 分发 worker
    - 人类不批准：把 stage 置为 need_redesign，回到 designer 重新设计
    """
    # 你可以把这里的字段名改成你 GlobalState / designer_agent 实际产出的字段
    review_payload = {
        "title": "Human Review Required",
        "stage": state.get("stage"),
        "experiments_plan": state.get("experiments_plan"),
        "note": "请返回 {action: str, feedback: str, experiment: ...,}",
    }

    # interrupt 会让图暂停；resume 时，interrupt() 会“返回”你传入的 resume 数据
    resp = interrupt(review_payload) or {}

    action = resp.get("action")
    feedback = resp.get("feedback")

    updates: Dict[str, Any] = {}

    # 人类可以直接改 plan / dispatch_spec（按需）
    if "experiments" in resp:
        updates["experiments"] = resp["experiments"]
    if feedback is not None:
        updates["human_feedback"] = feedback

    if not action:
        raise "Invalid human action"

    if action == "accept" or action == "edit":
        # ✅ 批准后回到 designer，让 designer 在 dispatch 阶段真正执行 Send 分发
        # TODO：edit时记录正负样本
        updates["stage"] = "dispatch_ready"
    else:
        # ❌ 不批准则直接推出
        # TODO：记录负样本
        updates["stage"] = "done"

    return updates