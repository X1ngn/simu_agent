from __future__ import annotations

from typing import Optional, Sequence
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from backend.agent.state import GlobalState
from backend.agent.agents.designer import designer_agent
from backend.agent.agents.exam_worker import exam_worker
from backend.agent.agents.analyst import analyst_agent
from backend.agent.agents.human_review import human_review_node


def _route_after_designer(state: GlobalState) -> str:
    stage = state.get("stage", "init")

    # ✅ 关键：等待用户补充信息时，直接停
    if stage == "await_user":
        return END

    if stage == "need_human":
        return "human_review"

    # 你注释写的是 dispatch，但你系统里实际用的是 dispatch_ready / collect
    # 这里只保留 done/分析的终止逻辑即可
    if stage == "analyze":
        return "analyst"

    if stage == "done":
        return END

    # worker fan-in：exam_worker -> designer
    if stage == "collect":
        return "designer"

    # 需要重设计：回 designer
    if stage == "need_redesign":
        return "designer"

    # 默认继续走 designer（init/design/dispatch_ready 都会落到这里）
    return "designer"


def _route_after_analyst(state: GlobalState) -> str:
    report = state.get("analyst_report", {}) or {}
    ok = bool(report.get("ok", False))
    if ok:
        return END

    # ✅ analyst 发现异常，回 designer 走重设计流程
    # 这里直接回 designer，并由 update 把 stage 设置为 need_redesign（在 analyst_agent 或这里都行）
    return "designer"


def build_graph(
    designer=designer_agent,
    human_review=human_review_node,
    worker=exam_worker,
    analyst=analyst_agent,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    interrupt_before: Optional[Sequence[str]] = None,
):
    g = StateGraph(GlobalState)

    g.add_node("designer", designer)
    g.add_node("human_review", human_review)
    g.add_node("exam_worker", worker)
    g.add_node("analyst", analyst)

    g.add_edge("exam_worker", "designer")
    g.add_edge("human_review", "designer")

    g.set_entry_point("designer")
    g.add_conditional_edges("designer", _route_after_designer)
    g.add_conditional_edges("analyst", _route_after_analyst)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )
