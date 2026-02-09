from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


from backend.agent.state import GlobalState
from backend.agent.agents.designer import designer_agent
from backend.agent.agents.exam_worker import exam_worker
from backend.agent.agents.analyst import analyst_agent
from backend.agent.agents.human_review import human_review_node


def _route_after_designer(state: GlobalState) -> str:
    stage = state.get("stage", "init")

    # ✅ 设计：designer 先产出“待确认的方案”，进入 need_human
    # 人类确认后再回到 designer，把 stage 推到 dispatch（此时 designer 才 Send 分发 worker）
    if stage == "need_human":
        return "human_review"

    # dispatch 这一轮 designer 已经用 Command(goto=[Send...]) 把 worker 发出去了
    # 不要立刻再次运行 designer，等待 worker -> designer 触发 fan-in
    if stage == "dispatch":
        return END

    if stage == "analyze":
        return "analyst"

    if stage == "done":
        return END

    # worker 返回时会把 stage 推到 collect
    if stage == "collect":
        return "designer"

    if stage == "need_redesign":
        return "designer"

    return "designer"


def _route_after_analyst(state: GlobalState) -> str:
    report = state.get("analyst_report", {}) or {}
    ok = bool(report.get("ok", False))
    if ok:
        return END
    # analyst 发现异常，回 designer 决策是否重跑/重设实验
    return "need_redesign"





# backend/agent/graph.py
from typing import Optional, Sequence
from langgraph.checkpoint.base import BaseCheckpointSaver

def build_graph(
    designer=designer_agent,
    human_review=human_review_node,
    worker=exam_worker,
    analyst=analyst_agent,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    interrupt_before: Optional[Sequence[str]] = None,   # ✅ 新增
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
        interrupt_before=interrupt_before,   # ✅ 透传
    )


