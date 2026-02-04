from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


from state import GlobalState
from agents.designer import designer_agent
from agents.exam_worker import exam_worker
from agents.analyst import analyst_agent
from agents.human_review import human_review_node


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





def build_graph():
    g = StateGraph(GlobalState)

    g.add_node("designer", designer_agent)
    g.add_node("human_review", human_review_node)
    g.add_node("exam_worker", exam_worker)
    g.add_node("analyst", analyst_agent)

    # fan-in：worker 完成后回 designer 聚合结果
    g.add_edge("exam_worker", "designer")

    # 人审后回 designer（批准 -> dispatch；不批准 -> redesign）
    g.add_edge("human_review", "designer")

    g.set_entry_point("designer")

    g.add_conditional_edges("designer", _route_after_designer)
    g.add_conditional_edges("analyst", _route_after_analyst)

    checkpointer = MemorySaver()
    graph = g.compile(checkpointer=checkpointer, interrupt_before=None)

    return graph
