from __future__ import annotations

from typing import Any, Dict, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import GlobalState
from agents.designer import designer_agent
from agents.exam_worker import exam_worker
from agents.analyst import analyst_agent


# reducer：用于聚合 exam_worker 的输出
def _merge_exam_results(old: List[Dict[str, Any]] | None, new: List[Dict[str, Any]] | None):
    old = old or []
    new = new or []
    return old + new


def _route_after_designer(state: GlobalState) -> str:
    stage = state.get("stage", "init")

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
    return "designer"


def build_graph():
    g = StateGraph(GlobalState)

    g.add_node("designer", designer_agent)
    g.add_node("exam_worker", exam_worker)
    g.add_node("analyst", analyst_agent)

    # 关键：为 exam_results 字段配置 reducer（map-reduce 聚合 worker 输出）
    g.add_edge("exam_worker", "designer")

    g.set_entry_point("designer")

    g.add_conditional_edges("designer", _route_after_designer)
    g.add_conditional_edges("analyst", _route_after_analyst)

    # 可选：加 checkpoint，方便后续扩展 memory / RAG / 断点续跑
    checkpointer = MemorySaver()
    graph = g.compile(checkpointer=checkpointer, interrupt_before=None)
    # print(graph.get_graph().draw_mermaid())

    return graph
