from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import interrupt, Command

from backend.agent.state import AgentState, ExperimentSpec
from backend.agent.llm_factory import get_llm_for_agent
from backend.agent.prompts.plan_agent import PLAN_SYSTEM_PROMPT
from backend.agent.prompts.exec_agent import EXEC_SYSTEM_PROMPT
from backend.agent.tools.agent_tools import (
    search_memory,
    ask_user,
    submit_experiment_plan,
    run_simulation_tool,
    run_batch,
    analyze_results,
    finish,
)


PLAN_TOOLS = [search_memory, ask_user, submit_experiment_plan]
EXEC_TOOLS = [run_simulation_tool, run_batch, analyze_results, finish]


async def plan_agent_node(state: AgentState, config):
    """Phase 1: Plan Agent - designs experiment plan via tool calling."""
    llm = get_llm_for_agent("designer").bind_tools(PLAN_TOOLS)
    messages = list(state.get("messages") or [])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=PLAN_SYSTEM_PROMPT)] + messages
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Argument coercion helper – some LLMs serialise complex arguments as JSON
# strings instead of native dicts/lists; we parse them back before Pydantic
# validation.
# ---------------------------------------------------------------------------

def _coerce_tool_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-process tool call arguments: if a value is a string that looks like
    a JSON list or object, parse it into a native Python structure."""
    coerced = {}
    for k, v in args.items():
        if isinstance(v, str):
            stripped = v.strip()
            if (stripped.startswith("[") and stripped.endswith("]")) or \
               (stripped.startswith("{") and stripped.endswith("}")):
                try:
                    v = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    pass
        coerced[k] = v
    return coerced


# Custom plan_tools node: wraps ToolNode but also extracts experiments_plan
# from submit_experiment_plan into state (SPEC 3.6)
_plan_tool_map = {t.name: t for t in PLAN_TOOLS}


async def plan_tool_node(state: AgentState, config):
    """Execute plan-phase tool calls. Extracts experiments_plan on submit."""
    messages = state.get("messages") or []
    if not messages:
        return {}

    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}

    results = []
    updates: Dict[str, Any] = {}

    for tc in last.tool_calls:
        tool_fn = _plan_tool_map.get(tc["name"])
        if tool_fn is None:
            results.append(ToolMessage(
                content=f"Unknown tool: {tc['name']}",
                tool_call_id=tc["id"],
            ))
            continue

        coerced_args = _coerce_tool_args(tc["args"])
        result = await tool_fn.ainvoke(coerced_args)
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        # Extract experiments_plan when submit is called
        if tc["name"] == "submit_experiment_plan":
            updates["experiments_plan"] = tc["args"].get("experiments", [])

    updates["messages"] = results
    return updates


async def human_review_node(state: AgentState, config):
    """HITL gate: interrupt for human approval of experiment plan."""
    messages = state.get("messages") or []
    experiments_plan = state.get("experiments_plan") or []
    if not experiments_plan:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "submit_experiment_plan":
                        experiments_plan = tc["args"].get("experiments", [])
                        break
                if experiments_plan:
                    break
    payload = {
        "title": "Human Review Required",
        "user_intent": state.get("user_intent"),
        "experiments_plan": experiments_plan,
        "note": "Return {action: accept|edit|reject, feedback?: str, experiments?: list}",
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
    if action in ("accept", "edit"):
        final_experiments = resp.get("experiments") or experiments_plan
        updates["experiments_plan"] = final_experiments
        updates["human_approved"] = True
        updates["messages"] = [
            AIMessage(content=f"[HITL][{action.upper()}] plan approved. {len(final_experiments)} experiments.")
        ]
    else:
        updates["human_approved"] = False
        updates["messages"] = [
            HumanMessage(content=f"[REJECT] {feedback}. Please redesign the experiment plan.")
        ]
    return updates


async def exec_agent_node(state: AgentState, config):
    """Phase 2: Exec Agent - executes experiments and analyzes results.

    When the finish tool has already been called (finished=True), we invoke
    the LLM **without** tool bindings so it produces a plain-text summary
    instead of issuing more tool calls.
    """
    already_finished = state.get("finished", False)

    if already_finished:
        # No tools – force a plain-text summary
        llm = get_llm_for_agent("designer")
    else:
        llm = get_llm_for_agent("designer").bind_tools(EXEC_TOOLS)

    messages = list(state.get("messages") or [])
    has_exec_sys = any(
        isinstance(m, SystemMessage) and "Exec Agent" in (m.content or "")
        for m in messages
    )
    if not has_exec_sys:
        experiments_plan = state.get("experiments_plan") or []
        run_id = state.get("run_id") or uuid.uuid4().hex[:8]
        context_msg = (
            f"Experiment plan approved. run_id={run_id}\n"
            f"Experiments:\n{json.dumps(experiments_plan, ensure_ascii=False, indent=2)}\n"
            f"Please use run_batch to execute, then analyze_results, then finish."
        )
        # Persist the system message and context into state so that subsequent
        # visits to this node see has_exec_sys=True and don't re-inject them,
        # which previously caused an infinite loop of growing message lists.
        sys_msg = SystemMessage(content=EXEC_SYSTEM_PROMPT)
        ctx_msg = HumanMessage(content=context_msg)
        messages = messages + [sys_msg, ctx_msg]
        response = await llm.ainvoke(
            [sys_msg] + list(state.get("messages") or []) + [ctx_msg]
        )
        return {"messages": [sys_msg, ctx_msg, response], "run_id": run_id}
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


_exec_tool_map = {t.name: t for t in EXEC_TOOLS}


async def exec_tool_node(state: AgentState, config):
    """Execute exec-phase tool calls.  Sets finished=True when finish is called."""
    messages = state.get("messages") or []
    if not messages:
        return {}

    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}

    results = []
    updates: Dict[str, Any] = {}

    for tc in last.tool_calls:
        tool_fn = _exec_tool_map.get(tc["name"])
        if tool_fn is None:
            results.append(ToolMessage(
                content=f"Unknown tool: {tc['name']}",
                tool_call_id=tc["id"],
            ))
            continue

        coerced_args = _coerce_tool_args(tc["args"])
        result = await tool_fn.ainvoke(coerced_args)
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        if tc["name"] == "finish":
            updates["finished"] = True

    updates["messages"] = results
    return updates


def _route_after_plan_agent(state: AgentState) -> str:
    """Route after plan_agent: tool call -> plan_tools, else -> END."""
    messages = state.get("messages") or []
    if not messages:
        return END
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "plan_tools"
    return END


def _route_after_plan_tools(state: AgentState) -> str:
    """After plan tools: submit -> human_review, else -> plan_agent."""
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "submit_experiment_plan":
                    return "human_review"
            break
    return "plan_agent"


def _route_after_human_review(state: AgentState) -> str:
    """After human review: approved -> exec_agent, rejected -> plan_agent."""
    if state.get("human_approved"):
        return "exec_agent"
    return "plan_agent"


def _route_after_exec_agent(state: AgentState) -> str:
    """Route after exec_agent: tool call -> exec_tools, else -> END."""
    messages = state.get("messages") or []
    if not messages:
        return END
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "exec_tools"
    return END


def _route_after_exec_tools(state: AgentState) -> str:
    """After exec tools: always route back to exec_agent.

    Even after the finish tool is called, we go back to exec_agent so it can
    produce a final AIMessage summarising the results.  exec_agent will then
    return *without* tool_calls, and _route_after_exec_agent will send the
    graph to END.
    """
    return "exec_agent"


def build_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    interrupt_before: Optional[Sequence[str]] = None,
):
    g = StateGraph(AgentState)
    g.add_node("plan_agent", plan_agent_node)
    g.add_node("plan_tools", plan_tool_node)
    g.add_node("human_review", human_review_node)
    g.add_node("exec_agent", exec_agent_node)
    g.add_node("exec_tools", exec_tool_node)
    g.set_entry_point("plan_agent")
    g.add_conditional_edges("plan_agent", _route_after_plan_agent)
    g.add_conditional_edges("plan_tools", _route_after_plan_tools)
    g.add_conditional_edges("human_review", _route_after_human_review)
    g.add_conditional_edges("exec_agent", _route_after_exec_agent)
    g.add_conditional_edges("exec_tools", _route_after_exec_tools)
    if checkpointer is None:
        checkpointer = MemorySaver()
    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )
