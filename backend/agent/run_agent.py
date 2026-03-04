from __future__ import annotations

import os
import json
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, Callable, List

from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage

from backend.agent.memory import Mem0MilvusMemoryStore
from backend.agent.graph import build_graph


Action = Literal["accept", "edit", "reject"]


@dataclass
class HitlDecision:
    action: Action
    edited_text: Optional[str] = None


def _extract_interrupt_payload(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract interrupt payload from LangGraph state.

    LangGraph stores interrupts as:
      state["__interrupt__"] = (Interrupt(value={...}, id="..."), ...)
    where Interrupt is a namedtuple with .value and .id attributes.
    """
    intr = state.get("__interrupt__")
    if not intr:
        return None

    # Get the first interrupt entry (tuple/list of Interrupt objects)
    if isinstance(intr, (list, tuple)) and intr:
        first = intr[0]
    else:
        first = intr

    # Unwrap Interrupt namedtuple: has .value attribute (not a plain dict)
    if hasattr(first, "value") and not isinstance(first, dict):
        val = first.value
        if isinstance(val, dict):
            return val
        # val might itself be an Interrupt (nested case from error output)
        if hasattr(val, "value") and not isinstance(val, dict):
            inner = val.value
            return inner if isinstance(inner, dict) else {"value": inner}
        return {"value": val}

    # Plain dict with "value" key (e.g. from older LangGraph versions)
    if isinstance(first, dict) and "value" in first:
        inner = first["value"]
        if hasattr(inner, "value") and not isinstance(inner, dict):
            val = inner.value
            return val if isinstance(val, dict) else {"value": val}
        return inner if isinstance(inner, dict) else {"value": inner}

    if isinstance(first, dict):
        return first

    return {"value": first}


def _default_cli_decider(payload: Dict[str, Any], state: Dict[str, Any]) -> HitlDecision:
    """Default CLI-based human review handler."""

    # ask_user interrupt from plan_agent: show question, get answer
    if payload.get("type") == "ask_user":
        question = payload.get("question", "")
        print(f"\n--- Agent Question ---")
        print(f"  {question}")
        answer = input("Your answer > ").strip()
        return HitlDecision(action="accept", edited_text=answer)

    # human_review interrupt: show plan, get accept/edit/reject
    print("\n--- Human Review Required ---")
    experiments = payload.get("experiments_plan", [])
    if experiments and isinstance(experiments, list):
        print(f"\nExperiment plan ({len(experiments)} experiments):")
        for i, exp in enumerate(experiments):
            if isinstance(exp, dict):
                eid = exp.get("exp_id", "?")
                desc = exp.get("description", "")
                print(f"  {i+1}. [{eid}] {desc}")
    else:
        try:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception:
            print(payload)

    print("\nChoose action:")
    print("  1) Accept (proceed with experiment execution)")
    print("  2) Edit (provide modified experiment plan JSON)")
    print("  3) Reject (send back to redesign)")
    choice = input("Enter 1/2/3 > ").strip()

    if choice == "1":
        return HitlDecision(action="accept")
    if choice == "2":
        edited = input("\nEnter modified experiment plan (JSON list):\n> ").strip()
        return HitlDecision(action="edit", edited_text=edited)
    return HitlDecision(action="reject")


async def handle_hitl_interrupts_async(
    app: Any,
    state: Dict[str, Any],
    config: Dict[str, Any],
    decider: Optional[Callable[[Dict[str, Any], Dict[str, Any]], HitlDecision]] = None,
) -> Dict[str, Any]:
    """Async HITL handler: process all interrupts until graph no longer pauses."""
    decider = decider or _default_cli_decider

    while True:
        payload = _extract_interrupt_payload(state)
        if payload is None:
            return state

        decision = decider(payload, state)

        # ask_user interrupt: resume with the answer text
        if payload.get("type") == "ask_user":
            resume_value = decision.edited_text or ""
            state = await app.ainvoke(Command(resume=resume_value), config=config)
            continue

        # human_review interrupt
        if decision.action == "reject":
            resume_value = {"action": "reject", "feedback": "user rejected design"}
            state = await app.ainvoke(Command(resume=resume_value), config=config)
            continue

        if decision.action == "accept":
            resume_value = {"action": "accept"}
            state = await app.ainvoke(Command(resume=resume_value), config=config)
            continue

        if decision.action == "edit":
            edited_text = (decision.edited_text or "").strip()
            experiments = None
            if edited_text:
                try:
                    experiments = json.loads(edited_text)
                except json.JSONDecodeError:
                    print("ERROR: edited_text must be valid JSON; falling back to original plan")
            resume_value = {
                "action": "edit",
                "feedback": "user edited design",
            }
            if experiments is not None:
                resume_value["experiments"] = experiments
            state = await app.ainvoke(Command(resume=resume_value), config=config)
            continue


async def amain() -> None:
    debug = True
    user_intent = "固定模型与硬件，扫 global_batch_size 对吞吐与显存峰值的影响，并找出失败边界。"

    app = build_graph()

    init_state: Dict[str, Any] = {
        "debug": debug,
        "user_intent": user_intent,
        "messages": [HumanMessage(content=user_intent)],
    }

    mem_store = Mem0MilvusMemoryStore()

    config = {
        "configurable": {
            "thread_id": "demo_thread",
            "mem_store": mem_store,
        }
    }

    # Run first invocation (enters plan_agent)
    state = await app.ainvoke(init_state, config=config)

    # Handle all interrupts (ask_user + human_review + exec phase)
    state = await handle_hitl_interrupts_async(app, state, config=config)

    print("\n--- FINAL STATE ---")
    print(f"finished: {state.get('finished')}")

    # Print last AI message as report
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            print(f"\nFinal AI message:\n{msg.content}")
            break

    report = state.get("analyst_report", {}) or {}
    if report:
        print("\n--- ANALYST REPORT ---")
        for k, v in report.items():
            print(f"{k}: {v}")


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "data", "runs")
    os.makedirs(base, exist_ok=True)
    asyncio.run(amain())


if __name__ == "__main__":
    main()
