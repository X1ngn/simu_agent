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

# =========================
# HITL 高内聚处理模块
# =========================

Action = Literal["accept", "edit", "reject"]


@dataclass
class HitlDecision:
    action: Action
    edited_text: Optional[str] = None


def _safe_get_experiments_from_payload(payload: Dict[str, Any]) -> Any:
    """
    payload 可能是 {"experiments_plan": ...} 或 {"value": {...}}
    human_review_node 通常传 experiments_plan(list)；这里返回 Any 更稳。
    """
    if "experiments_plan" in payload:
        return payload["experiments_plan"]
    v = payload.get("value")
    if isinstance(v, dict) and "experiments_plan" in v:
        return v["experiments_plan"]
    return None


def _extract_interrupt_payload(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从 LangGraph 返回的 state 中提取 interrupt payload。
    interrupt 信息通常在 state["__interrupt__"] 中，可能是 list。
    """
    intr = state.get("__interrupt__")
    if not intr:
        return None

    if isinstance(intr, list) and intr:
        first = intr[0]
        if isinstance(first, dict) and "value" in first:
            val = first["value"]
            return val if isinstance(val, dict) else {"value": val}
        return first if isinstance(first, dict) else {"value": first}

    return intr if isinstance(intr, dict) else {"value": intr}


def _default_cli_decider(payload: Dict[str, Any], state: Dict[str, Any]) -> HitlDecision:
    print("\n=== HITL: Human Review Required ===")
    try:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        print(payload)

    print("\n请选择操作：")
    print("  1) 接受（继续执行仿真实验）")
    print("  2) 编辑（输入修改后的实验设计 JSON，后台直接采用）")
    print("  3) 拒绝（终止本次调用）")

    choice = input("输入 1/2/3 > ").strip()

    if choice == "1":
        return HitlDecision(action="accept")

    if choice == "2":
        edited = input(
            "\n请输入修改后的实验设计（JSON，通常是 experiments 列表或 dict），将写入 resume.experiments\n> "
        ).strip()
        return HitlDecision(action="edit", edited_text=edited)

    return HitlDecision(action="reject")


async def handle_hitl_interrupts_async(
    app: Any,
    state: Dict[str, Any],
    config: Dict[str, Any],
    decider: Optional[Callable[[Dict[str, Any], Dict[str, Any]], HitlDecision]] = None,
) -> Dict[str, Any]:
    """
    Async 版 HITL 处理器：处理所有 interrupt，直到图不再暂停为止。
    """
    decider = decider or _default_cli_decider

    while True:
        payload = _extract_interrupt_payload(state)
        if payload is None:
            return state

        decision = decider(payload, state)

        if decision.action == "reject":
            resume_value = {"action": "reject", "feedback": "user reject design; run aborted"}
            state = await app.ainvoke(Command(resume=resume_value), config=config)
            continue

        if decision.action == "accept":
            resume_value = {"action": "accept"}
            state = await app.ainvoke(Command(resume=resume_value), config=config)
            continue

        if decision.action == "edit":
            edited_text = (decision.edited_text or "").strip()
            experiments = _safe_get_experiments_from_payload(payload)

            if edited_text:
                try:
                    experiments = json.loads(edited_text)
                except json.JSONDecodeError:
                    print("ERROR: edited_text must be valid JSON; fallback to payload experiments_plan")

            resume_value = {
                "action": "edit",
                "feedback": "user edited design; please apply edited experiments",
                "experiments": experiments,
            }
            state = await app.ainvoke(Command(resume=resume_value), config=config)
            continue


# =========================
# CLI 澄清循环（await_user）
# =========================

def _get_last_ai_message_text(state: Dict[str, Any]) -> str:
    """
    尽量从 state["messages"] 中找最后一条 AIMessage。
    如果没有 messages，就兜底打印 clarify_questions。
    """
    msgs = state.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, AIMessage):
                return str(m.content or "")
            # 有些框架会把 message 序列化成 dict
            if isinstance(m, dict) and m.get("type") == "ai":
                return str(m.get("content") or "")
    # fallback: clarify_questions
    qs = state.get("clarify_questions")
    if isinstance(qs, list) and qs:
        lines = []
        for q in qs:
            if isinstance(q, dict):
                lines.append(f"- ({q.get('key')}) {q.get('question')}")
        if lines:
            return "[CLARIFY]\n" + "\n".join(lines)
    return "[CLARIFY] 需要补充信息，但未找到具体问题列表。"


def _append_user_message(state: Dict[str, Any], text: str) -> None:
    """
    把用户补充信息写入 messages，配合 add_messages reducer。
    没有 messages 字段时也兜底能跑。
    """
    if not text.strip():
        return
    msg = HumanMessage(content=text.strip())
    if "messages" in state and isinstance(state["messages"], list):
        state["messages"] = state["messages"] + [msg]  # 让 reducer 合并更自然
    else:
        state["messages"] = [msg]


async def _clarify_loop_if_needed(app: Any, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    如果 stage == await_user：打印问题，收集用户输入，写回 state，然后继续 ainvoke。
    注意：这里不使用 interrupt/resume，而是“下一轮用户输入后再跑一轮图”。
    """
    while state.get("stage") == "await_user":
        print("\n=== CLARIFICATION REQUIRED ===")
        print(_get_last_ai_message_text(state))

        print("\n请补充信息（可多行输入，空行结束）：")
        lines: List[str] = []
        while True:
            line = input("> ")
            if line.strip() == "":
                break
            lines.append(line)
        user_reply = "\n".join(lines).strip()

        if not user_reply:
            print("你没有输入任何补充信息，将保持 await_user（你可以按 Ctrl+C 退出）。")
            continue

        # 写回 messages（让 designer 下次能看到上下文）
        _append_user_message(state, user_reply)

        # 关键：把 stage 拉回 design，让 designer 再跑一次 clarify+design
        state["stage"] = "design"

        # 再推进一轮图
        state = await app.ainvoke(state, config=config)

        # 推进后如果又触发 human_review interrupt，也顺便处理掉
        state = await handle_hitl_interrupts_async(app, state, config=config)

    return state


# =========================
# main 主流程
# =========================

async def amain() -> None:
    debug = True
    user_intent = "固定模型与硬件，扫 global_batch_size 对吞吐与显存峰值的影响，并找出失败边界。"

    app = build_graph()

    init_state: Dict[str, Any] = {
        "debug": debug,
        "stage": "init",
        "user_intent": user_intent,
        "retries": 0,
        "max_retries": 1,
        # ✅ 建议把用户初始意图也放进 messages，便于 clarify LLM 看上下文
        "messages": [HumanMessage(content=user_intent)],
    }

    mem_store = Mem0MilvusMemoryStore()

    config = {
        "configurable": {
            "thread_id": "demo_thread",
            "mem_store": mem_store,
        }
    }

    # 1) 先跑一轮
    state = await app.ainvoke(init_state, config=config)

    # 2) 若触发 human_review interrupt，先处理
    state = await handle_hitl_interrupts_async(app, state, config=config)

    # 3) 若需要澄清（await_user），进入 CLI 澄清循环
    state = await _clarify_loop_if_needed(app, state, config=config)

    print("\n=== FINAL REPORT ===")
    report = state.get("analyst_report", {}) or {}
    if not report:
        print("ok: False")
        print("message: run finished without analyst_report (possibly rejected or awaiting more input)")
        print("stage:", state.get("stage"))
    else:
        for k, v in report.items():
            print(f"{k}: {v}")


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "data", "runs")
    os.makedirs(base, exist_ok=True)
    asyncio.run(amain())


if __name__ == "__main__":
    main()
