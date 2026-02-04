from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, Callable

from graph import build_graph

from langgraph.types import Command


# =========================
# HITL 高内聚处理模块
# =========================

Action = Literal["accept", "edit", "reject"]


@dataclass
class HitlDecision:
    action: Action
    # edit 模式下：用户输入的“修改后的字符串”（当前版本直接当 user_intent 或交给解析器）
    edited_text: Optional[str] = None

def _safe_get_experiments_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    # payload 可能是 {"experiments_plan": ...} 或 {"value": {...}}
    if "experiments_plan" in payload and isinstance(payload["experiments_plan"], dict):
        return payload["experiments_plan"]
    v = payload.get("value")
    if isinstance(v, dict) and "experiments_plan" in v and isinstance(v["experiments_plan"], dict):
        return v["experiments_plan"]
    return {}


def _extract_interrupt_payload(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从 LangGraph 返回的 state 中提取 interrupt payload。
    LangGraph 的 interrupt 信息通常在 state["__interrupt__"] 中，可能是 list。
    """
    intr = state.get("__interrupt__")
    if not intr:
        return None
    # 常见结构：list[{"value": payload, ...}]
    if isinstance(intr, list) and intr:
        first = intr[0]
        if isinstance(first, dict) and "value" in first:
            val = first["value"]
            return val if isinstance(val, dict) else {"value": val}
        # 兜底
        return first if isinstance(first, dict) else {"value": first}
    # 兜底
    return intr if isinstance(intr, dict) else {"value": intr}


def _default_cli_decider(payload: Dict[str, Any], state: Dict[str, Any]) -> HitlDecision:
    """
    CLI 交互版决策器（后续你改成 API：直接替换这个函数即可）。
    """
    print("\n=== HITL: Human Review Required ===")
    # 尽量不耦合 payload 字段名：直接把 payload 打出来
    try:
        import json
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        print(payload)

    print("\n请选择操作：")
    print("  1) 接受（继续执行仿真实验）")
    print("  2) 编辑（输入修改后的实验设计字符串，后台解析/重跑）")
    print("  3) 拒绝（终止本次调用）")

    choice = input("输入 1/2/3 > ").strip()

    if choice == "1":
        return HitlDecision(action="accept")

    if choice == "2":
        edited = input("\n请输入修改后的实验设计字符串（当前版本：直接作为新的 user_intent 触发重新设计）\n> ").strip()
        return HitlDecision(action="edit", edited_text=edited)

    return HitlDecision(action="reject")


def handle_hitl_interrupts(
    app: Any,
    state: Dict[str, Any],
    config: Dict[str, Any],
    decider: Optional[Callable[[Dict[str, Any], Dict[str, Any]], HitlDecision]] = None,
) -> Dict[str, Any]:
    """
    处理所有 interrupt，直到图不再暂停为止。

    设计目标：
    - 高内聚：interrupt 提取、用户决策、resume 调用都在这里
    - 低耦合：decider 可替换（CLI / Web / RPC / 前端）
    - 三种动作：
      - accept：resume approved=True，让 human_review 节点设置 stage=dispatch_ready，再回 designer 分发
      - edit：把用户输入写回 state（user_intent / stage），resume approved=False + feedback，促使重设计
      - reject：直接返回当前 state，不再继续 invoke（上层据此结束）
    """
    decider = decider or _default_cli_decider

    while True:
        payload = _extract_interrupt_payload(state)
        if payload is None:
            return state  # 没有中断，正常结束

        decision = decider(payload, state)

        if decision.action == "reject":
            # 直接结束本次调用：不再 resume
            resume_value = {"action": "reject", "feedback": "user reject design; please redesign based on new intent"}
            state = app.invoke(Command(resume=resume_value), config=config)
            continue

        if decision.action == "accept":
            # 告诉 human_review：批准
            resume_value = {"action": "accept"}
            state = app.invoke(Command(resume=resume_value), config=config)
            continue

        if decision.action == "edit":
            # 当前版本：用户输入一段字符串 -> 后台“先当作新的 user_intent”
            # 你后续可以在这里接解析器：把 edited_text 解析成 experiments结构
            edited_text = (decision.edited_text or "").strip()
            experiments = _safe_get_experiments_from_payload(payload)

            if edited_text:
                # 写回 state，触发重新设计
                # 约定：回到 designer 直接使用这个 experiments 进行分发
                try:
                    experiments = json.loads(edited_text)
                except json.decoder.JSONDecodeError:
                    print("ERROR: text should be a json")

            # 让 human_review 走“不批准”分支，回 designer 重新规划（与你 human_review_node 的逻辑对齐）
            resume_value = {
                "action": "edit",
                "feedback": "user edited design; please redesign based on new intent",
                "experiments": experiments
            }
            state = app.invoke(Command(resume=resume_value), config=config)
            continue


# =========================
# 原 main 主流程（最小改动）
# =========================
def main():
    # debug 开关
    debug = True

    # 你也可以把 user_intent 换成从 CLI / Web / API 输入
    user_intent = "固定模型与硬件，扫 global_batch_size 对吞吐与显存峰值的影响，并找出失败边界。"

    app = build_graph()

    init_state: Dict[str, Any] = {
        "debug": debug,
        "stage": "init",
        "user_intent": user_intent,
        "retries": 0,
        "max_retries": 1,
    }

    # thread_id 用于 checkpoint / 多会话并行
    config = {"configurable": {"thread_id": "demo_thread"}}

    # 1) 先正常 invoke 一次（保持你原有逻辑）
    final_state = app.invoke(init_state, config=config)

    # 2) 如果中断了，在不破坏主流程的前提下，进入 HITL 处理器（可替换为 API）
    final_state = handle_hitl_interrupts(app, final_state, config=config)

    print("\n=== FINAL REPORT ===")
    report = final_state.get("analyst_report", {}) or {}
    if not report:
        # 拒绝时给一个最小可读输出（不影响你原 report 打印结构）
        print("ok: False")
        print("message: user rejected the experiment design; run aborted")
    else:
        for k, v in report.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    # 确保 data/runs 存在
    base = os.path.join(os.path.dirname(__file__), "data", "runs")
    os.makedirs(base, exist_ok=True)
    main()
