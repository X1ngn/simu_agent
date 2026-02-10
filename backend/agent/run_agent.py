from __future__ import annotations

import os
import json
import asyncio

from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, Callable

from langgraph.types import Command

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
    你的 human_review_node 传的是 experiments_plan(list)，所以这里返回 Any 更稳。
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
    """
    CLI 交互版决策器（后续你改成 API：直接替换这个函数即可）。
    """
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
    ✅ Async 版 HITL 处理器：处理所有 interrupt，直到图不再暂停为止。

    关键修复点：
    - 图里存在 async 节点（如 designer_agent async）后，resume 必须使用 app.ainvoke
    - 不能再用 app.invoke，也不能用 to_thread 包 app.invoke

    三种动作：
      - accept：resume {"action":"accept"} -> human_review_node 设置 stage=dispatch_ready -> 回 designer 分发
      - edit：resume {"action":"edit","experiments":...} -> human_review_node 同样走 dispatch_ready（你的逻辑里 accept/edit 都 dispatch_ready）
      - reject：resume {"action":"reject"} -> human_review_node 走 done（你的逻辑里 else -> done）
    """
    decider = decider or _default_cli_decider

    while True:
        payload = _extract_interrupt_payload(state)
        if payload is None:
            return state  # 没有中断，正常结束

        decision = decider(payload, state)

        if decision.action == "reject":
            # 直接退出/或者继续推进到 done（取决于你 human_review_node 如何处理 reject）
            resume_value = {
                "action": "reject",
                "feedback": "user reject design; run aborted",
            }
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
                # 允许用户直接粘贴 JSON 覆盖 experiments
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
# 原 main 主流程（最小改动）
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
    }

    mem_store = Mem0MilvusMemoryStore()

    config = {
        "configurable": {
            "thread_id": "demo_thread",
            "mem_store": mem_store,
        }
    }

    # ✅ 全程 async 推进
    final_state = await app.ainvoke(init_state, config=config)

    # ✅ HITL：改为 async 版本，不再 to_thread
    final_state = await handle_hitl_interrupts_async(app, final_state, config=config)

    print("\n=== FINAL REPORT ===")
    report = final_state.get("analyst_report", {}) or {}
    if not report:
        print("ok: False")
        print("message: user rejected the experiment design; run aborted")
    else:
        for k, v in report.items():
            print(f"{k}: {v}")


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "data", "runs")
    os.makedirs(base, exist_ok=True)
    asyncio.run(amain())


if __name__ == "__main__":
    main()
