# tests/helpers/fakes.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, List
from langgraph.types import Command

@dataclass
class HitlDecision:
    action: str
    edited_text: Optional[str] = None

class FakeDesigner:
    """
    TODO:
    - 用脚本控制多次调用返回不同 stage
    - e.g. 第一次 -> need_human + experiments_plan
            第二次（accept后）-> dispatch 并 Command(goto=[Send...])（如果你要测 fan-out）
            第三次（collect后）-> analyze
    """
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.steps: List[Callable[[Dict[str, Any]], Any]] = []

    def push(self, fn: Callable[[Dict[str, Any]], Any]):
        self.steps.append(fn)

    def __call__(self, state: Dict[str, Any]) -> Any:
        self.calls.append({"node": "designer", "stage": state.get("stage")})

        if self.steps:
            return self.steps.pop(0)(state)

        # 默认行为：直接触发 need_human
        new_state = dict(state)
        new_state["stage"] = "need_human"
        new_state["experiments_plan"] = {"experiments": [{"exp_id": "gbs_64"}]}
        return new_state

class FakeHumanReview:
    """
    TODO:
    - 默认返回 Command(interrupt=payload)
    - payload 结构建议与你 _extract_interrupt_payload 对齐
    """
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, state: Dict[str, Any]) -> Any:
        self.calls.append({"node": "human_review", "stage": state.get("stage")})
        payload = {"experiments_plan": state.get("experiments_plan")}
        return Command(interrupt=payload)

class FakeWorker:
    """
    TODO:
    - 返回 worker 结果，并把 stage 推到 collect，触发 fan-in 回 designer
    - 可加入乱序/延迟模拟（如果你用 stream 或异步）
    """
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, state: Dict[str, Any]) -> Any:
        self.calls.append({"node": "exam_worker", "stage": state.get("stage")})
        new_state = dict(state)
        new_state["stage"] = "collect"
        new_state.setdefault("exam_results", [])
        new_state["exam_results"].append({"exp_id": "gbs_64", "oom": False, "tps": 123})
        return new_state

class FakeAnalyst:
    """
    TODO:
    - 通过构造 ok=True/False 覆盖 _route_after_analyst 两个分支
    """
    def __init__(self, ok: bool = True):
        self.ok = ok
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, state: Dict[str, Any]) -> Any:
        self.calls.append({"node": "analyst", "stage": state.get("stage")})
        new_state = dict(state)
        new_state["analyst_report"] = {"ok": self.ok, "message": "fake"}
        return new_state

# ------- scripted deciders for HITL -------
def decider_accept(payload: Dict[str, Any], state: Dict[str, Any]) -> HitlDecision:
    # TODO: 你可以在这里对 payload 做结构断言（例如 experiments_plan 存在）
    return HitlDecision(action="accept")

def decider_reject(payload: Dict[str, Any], state: Dict[str, Any]) -> HitlDecision:
    return HitlDecision(action="reject")

def decider_edit(payload: Dict[str, Any], state: Dict[str, Any]) -> HitlDecision:
    # TODO: 这里给一个用户编辑后的 JSON 字符串（要符合你后续解析约定）
    return HitlDecision(action="edit", edited_text='{"experiments":[{"exp_id":"gbs_128"}]}')
