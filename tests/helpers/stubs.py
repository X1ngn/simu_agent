from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
from langgraph.types import Command

@dataclass
class HitlDecision:
    action: str
    edited_text: Optional[str] = None

def decider_accept(payload: Dict[str, Any], state: Dict[str, Any]) -> HitlDecision:
    return HitlDecision("accept")

def decider_reject(payload: Dict[str, Any], state: Dict[str, Any]) -> HitlDecision:
    return HitlDecision("reject")

def decider_edit(payload: Dict[str, Any], state: Dict[str, Any]) -> HitlDecision:
    # TODO: 改成你期望的 edited_text JSON
    return HitlDecision("edit", edited_text='{"experiments":[{"exp_id":"gbs_128"}]}')

def stub_designer_need_human(state: Dict[str, Any]) -> Dict[str, Any]:
    """第一轮：产出方案 -> need_human"""
    s = dict(state)
    s["stage"] = "need_human"
    s["experiments_plan"] = {"experiments": [{"exp_id": "gbs_64"}]}
    return s

def stub_human_review_interrupt(state: Dict[str, Any]) -> Command:
    """触发 interrupt，让 handle_hitl_interrupts 驱动"""
    return Command(interrupt={"experiments_plan": state.get("experiments_plan")})

def stub_worker_collect(state: Dict[str, Any]) -> Dict[str, Any]:
    """worker 返回 -> stage=collect 触发 fan-in 回 designer"""
    s = dict(state)
    s["stage"] = "collect"
    s.setdefault("exam_results", [])
    s["exam_results"].append({"exp_id": "gbs_64", "tps": 123, "oom": False})
    return s

def stub_designer_after_collect_to_analyze(state: Dict[str, Any]) -> Dict[str, Any]:
    """fan-in 后：designer 聚合 -> 进入 analyze"""
    s = dict(state)
    s["stage"] = "analyze"
    return s

def stub_analyst_ok(state: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(state)
    s["analyst_report"] = {"ok": True, "message": "ok"}
    s["stage"] = "done"
    return s

def stub_analyst_not_ok(state: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(state)
    s["analyst_report"] = {"ok": False, "message": "need redesign"}
    s["stage"] = "need_redesign"
    return s
