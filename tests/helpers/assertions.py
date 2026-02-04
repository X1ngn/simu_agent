# tests/helpers/assertions.py
from __future__ import annotations
from typing import Any, Dict, List


def assert_has_interrupt(state: Dict[str, Any]):
    """
    TODO:
    - 断言 state["__interrupt__"] 存在
    - 断言 payload 里有 experiments_plan
    """
    pass


def assert_stage(state: Dict[str, Any], expected: str):
    """
    TODO:
    - assert state.get("stage") == expected
    """
    pass


def assert_report_ok(state: Dict[str, Any]):
    """
    TODO:
    - assert state["analyst_report"]["ok"] is True
    """
    pass


def assert_execution_path(call_logs: List[Dict[str, Any]], expected_nodes: List[str]):
    """
    TODO:
    - 用 fakes["designer"].log.calls 等收集到的调用序列断言路径
    - 注意：并发 worker 时顺序可能不固定，必要时用集合/计数断言
    """
    pass
