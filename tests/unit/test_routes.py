# tests/unit/test_routes.py
import pytest
from langgraph.graph import END
from backend.agent.graph import _route_after_designer, _route_after_analyst

@pytest.mark.unit
@pytest.mark.parametrize("stage,expected", [
    ("need_human", "human_review"),
    ("dispatch", END),
    ("analyze", "analyst"),
    ("done", END),
    ("collect", "designer"),
    ("need_redesign", "designer"),
    ("init", "designer"),
])
def test_route_after_designer(stage, expected):
    # TODO: 如果 GlobalState 需要其他字段，这里补齐
    assert _route_after_designer({"stage": stage}) == expected

@pytest.mark.unit
def test_route_after_analyst_ok():
    assert _route_after_analyst({"analyst_report": {"ok": True}}) == END

@pytest.mark.unit
def test_route_after_analyst_not_ok():
    assert _route_after_analyst({"analyst_report": {"ok": False}}) == "need_redesign"
