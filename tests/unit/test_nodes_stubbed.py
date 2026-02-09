import json
import pytest
from types import SimpleNamespace

from backend.agent.agents.designer import designer_agent
from backend.agent.state import GlobalState


# =========================
# 通用 Stub / Fake
# =========================

class FakeLLM:
    """最小 LLM stub：只实现 invoke，返回带 content 的对象"""

    def __init__(self, content: str):
        self._content = content

    def invoke(self, messages):
        # designer 只用 resp.content
        return SimpleNamespace(content=self._content)


class FakeMemStore:
    """memory store stub"""

    def search_for_designer(self, state, query, top_k=5):
        # 返回“历史记忆”，但对本测试不重要
        return [
            {
                "memory": "previous plan rejected due to OOM",
                "metadata": {"status": "rejected"},
                "created_at": "2024-01-01",
            }
        ]


class DummyConfig(dict):
    """RunnableConfig 最小替代（本测试只关心能被 _get_mem_store 使用）"""
    pass


# =========================
# 工具函数：patch LLM + memory
# =========================

def patch_llm_and_memory(monkeypatch, llm_content: str):
    """
    正确 patch 点：
    - designer 模块内已经 import 的符号
    """
    import backend.agent.agents.designer as designer_mod

    monkeypatch.setattr(
        designer_mod,
        "get_llm_for_agent",
        lambda role: FakeLLM(llm_content),
    )

    monkeypatch.setattr(
        designer_mod,
        "_get_mem_store",
        lambda config: FakeMemStore(),
    )

    import backend.agent.agents.designer as designer_mod

    assert isinstance(
        designer_mod.get_llm_for_agent("designer"),
        FakeLLM
    )


# =========================
# Test 1: 正常生成 experiments_plan
# =========================

@pytest.mark.unit
def test_designer_generates_experiments_plan(monkeypatch):
    """
    覆盖分支：
    - stage=init
    - LLM 返回合法 JSON
    - 正常生成 experiments_plan -> stage=need_human
    """

    llm_json = {
        "experiments": [
            {
                "exp_id": "gbs_64",
                "description": "sweep gbs=64",
                "model_patch": {},
                "run_patch": {"train.global_batch_size": 64},
                "hw_patch": {},
            }
        ]
    }

    patch_llm_and_memory(monkeypatch, json.dumps(llm_json))

    state: GlobalState = {
        "stage": "init",
        "debug": False,
        "user_intent": "测试 gbs 对吞吐的影响",
        "retries": 0,
        "max_retries": 1,
    }
    config = DummyConfig()

    out = designer_agent(state, config)

    assert out["stage"] == "need_human"
    assert "experiments_plan" in out
    assert isinstance(out["experiments_plan"], list)
    assert len(out["experiments_plan"]) == 1

    exp = out["experiments_plan"][0]
    assert exp["exp_id"] == "gbs_64"
    assert "out_dir" in exp
    assert isinstance(exp["run_patch"], dict)


# =========================
# Test 2: LLM 返回非法 JSON -> fallback -> RuntimeError
# =========================

@pytest.mark.unit
def test_designer_llm_bad_json_fallback(monkeypatch):
    """
    覆盖分支：
    - LLM 返回非 JSON
    - json.loads 失败
    - 触发 _fallback_default -> RuntimeError
    """

    patch_llm_and_memory(monkeypatch, "THIS IS NOT JSON")

    state: GlobalState = {
        "stage": "init",
        "debug": False,
        "user_intent": "测试 gbs 对吞吐的影响",
        "retries": 0,
        "max_retries": 1,
    }
    config = DummyConfig()

    with pytest.raises(RuntimeError):
        designer_agent(state, config)


# =========================
# Test 3: experiments 为空 -> fallback
# =========================

@pytest.mark.unit
def test_designer_empty_experiments_fallback(monkeypatch):
    """
    覆盖分支：
    - LLM 返回 {"experiments": []}
    - experiments 为空
    - 触发 fallback
    """

    llm_json = {"experiments": []}
    patch_llm_and_memory(monkeypatch, json.dumps(llm_json))

    state: GlobalState = {
        "stage": "init",
        "debug": False,
        "user_intent": "测试 gbs 对吞吐的影响",
        "retries": 0,
        "max_retries": 1,
    }
    config = DummyConfig()

    with pytest.raises(RuntimeError):
        designer_agent(state, config)
