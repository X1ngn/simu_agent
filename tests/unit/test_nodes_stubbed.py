import json
import pytest
from types import SimpleNamespace

from backend.agent.agents.designer import designer_agent
from backend.agent.state import GlobalState
from langchain_core.runnables import RunnableConfig


# =========================
# 通用 Stub / Fake
# =========================

class FakeLLM:
    """最小 LLM stub：同时支持 invoke/ainvoke"""

    def __init__(self, content: str):
        self._content = content

    def invoke(self, messages):
        return SimpleNamespace(content=self._content)

    async def ainvoke(self, messages):
        return SimpleNamespace(content=self._content)

    def bind(self, **kwargs):
        # 兼容 judge/temperature=0 这类 bind 用法
        return self


class FakeMemStore:
    """memory store stub"""

    def search_for_designer(self, state, query, top_k=5):
        return [
            {
                "memory": "previous plan rejected due to OOM",
                "metadata": {"status": "rejected"},
                "created_at": "2024-01-01",
            }
        ]

    def __getattr__(self, name: str):
        # 兜底：human_review / 其它地方调用 store.xxx 不至于崩
        def _noop(*args, **kwargs):
            return None
        return _noop


# =========================
# 工具函数：patch LLM + memory
# =========================

def patch_llm_and_memory(monkeypatch, llm_content: str):
    """
    正确 patch 点：
    - patch designer 模块内已 import 的符号（designer_mod.get_llm_for_agent / designer_mod._get_mem_store）
    """
    import backend.agent.agents.designer as designer_mod

    monkeypatch.setattr(designer_mod, "get_llm_for_agent", lambda role: FakeLLM(llm_content))
    monkeypatch.setattr(designer_mod, "_get_mem_store", lambda config: FakeMemStore())

    assert isinstance(designer_mod.get_llm_for_agent("designer"), FakeLLM)


def _mk_config() -> RunnableConfig:
    # ✅ 用 RunnableConfig，避免你生产代码里 _get_mem_store(config) 依赖 config.configurable
    return RunnableConfig(configurable={"thread_id": "unit_test", "mem_store": FakeMemStore()})


# =========================
# Test 1: 正常生成 experiments_plan
# =========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_designer_generates_experiments_plan(monkeypatch):
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
    config = _mk_config()

    out = await designer_agent(state, config)

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
@pytest.mark.asyncio
async def test_designer_llm_bad_json_fallback(monkeypatch):
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
    config = _mk_config()

    with pytest.raises(RuntimeError):
        await designer_agent(state, config)


# =========================
# Test 3: experiments 为空 -> fallback
# =========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_designer_empty_experiments_fallback(monkeypatch):
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
    config = _mk_config()

    with pytest.raises(RuntimeError):
        await designer_agent(state, config)
