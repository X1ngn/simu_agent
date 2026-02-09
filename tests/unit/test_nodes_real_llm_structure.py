import os
import json
import pytest

from langchain_core.messages import SystemMessage, HumanMessage

from backend.agent.llm_factory import get_llm_for_agent
from backend.agent.prompts.designer import designer_prompt

'''
$env:RUN_REAL_LLM="1"                                                                                      
pytest tests/unit/test_nodes_real_llm_structure.py::test_real_llm_temperature_zero_returns_structured_fields
'''
@pytest.mark.unit
@pytest.mark.skipif(os.getenv("RUN_REAL_LLM") != "1", reason="Real LLM disabled by default")
def test_real_llm_temperature_zero_returns_structured_fields():
    """
    验证目标：
    - 使用真实 LLM
    - temperature=0
    - 返回内容可 parse 为 JSON
    - 且包含 designer 约定的结构性字段（experiments[*].exp_id 等）
    """

    # ===== 1. 获取真实 LLM（designer 角色）=====
    llm = get_llm_for_agent("designer")

    # ===== 2. 构造最小但合法的 designer 请求 =====
    # 关键原则：
    # - prompt 必须符合 designer_prompt 的“STRICT JSON ONLY”约定
    # - 内容要足够简单，避免模型自由发挥
    llm_req = {
        "task": "design_experiments",
        "rules": [
            "Return STRICT JSON ONLY.",
            "Output must be an object with key 'experiments'.",
        ],
        "context": {
            "user_intent": "测试 temperature=0 时是否稳定返回结构化实验设计",
        },
        "output_schema": {
            "experiments": [
                {
                    "exp_id": "string",
                    "description": "string",
                    "model_patch": {},
                    "run_patch": {},
                    "hw_patch": {},
                }
            ]
        },
    }

    prompt = json.dumps(llm_req, ensure_ascii=False)

    # ===== 3. 调用真实 LLM（显式 temperature=0）=====
    resp = llm.invoke(
        [
            SystemMessage(content=designer_prompt),
            HumanMessage(content=prompt),
        ],
        temperature=0.0,   # ★ 核心检查点
    )

    assert resp is not None
    assert hasattr(resp, "content")
    assert isinstance(resp.content, str)
    assert resp.content.strip(), "LLM returned empty content"

    # ===== 4. 必须能 parse 成 JSON =====
    try:
        parsed = json.loads(resp.content)
    except Exception as e:
        pytest.fail(
            f"LLM output is not valid JSON.\n"
            f"Error: {e}\n"
            f"Raw content:\n{resp.content[:500]}"
        )

    # ===== 5. 结构性断言（而非全文断言）=====
    assert isinstance(parsed, dict), "Top-level JSON must be an object"
    assert "experiments" in parsed, "Missing 'experiments' key"
    assert isinstance(parsed["experiments"], list), "'experiments' must be a list"
    assert len(parsed["experiments"]) > 0, "'experiments' list should not be empty"

    # 只检查 schema，不检查具体内容
    exp = parsed["experiments"][0]
    assert isinstance(exp, dict)

    # designer 协议里的关键字段
    assert "exp_id" in exp
    assert isinstance(exp["exp_id"], str)
    assert exp["exp_id"].strip() != ""

    assert "model_patch" in exp
    assert isinstance(exp["model_patch"], dict)

    assert "run_patch" in exp
    assert isinstance(exp["run_patch"], dict)

    assert "hw_patch" in exp
    assert isinstance(exp["hw_patch"], dict)
