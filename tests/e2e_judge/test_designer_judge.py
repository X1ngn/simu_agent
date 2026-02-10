import os
import json
import pytest
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from backend.agent.agents.designer import designer_agent
from backend.agent.llm_factory import get_llm_for_agent


# =========================
# Judge Prompt（工程化、稳定）
# =========================

JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator of experiment design quality.

Your task:
- Judge whether the proposed experiments MATCH the user's intent.
- Be conservative. If uncertain, lower the score.

Rules:
- Return STRICT JSON ONLY.
- Do NOT include markdown or commentary.
- Output schema:
{
  "pass": boolean,
  "score": number,   // between 0.0 and 1.0
  "reasons": [string]
}
"""

JUDGE_RUBRIC = {
    "criteria": [
        "Experiments must clearly operationalize the user's intent.",
        "Key variables mentioned in the intent must be reflected in run/model/hw patches.",
        "The experiment set should be coherent and non-random.",
        "Experiments should aim to reveal trends or failure boundaries if requested.",
    ],
    "scoring": {
        "1.0": "Perfect alignment with user intent.",
        "0.7": "Mostly aligned, minor omissions.",
        "0.4": "Weak or partial alignment.",
        "0.0": "Unrelated or incorrect.",
    },
    "pass_threshold": 0.7,
}


# =========================
# 测试本体（异步版）
# =========================
"""
$env:RUN_E2E_JUDGE="1"
pytest -s -q tests/e2e_judge/test_designer_judge.py::test_designer_plan_matches_intent
"""
@pytest.mark.e2e_judge
@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_E2E_JUDGE") != "1", reason="E2E judge disabled by default")
async def test_designer_plan_matches_intent():
    """
    E2E Judge Test (async):
    - 真实运行 designer（async），得到 experiments_plan
    - 用独立 judge LLM 评估设计是否匹配 user_intent
    - judge 输出结构化 JSON：pass/score/reasons
    - assert score >= 阈值
    """

    # ------------------------------------------------------------
    # 1. 运行 designer，得到 experiments_plan
    # ------------------------------------------------------------
    user_intent = "固定模型与硬件，扫 global_batch_size 对吞吐与显存峰值的影响，并找出失败边界。"

    state = {
        "stage": "init",
        "debug": False,
        "user_intent": user_intent,
        "retries": 0,
        "max_retries": 1,
    }

    # ✅ 建议统一用 RunnableConfig（你生产节点也在用这个类型）
    config = RunnableConfig(configurable={"thread_id": "e2e_judge_designer"})

    # ✅ 异步 designer
    out = await designer_agent(state, config)

    assert out["stage"] == "need_human"
    assert "experiments_plan" in out
    experiments = out["experiments_plan"]
    assert isinstance(experiments, list)
    assert len(experiments) > 0

    # ------------------------------------------------------------
    # 2. 构造 judge 输入（intent + experiments + rubric）
    # ------------------------------------------------------------
    judge_input: Dict[str, Any] = {
        "user_intent": user_intent,
        "experiments_plan": [
            {
                "exp_id": e["exp_id"],
                "description": e.get("description"),
                "model_patch": e.get("model_patch"),
                "run_patch": e.get("run_patch"),
                "hw_patch": e.get("hw_patch"),
            }
            for e in experiments
        ],
        "rubric": JUDGE_RUBRIC,
    }

    judge_prompt = json.dumps(judge_input, ensure_ascii=False)

    # ------------------------------------------------------------
    # 3. 调用 judge LLM（temperature=0，强一致性）
    # ------------------------------------------------------------
    # 复用 designer 的 LLM 实例即可（你目前没有 judge agent 配置）
    judge_llm = get_llm_for_agent("designer")

    # ✅ async LLM call
    # 注意：不同 ChatModel 对温度参数传递方式不同：
    # - 有的用 model.bind(temperature=0) 再 ainvoke
    # - 有的允许 ainvoke(..., temperature=0)
    # 这里采用最稳的 bind 方式。
    llm0 = judge_llm.bind(temperature=0.0)

    resp = await llm0.ainvoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=judge_prompt),
        ]
    )

    assert resp is not None
    assert hasattr(resp, "content")
    assert isinstance(resp.content, str)
    assert resp.content.strip(), "Judge LLM returned empty content"

    # ------------------------------------------------------------
    # 4. 解析 judge 输出（必须是结构化 JSON）
    # ------------------------------------------------------------
    try:
        verdict = json.loads(resp.content)
    except Exception as e:
        pytest.fail(
            f"Judge output is not valid JSON.\n"
            f"Error: {e}\n"
            f"Raw content:\n{resp.content[:500]}"
        )

    # ------------------------------------------------------------
    # 5. 结构断言（不是自然语言）
    # ------------------------------------------------------------
    assert isinstance(verdict, dict)
    assert "pass" in verdict
    assert "score" in verdict
    assert "reasons" in verdict

    assert isinstance(verdict["pass"], bool)
    assert isinstance(verdict["score"], (int, float))
    assert isinstance(verdict["reasons"], list)

    # ------------------------------------------------------------
    # 6. 质量门槛断言（核心）
    # ------------------------------------------------------------
    threshold = JUDGE_RUBRIC["pass_threshold"]
    assert verdict["score"] >= threshold, (
        f"Designer plan does not sufficiently match intent.\n"
        f"Score={verdict['score']} < threshold={threshold}\n"
        f"Reasons={verdict.get('reasons')}"
    )
