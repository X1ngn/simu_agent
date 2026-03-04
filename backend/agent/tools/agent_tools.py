"""
Tool definitions for the two-phase Agent architecture.

Phase 1 (Plan Agent): search_memory, ask_user, submit_experiment_plan
Phase 2 (Exec Agent): run_simulation_tool, run_batch, analyze_results, finish
"""
from __future__ import annotations

import json
import os
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict
from langgraph.types import interrupt

# Suppress Pydantic warning for 'model_patch' field which clashes with
# Pydantic v2's protected 'model_' namespace.  This is cosmetic – the field
# works correctly; the warning comes from langchain-core's internal
# _create_subset_model which does not inherit our ConfigDict override.
warnings.filterwarnings(
    "ignore",
    message=r'.*Field "model_patch".*protected namespace.*',
    category=UserWarning,
)

from backend.agent.memory import Mem0MilvusMemoryStore
from backend.agent.tools.json_io import read_json, write_json, apply_dotpatch
from backend.agent.tools.sim_engine import run_simulation as _run_sim_engine, SimulationError
from backend.agent.tools.csv_io import read_csv_head_row, write_csv


SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "samples")


# ---------------------------------------------------------------------------
# Phase 1 Tools
# ---------------------------------------------------------------------------

@tool
def search_memory(query: str) -> str:
    """搜索历史实验记忆，了解过去类似实验的结果和反馈。

    Args:
        query: 搜索查询，通常是用户意图的关键词

    Returns:
        历史记忆列表的 JSON 字符串，包含过去的实验方案和人工反馈
    """
    try:
        store = Mem0MilvusMemoryStore()
        memories = store.search_for_designer(
            state={"user_id": "default_user"},
            query=query,
            top_k=5,
        )
        if not memories:
            return json.dumps({"memories": [], "note": "未找到相关历史记忆"}, ensure_ascii=False)
        return json.dumps({"memories": memories}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"memories": [], "error": f"记忆搜索失败: {e}"}, ensure_ascii=False)


@tool
def ask_user(question: str) -> str:
    """向用户提问获取更多信息。当实验意图不明确时使用。

    Args:
        question: 要问用户的问题

    Returns:
        用户的回答
    """
    resp = interrupt({"type": "ask_user", "question": question})
    return str(resp)


@tool
def submit_experiment_plan(experiments: list) -> str:
    """提交最终实验方案，进入人工审核环节。设计完成后必须调用此工具提交方案。

    Args:
        experiments: 实验方案列表，每个实验包含 exp_id, description, model_patch, run_patch, hw_patch

    Returns:
        确认消息
    """
    if not experiments or not isinstance(experiments, list):
        return "错误：experiments 必须是非空列表"

    for exp in experiments:
        if not isinstance(exp, dict):
            return f"错误：每个实验必须是 dict，收到 {type(exp)}"
        required = ["exp_id", "description", "model_patch", "run_patch", "hw_patch"]
        missing = [k for k in required if k not in exp]
        if missing:
            return f"错误：实验 {exp.get('exp_id', '?')} 缺少字段: {missing}"

    return f"方案已提交，包含 {len(experiments)} 个实验，等待人工审核。"


# ---------------------------------------------------------------------------
# Phase 2 Tools - internal helpers
# ---------------------------------------------------------------------------

def _make_run_id() -> str:
    return uuid.uuid4().hex[:8]


def _run_single_experiment(
    exp: Dict[str, Any],
    run_id: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """Execute a single simulation experiment. Core logic extracted from exam_worker."""
    exp_id = exp.get("exp_id", f"exp_{uuid.uuid4().hex[:4]}")
    base_out = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "runs", run_id)
    )
    out_dir = os.path.join(base_out, exp_id)

    try:
        sample_model = read_json(os.path.join(SAMPLES_DIR, "model.json"))
        sample_run = read_json(os.path.join(SAMPLES_DIR, "run.json"))
        sample_hw = read_json(os.path.join(SAMPLES_DIR, "hw.json"))

        model_cfg = apply_dotpatch(sample_model, exp.get("model_patch", {}))
        run_cfg = apply_dotpatch(sample_run, exp.get("run_patch", {}))
        hw_cfg = apply_dotpatch(sample_hw, exp.get("hw_patch", {}))

        model_path = os.path.join(out_dir, "model.json")
        run_path = os.path.join(out_dir, "run.json")
        hw_path = os.path.join(out_dir, "hw.json")
        write_json(model_path, model_cfg)
        write_json(run_path, run_cfg)
        write_json(hw_path, hw_cfg)

        csv_path = os.path.join(out_dir, "result.csv")
        _run_sim_engine(
            model_json_path=model_path,
            run_json_path=run_path,
            hw_json_path=hw_path,
            out_csv_path=csv_path,
            debug=debug,
        )

        # Read key metrics from csv for summary
        head = read_csv_head_row(csv_path)
        return {
            "exp_id": exp_id,
            "ok": True,
            "csv_path": csv_path,
            "out_dir": out_dir,
            "throughput_tokens_per_s": head.get("throughput_tokens_per_s"),
            "peak_mem_gb": head.get("peak_mem_gb"),
            "status": head.get("status", "ok"),
        }

    except SimulationError as e:
        return {"exp_id": exp_id, "ok": False, "out_dir": out_dir, "error": str(e)}
    except Exception as e:
        return {"exp_id": exp_id, "ok": False, "out_dir": out_dir, "error": f"unexpected: {repr(e)}"}


# ---------------------------------------------------------------------------
# Phase 2 Tools
# ---------------------------------------------------------------------------

class RunSimulationInput(BaseModel):
    """Input schema for run_simulation_tool."""
    model_config = ConfigDict(protected_namespaces=())

    exp_id: str
    description: str
    model_patch: dict
    run_patch: dict
    hw_patch: dict
    run_id: str = ""


@tool(args_schema=RunSimulationInput)
def run_simulation_tool(
    exp_id: str,
    description: str,
    model_patch: dict,
    run_patch: dict,
    hw_patch: dict,
    run_id: str = "",
) -> str:
    """执行单个仿真实验。

    Args:
        exp_id: 实验唯一标识
        description: 实验描述
        model_patch: 模型配置补丁
        run_patch: 运行时配置补丁 (dot-path 格式)
        hw_patch: 硬件配置补丁
        run_id: 运行批次ID（可选，不传则自动生成）

    Returns:
        实验结果摘要 JSON，包含 csv_path、throughput、peak_mem_gb 等。
        如果实验失败（如 OOM），返回错误信息。
    """
    rid = run_id or _make_run_id()
    exp = {
        "exp_id": exp_id,
        "description": description,
        "model_patch": model_patch,
        "run_patch": run_patch,
        "hw_patch": hw_patch,
    }
    result = _run_single_experiment(exp, rid)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def run_batch(experiments: list, run_id: str = "") -> str:
    """批量执行多个仿真实验。适用于初始扫描阶段。内部使用线程池并行执行。

    Args:
        experiments: 实验列表，每个实验包含 exp_id, description, model_patch, run_patch, hw_patch
        run_id: 运行批次ID（可选，不传则自动生成）

    Returns:
        所有实验结果摘要的 JSON 列表
    """
    rid = run_id or _make_run_id()
    results = []

    with ThreadPoolExecutor(max_workers=min(len(experiments), 8)) as pool:
        futures = {
            pool.submit(_run_single_experiment, exp, rid): exp
            for exp in experiments
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by exp_id for deterministic output
    results.sort(key=lambda r: r.get("exp_id", ""))
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def analyze_results(exp_ids: list, run_id: str = "") -> str:
    """分析已完成的实验结果，识别趋势和异常。

    Args:
        exp_ids: 要分析的实验 ID 列表
        run_id: 运行批次ID

    Returns:
        分析报告 JSON，包含每个实验的指标数据和汇总
    """
    rid = run_id
    base_out = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "runs", rid)
    )

    rows: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []

    for eid in exp_ids:
        csv_path = os.path.join(base_out, eid, "result.csv")
        if not os.path.exists(csv_path):
            anomalies.append({"exp_id": eid, "reason": "missing_csv", "csv_path": csv_path})
            continue
        head = read_csv_head_row(csv_path)
        if not head:
            anomalies.append({"exp_id": eid, "reason": "empty_csv", "csv_path": csv_path})
            continue
        head["_exp_id"] = eid
        rows.append(head)

    summary_path = ""
    if rows:
        summary_path = os.path.join(base_out, "summary.csv")
        write_csv(summary_path, rows)

    report = {
        "summary_csv_path": summary_path,
        "experiment_count": len(exp_ids),
        "success_count": len(rows),
        "anomaly_count": len(anomalies),
        "data": rows,
        "anomalies": anomalies,
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


@tool
def finish(summary: str, findings: list) -> str:
    """所有实验完成，输出最终结论。当已有足够数据回答用户问题时调用。

    Args:
        summary: 总结文本
        findings: 发现列表，每项包含 metric 和 trend

    Returns:
        确认消息
    """
    return json.dumps({
        "status": "completed",
        "summary": summary,
        "findings": findings,
    }, ensure_ascii=False, indent=2)
