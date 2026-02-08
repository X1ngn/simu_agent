from __future__ import annotations

import os
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig

from backend.agent.state import GlobalState, AnalystReport, ExamResult
from backend.agent.tools.csv_io import read_csv_head_row, read_csv_all, write_csv

from backend.agent.llm_factory import get_llm_for_agent
from backend.agent.prompts.analyst import analyst_prompt


from langchain_core.messages import SystemMessage, HumanMessage
import json

def _dbg(debug: bool, msg: str) -> None:
    if debug:
        print(msg)


def gen_brief(debug, successes, rows, anomalies) -> None:
    for r in successes:
        csv_path = r.get("csv_path", "")
        exp_id = r.get("exp_id", "unknown")
        if not csv_path or not os.path.exists(csv_path):
            anomalies.append({"exp_id": exp_id, "reason": "missing_csv", "csv_path": csv_path})
            continue

        head = read_csv_head_row(csv_path)
        if not head:
            anomalies.append({"exp_id": exp_id, "reason": "empty_csv", "csv_path": csv_path})
            continue

        # 打一些基本字段到汇总表
        head["_exp_id"] = exp_id
        head["_csv_path"] = csv_path
        rows.append(head)

    _dbg(debug, f"gen_brief original num: {len(successes)}, brief num: {len(rows)}")


def save_brief(debug, rows, run_id) -> str:
    # 保存汇总结果
    summary_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "runs",
        run_id,
        "summary.csv",
    )
    summary_path = os.path.abspath(summary_path)
    write_csv(summary_path, rows)

    _dbg(debug, f" {summary_path}")

    return summary_path


def analyze_data(debug, state, rows, anomalies, summary_path) -> None:
    _dbg(debug, f"Analyzing {summary_path}")

    llm = get_llm_for_agent("analyst")
    system_prompt = analyst_prompt

    run_id = state.get("run_id", "run_unknown")
    user_intent = state.get("user_intent", "")

    # TODO: 修改为真正的业务逻辑prompt
    llm_task = {
        "goal": "Help decide what parts of the pipeline should be treated as anomalies and what deep-dive actions to run.",
        "constraints": [
            "Return strict JSON only.",
            "Do not include markdown.",
        ],
        "inputs": {
            "run_id": run_id,
            "user_intent": user_intent,
            "successes_count": len(rows),
            "successes_data": rows,
            "anomalies": anomalies,
            "current_rules": {
                "throughput_tokens_per_s": "flag if <= 0",
                "peak_mem_gb": "flag if >= 200 (dummy threshold)",
                "missing_csv": "flag if csv_path missing or file not exists",
                "empty_csv": "flag if head row empty",
            },
        },
        "output_schema": {
            "anomaly_rules": "list of rules with metric, operator, threshold, reason",
            "deep_dive_plan": "list of actions, each has exp_selector and action_type",
            "notes": "short string",
        },
    }

    llm_resp = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(llm_task, ensure_ascii=False)),
        ]
    )

    # === [NEW] parse JSON from LLM ===
    try:
        llm_json = json.loads(llm_resp.content)
    except Exception:
        # LLM 输出不合规时：降级，不影响你原流程
        llm_json = {
            "anomaly_rules": [],
            "deep_dive_plan": [],
            "notes": "LLM output not valid JSON; fallback to hard-coded rules.",
        }

    if debug:
        print("[analyst][llm] decision:", llm_json)

    return llm_json


def analyst_agent(state: GlobalState, config: RunnableConfig) -> Dict[str, Any]:
    """
    analyst_agent（ReAct）：读取每个实验 csv 的头行汇总为 summary.csv。
    如发现异常（例如 throughput 为 0 或 peak_mem 异常爆炸），可进一步读全量做诊断。
    """
    debug = bool(state.get("debug", False))

    run_id = state.get("run_id", "run_unknown")
    successes: List[ExamResult] = state.get("success_results", [])  # type: ignore
    user_intent = state.get("user_intent", "")

    _dbg(debug, f"[analyst] start, run_id={run_id}, successes={len(successes)}")
    _dbg(debug, f"[analyst] user_intent={user_intent}")

    rows: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []

    # 读取所有最优结果
    gen_brief(debug, successes, rows, anomalies)
    summary_path = save_brief(debug, rows, run_id)

    # 调用大模型进行分析
    analyze_data(debug, state, rows, anomalies, summary_path)

    report: AnalystReport
    if anomalies:
        report = {
            "ok": False,
            "summary_csv_path": summary_path,
            "anomalies": anomalies,
            "message": "Found anomalies in simulation outputs; may need redesign/re-run.",
        }
    else:
        report = {
            "ok": True,
            "summary_csv_path": summary_path,
            "anomalies": [],
            "message": "All experiments aggregated successfully.",
        }

    _dbg(debug, f"[analyst] wrote summary: {summary_path}")
    return {"analyst_report": report}


    # 简单异常规则（你后续可以替换为更严格的 domain 规则）
    #     try:
    #         thr = float(head.get("throughput_tokens_per_s", "0") or "0")
    #         mem = float(head.get("peak_mem_gb", "0") or "0")
    #         if thr <= 0:
    #             anomalies.append({"exp_id": exp_id, "reason": "non_positive_throughput", "thr": thr})
    #         if mem >= 200:  # dummy 阈值
    #             anomalies.append({"exp_id": exp_id, "reason": "mem_too_high", "mem": mem})
    #     except Exception:
    #         anomalies.append({"exp_id": exp_id, "reason": "metric_parse_error", "head": head})
    #
    # # 如发现异常，进一步读取全量做诊断（示例：只对异常的 exp 做全量读取）
    # if anomalies:
    #     _dbg(debug, f"[analyst] anomalies found: {len(anomalies)} -> deep dive")
    #     for a in anomalies:
    #         exp_id = a.get("exp_id", "")
    #         target = next((x for x in successes if x.get("exp_id") == exp_id), None)
    #         if not target:
    #             continue
    #         csv_path = target.get("csv_path", "")
    #         if csv_path and os.path.exists(csv_path):
    #             all_rows = read_csv_all(csv_path)
    #             a["rows_count"] = len(all_rows)