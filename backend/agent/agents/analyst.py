from __future__ import annotations

import os
import json
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agent.state import GlobalState, AnalystReport, ExamResult
from backend.agent.tools.csv_io import read_csv_head_row, write_csv

from backend.agent.llm_factory import get_llm_for_agent
from backend.agent.prompts.analyst import analyst_prompt


def _dbg(debug: bool, msg: str) -> None:
    if debug:
        print(msg)


def gen_brief(debug: bool, successes: List[ExamResult], rows: List[Dict[str, Any]], anomalies: List[Dict[str, Any]]) -> None:
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

        head["_exp_id"] = exp_id
        head["_csv_path"] = csv_path
        rows.append(head)

    _dbg(debug, f"gen_brief original num: {len(successes)}, brief num: {len(rows)}")


def save_brief(debug: bool, rows: List[Dict[str, Any]], run_id: str) -> str:
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
    _dbg(debug, f"{summary_path}")
    return summary_path


async def analyze_data(debug: bool, state: GlobalState, rows: List[Dict[str, Any]], anomalies: List[Dict[str, Any]], summary_path: str) -> Dict[str, Any]:
    _dbg(debug, f"Analyzing {summary_path}")

    llm = get_llm_for_agent("analyst")
    system_prompt = analyst_prompt

    run_id = state.get("run_id", "run_unknown")
    user_intent = state.get("user_intent", "")

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

    # ✅ async call
    llm_resp = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(llm_task, ensure_ascii=False)),
        ]
    )

    try:
        llm_json = json.loads(llm_resp.content)
    except Exception:
        llm_json = {
            "anomaly_rules": [],
            "deep_dive_plan": [],
            "notes": "LLM output not valid JSON; fallback to hard-coded rules.",
        }

    if debug:
        print("[analyst][llm] decision:", llm_json)

    return llm_json


async def analyst_agent(state: GlobalState, config: RunnableConfig) -> Dict[str, Any]:
    """
    analyst_agent：读取每个实验 csv 的头行汇总为 summary.csv，并调用 LLM 分析异常策略。
    """
    debug = bool(state.get("debug", False))

    run_id = state.get("run_id", "run_unknown")
    successes: List[ExamResult] = state.get("success_results", [])  # type: ignore
    user_intent = state.get("user_intent", "")

    _dbg(debug, f"[analyst] start, run_id={run_id}, successes={len(successes)}")
    _dbg(debug, f"[analyst] user_intent={user_intent}")

    rows: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []

    gen_brief(debug, successes, rows, anomalies)
    summary_path = save_brief(debug, rows, run_id)

    # ✅ async LLM analysis
    llm_decision = await analyze_data(debug, state, rows, anomalies, summary_path)
    # 你如果想把 llm_decision 带到 report 里，也可以加进去
    # 例如：report["llm_decision"] = llm_decision

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
