from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Literal, Annotated
import operator

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ExperimentSpec(TypedDict):
    """designer 规划出的单个仿真任务"""
    exp_id: str
    description: str
    model_patch: Dict[str, Any]
    run_patch: Dict[str, Any]
    hw_patch: Dict[str, Any]


class ExamResult(TypedDict, total=False):
    exp_id: str
    ok: bool
    out_dir: str
    model_path: str
    run_path: str
    hw_path: str
    csv_path: str
    error: str
    logs: List[str]


class AnalystReport(TypedDict, total=False):
    ok: bool
    summary_csv_path: str
    anomalies: List[Dict[str, Any]]
    message: str


class AgentState(TypedDict, total=False):
    # 对话历史（核心，tool calling 的消息链全在这里）
    messages: Annotated[List[BaseMessage], add_messages]

    # 业务数据
    user_intent: str
    run_id: str
    debug: bool

    # plan_agent 输出
    experiments_plan: List[ExperimentSpec]

    # human_review 输出
    human_approved: bool

    # exec_agent 产生的结果
    experiments: List[ExperimentSpec]
    exam_results: List[ExamResult]

    # 最终报告
    analyst_report: AnalystReport

    # 长期记忆
    rolling_summary: Dict[str, Any]

    # 控制
    finished: bool


# Keep GlobalState as alias for backward compatibility during transition
GlobalState = AgentState
