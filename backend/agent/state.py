from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Literal, Annotated
import operator

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


Stage = Literal[
    "init",
    "design",
    "dispatch",
    "collect",
    "analyze",
    "done",
    "need_redesign",
    "await_user",
]


class ExperimentSpec(TypedDict):
    """designer 规划出的单个仿真任务"""
    exp_id: str
    description: str
    # 对三个样例配置分别做 patch（点路径/JSONPath 简化为 dot-path）
    model_patch: Dict[str, Any]
    run_patch: Dict[str, Any]
    hw_patch: Dict[str, Any]
    # 输出目录（exam_worker 会在这里写 json 与 csv）
    out_dir: str


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


Stage = Literal[
    "init",
    "design",
    "need_user",
    "need_human",
    "dispatch_ready",
    "collect",
    "need_redesign",
    "analyze",
    "done",
]

class ClarifyQuestion(TypedDict, total=False):
    key: str
    question: str
    type: Literal["text", "choice", "json"]
    choices: List[str]

class GlobalState(TypedDict, total=False):
    # user intent
    user_intent: str

    # runtime
    debug: bool
    stage: Stage
    run_id: str

    # ✅ 对话上下文（用于 clarify / redesign / 失败归因）
    messages: Annotated[List[BaseMessage], add_messages]

    # ✅ 澄清子范式的结构化槽位（避免把 key/value 全塞进自然语言）
    clarify_questions: List[ClarifyQuestion]
    clarify_answers: Dict[str, Any]

    # ✅ 让“reject / fail -> 回到第一步”可追踪
    last_rejection: Dict[str, Any]
    last_failures: Dict[str, Any]

    _intent_logged: bool

    # designer output
    experiments_plan: List[ExperimentSpec]
    experiments: List[ExperimentSpec]

    # exam aggregation (map-reduce)
    pending: Annotated[int, operator.add]
    exam_results: Annotated[List[ExamResult], operator.add]
    failed_results: List[ExamResult]
    success_results: List[ExamResult]

    # analyst output
    analyst_report: AnalystReport

    # control
    retries: int
    max_retries: int
