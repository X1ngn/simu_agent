from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Literal, Annotated
import operator

def merge_list(old, new):
    old = old or []
    new = new or []
    return old + new



Stage = Literal[
    "init",
    "design",
    "dispatch",
    "collect",
    "analyze",
    "done",
    "need_redesign",
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


class GlobalState(TypedDict, total=False):
    # user intent
    user_intent: str

    # runtime
    debug: bool
    stage: Stage
    run_id: str

    # designer output
    experiments: List[ExperimentSpec]

    # exam aggregation (map-reduce)
    pending: Annotated[int, operator.add]
    exam_results: Annotated[List[ExamResult], operator.add] # reducer 聚合 worker 输出
    failed_results: List[ExamResult]
    success_results: List[ExamResult]

    # analyst output
    analyst_report: AnalystReport

    # control
    retries: int
    max_retries: int
