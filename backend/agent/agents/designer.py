from __future__ import annotations

import os
import uuid
from typing import List, Tuple

from backend.agent.state import GlobalState, ExperimentSpec, ExamResult

from pathlib import Path
import json
from typing import Any, Dict, List

from backend.agent.tools.json_io import read_json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from backend.agent.llm_factory import get_llm_for_agent
from backend.agent.prompts.designer import designer_prompt
from backend.agent.memory import _get_mem_store


def _dbg(debug: bool, msg: str) -> None:
    if debug:
        print(msg)


def _make_run_id() -> str:
    return uuid.uuid4().hex[:8]


def gen_experiments(
    user_intent: str,
    run_id: str,
    debug: bool,
    memories: List[Dict[str, Any]] | None = None,
) -> List[ExperimentSpec]:
    """
    使用 LLM 根据 user_intent 生成实验设计（List[ExperimentSpec]）。
    - memories：由外部检索得到的历史记忆（accept/edit/reject），用于给 LLM 提示避免重复错误。
    """
    base_out = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "runs", run_id)
    )

    def _fallback_default() -> List[ExperimentSpec]:
        raise RuntimeError("llm generate experiments failed")

    llm = get_llm_for_agent("designer")
    system_prompt = designer_prompt
    SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "samples")
    sample_model = read_json(os.path.join(SAMPLES_DIR, "model.json"))
    sample_run = read_json(os.path.join(SAMPLES_DIR, "run.json"))
    sample_hw = read_json(os.path.join(SAMPLES_DIR, "hw.json"))

    # memories 只压缩关键字段（避免 prompt 膨胀）
    memories_compact: List[Dict[str, Any]] = []
    if memories:
        for m in memories[:5]:
            memories_compact.append(
                {
                    "memory": m.get("memory"),
                    "metadata": m.get("metadata", {}),
                    "created_at": m.get("created_at"),
                }
            )

    llm_req: Dict[str, Any] = {
        "task": "design_experiments",
        "rules": [
            "Return STRICT JSON ONLY. No markdown. No commentary.",
            "Output must be an object with key 'experiments'.",
            "Each experiment must include: exp_id, description, model_patch, run_patch, hw_patch.",
            "Patches are dicts. run_patch uses dot-keys like 'run.gbs'.",
            "Do NOT include 'out_dir' in experiments; the caller will attach it.",
            "Do NOT reference files. Do NOT assume paths.",
            "Keep experiments count between 3 and 12 unless user explicitly asks otherwise.",
            "If retrieved_memories show plans were rejected or edited, avoid repeating those incorrect patterns.",
        ],
        "context": {
            "user_intent": user_intent,
            "retrieved_memories": memories_compact,
            "experiment_goal_hint": "Typical goal: control variables; sweep one or two knobs; detect failure boundary.",
            "allowed_patch_scopes": {
                "model_patch": "model config overrides (dict)",
                "run_patch": "runtime/training config overrides using dot-keys (dict)",
                "hw_patch": "hardware config overrides (dict)",
            },
            "could_patch_keys": [
                {"model": sample_model},
                {"run": sample_run},
                {"hw": sample_hw},
            ],
            "base_out_dir": base_out,
        },
        "output_schema": {
            "experiments": [
                {
                    "exp_id": "string (unique, short, no spaces)",
                    "description": "string",
                    "model_patch": {},
                    "run_patch": {},
                    "hw_patch": {},
                }
            ]
        },
    }

    prompt = json.dumps(llm_req, ensure_ascii=False)

    try:
        resp = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as e:
        _dbg(debug, f"[designer][llm] invoke failed: {e}")
        return _fallback_default()

    try:
        llm_json = json.loads(resp.content)
    except Exception as e:
        _dbg(debug, f"[designer][llm] bad json: {e}, content={resp.content[:300]}")
        return _fallback_default()

    _dbg(debug, f"[designer][llm] raw={str(llm_json)[:800]}")

    exps_raw = llm_json.get("experiments")
    if not isinstance(exps_raw, list) or not exps_raw:
        _dbg(debug, "[designer][llm] 'experiments' missing or empty -> fallback")
        return _fallback_default()

    normalized: List[ExperimentSpec] = []
    seen_ids: set[str] = set()

    def _is_dict(x: Any) -> bool:
        return isinstance(x, dict)

    for idx, e in enumerate(exps_raw, start=1):
        if not isinstance(e, dict):
            continue

        exp_id = str(e.get("exp_id") or "").strip()
        desc = str(e.get("description") or "").strip()

        model_patch = e.get("model_patch", {})
        run_patch = e.get("run_patch", {})
        hw_patch = e.get("hw_patch", {})

        if not exp_id:
            exp_id = f"exp_{idx:02d}"

        safe = []
        for ch in exp_id:
            if ch.isalnum() or ch in ("_", "-", "."):
                safe.append(ch)
            else:
                safe.append("_")
        exp_id = "".join(safe)

        if exp_id in seen_ids:
            exp_id = f"{exp_id}_{idx:02d}"
        seen_ids.add(exp_id)

        if not desc:
            desc = f"Experiment {exp_id}"

        if not _is_dict(model_patch) or not _is_dict(run_patch) or not _is_dict(hw_patch):
            _dbg(debug, f"[designer][llm] patch type invalid for {exp_id} -> skip")
            continue

        out_dir = os.path.join(base_out, exp_id)

        normalized.append(
            {
                "exp_id": exp_id,
                "description": desc,
                "model_patch": model_patch,
                "run_patch": run_patch,
                "hw_patch": hw_patch,
                "out_dir": out_dir,
            }
        )

    if not normalized:
        _dbg(debug, "[designer][llm] no valid experiments after normalize -> fallback")
        return _fallback_default()

    _dbg(debug, f"[designer][llm] experiments created: {len(normalized)}")
    return normalized



def _split_results(results: List[ExamResult]) -> Tuple[List[ExamResult], List[ExamResult]]:
    succ, fail = [], []
    for r in results:
        if r.get("ok"):
            succ.append(r)
        else:
            fail.append(r)
    return succ, fail


def designer_agent(state: GlobalState, config: RunnableConfig):
    """
    designer_agent：
    1) 设计实验（plan-and-solve + react）
    2) 动态派发 N 个 exam_worker 并发执行
    3) 收敛结果，决定：失败则分析&可能重试；成功则进入 analyst

    HITL 改造（最小侵入）：
    - init/design：只生成 experiments -> stage=need_human（不立刻 Send）
    - dispatch_ready：人类确认后才真正 Send 分发 -> stage=dispatch
    """
    debug = bool(state.get("debug", False))

    stage = state.get("stage", "init")
    retries = int(state.get("retries", 0) or 0)
    max_retries = int(state.get("max_retries", 1) or 1)

    _dbg(debug, f"[designer] enter stage={stage}, retries={retries}/{max_retries}")

    # ------------------------------------------------------------------
    # (1) init/design：生成实验，但先交给人审（不分发）
    # ------------------------------------------------------------------
    if stage in ("init", "design"):
        run_id = state.get("run_id") or _make_run_id()
        user_intent = state.get("user_intent", "").strip()
        if not user_intent:
            raise ValueError("user_intent invalid")

        # ✅ 从 DI 取 mem_store，检索记忆（best-effort）
        memories: List[Dict[str, Any]] = []
        store = _get_mem_store(config)
        if store is not None:
            # try:
                memories = store.search_for_designer(state=dict(state), query=user_intent, top_k=5)
                _dbg(debug, f"[designer][mem0] retrieved memories: {len(memories)}")
            # except Exception as e:
            #     _dbg(debug, f"[designer][mem0] search failed: {e}")

        experiments = gen_experiments(user_intent, run_id, debug, memories=memories)

        _dbg(debug, f"[designer] planned {len(experiments)} experiments; wait human review")

        return {
            "stage": "need_human",
            "run_id": run_id,
            "user_intent": user_intent,
            "experiments_plan": experiments,
        }

    # ------------------------------------------------------------------
    # (2) dispatch_ready：人类确认后，才执行你原本的 Send 分发逻辑
    # ------------------------------------------------------------------
    if stage == "dispatch_ready":
        run_id = state.get("run_id") or _make_run_id()
        user_intent = state.get("user_intent", "").strip()
        if not user_intent:
            raise ValueError("user_intent invalid")

        experiments = state.get("experiments") or state.get("experiments_plan") or []
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("experiments invalid: nothing to dispatch")

        # LANGGRAPH_ADJUST_HERE:
        # 返回 Send 列表以动态创建 N 个 exam_worker 并发任务
        from langgraph.types import Send  # noqa

        sends = []
        for exp in experiments:
            sends.append(
                Send(
                    "exam_worker",
                    {
                        "current_experiment": exp,
                        "debug": debug,
                        "run_id": run_id,
                    },
                )
            )

        _dbg(debug, f"[designer] dispatch {len(sends)} exam_worker jobs (human approved)")

        from langgraph.types import Command, Send  # 放到文件顶部或函数内部都可以

        return Command(
            update={
                "stage": "collect",
                "run_id": run_id,
                "user_intent": user_intent,
                "experiments": experiments,  # ✅ 固化已批准的 experiments
                "pending": len(experiments),
                "exam_results": [],
                "failed_results": [],
                "success_results": [],
            },
            goto=sends,
        )

    # stage: collect -> aggregate results and decide next
    if stage == "collect":
        pending = int(state.get("pending", 0) or 0)
        results = state.get("exam_results", []) or []
        _dbg(debug, f"[designer] fan-in pending={pending} results_len={len(results)}")

        if pending > 0:
            # 还有 worker 没回来：别做 collect，直接结束本 tick，等下一个 worker 触发
            return {}  # 或者干脆 return {} 也可以

        results: List[ExamResult] = state.get("exam_results", [])  # type: ignore
        succ, fail = _split_results(results)

        _dbg(debug, f"[designer] collected results: success={len(succ)}, fail={len(fail)}")

        # 如果失败：分析原因，决定是否重试/重设计
        if fail:
            # 简单策略：如果还没超过 max_retries，就把失败的实验去掉危险参数重跑
            if retries < max_retries:
                _dbg(debug, "[designer] retry enabled -> redesign failed exps and re-dispatch")

                # 例：把 gbs>=1024 的失败任务改成 768 重新跑
                # TODO: 使用llm更新实验策略
                new_exps: List[ExperimentSpec] = []
                for r in fail:
                    exp_id = r["exp_id"]
                    # 从原 experiments 找到 spec
                    orig = next((e for e in state.get("experiments", []) if e["exp_id"] == exp_id), None)  # type: ignore
                    if not orig:
                        continue
                    patched = dict(orig)
                    patched["exp_id"] = exp_id + "_retry"
                    patched["description"] = (orig.get("description", "") + " (retry)")
                    # 强行降 gbs
                    patched_run_patch = dict(orig.get("run_patch", {}))
                    patched_run_patch["train.global_batch_size"] = 768
                    patched["run_patch"] = patched_run_patch
                    # 新输出目录
                    base_out = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "data", "runs", state["run_id"])
                    )
                    patched["out_dir"] = os.path.join(base_out, patched["exp_id"])
                    new_exps.append(patched)  # type: ignore

                from langgraph.types import Send  # noqa
                sends = [
                    Send(
                        "exam_worker",
                        {"current_experiment": exp, "debug": debug, "run_id": state["run_id"]},
                    )
                    for exp in new_exps
                ]

                from langgraph.types import Command, Send

                return Command(
                    update={
                        "stage": "dispatch",
                        "retries": retries + 1,
                        "failed_results": fail,
                        "success_results": succ,
                        "experiments": state.get("experiments", []) + new_exps,
                        "pending": len(new_exps),
                    },
                    goto=sends,
                )

            # 超过重试次数：进入 need_redesign
            _dbg(debug, "[designer] retries exhausted -> need_redesign")
            return {
                "stage": "need_redesign",
                "failed_results": fail,
                "success_results": succ,
            }

        # 全部成功：进入 analyst
        return {
            "stage": "analyze",
            "failed_results": [],
            "success_results": succ,
        }

    # stage: need_redesign -> 在这里你可以实现更复杂的“向用户追问/重新规划”
    if stage == "need_redesign":
        # TODO：待实现真正的整合错误信息、向用户追问、重新规划功能，当前逻辑直接返回
        return {
            "stage": "done",
            "analyst_report": {
                "ok": False,
                "message": "Experiments failed and retries exhausted; please confirm variable fields / constraints.",
                "anomalies": [{"reason": "designer_need_more_info"}],
            },
        }

    # 默认兜底
    return {"stage": "done"}
