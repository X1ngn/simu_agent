from __future__ import annotations

import asyncio
import os
import uuid
import json
from typing import Any, Dict, List, Tuple

from backend.agent.state import GlobalState, ExperimentSpec, ExamResult
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


async def gen_experiments(
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

    # ✅ async llm call
    try:
        resp = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as e:
        _dbg(debug, f"[designer][llm] ainvoke failed: {e}")
        return _fallback_default()

    # ---- parse JSON ----
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

        # exp_id 规范化：只保留安全字符（防止路径/奇怪字符）
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


async def designer_agent(state: GlobalState, config: RunnableConfig):
    """
    designer_agent：
    1) 设计实验（LLM）
    2) HITL：先交给人审（need_human）
    3) 人类确认后 dispatch_ready -> Send 分发 exam_worker
    4) collect fan-in
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
        user_intent = str(state.get("user_intent") or "").strip()
        if not user_intent:
            raise ValueError("user_intent invalid")

        # ✅ 从 DI 取 mem_store，检索记忆（best-effort）
        memories: List[Dict[str, Any]] = []
        store = _get_mem_store(config)
        if store is not None:
            try:
                # store.search_for_designer 大概率是同步的 → to_thread 防止阻塞 event loop
                memories = await asyncio.to_thread(
                    store.search_for_designer,
                    state=dict(state),
                    query=user_intent,
                    top_k=5,
                )
                _dbg(debug, f"[designer][mem0] retrieved memories: {len(memories)}")
            except Exception as e:
                _dbg(debug, f"[designer][mem0] search failed: {e}")

        experiments = await gen_experiments(user_intent, run_id, debug, memories=memories)

        _dbg(debug, f"[designer] planned {len(experiments)} experiments; wait human review")

        return {
            "stage": "need_human",
            "run_id": run_id,
            "user_intent": user_intent,
            "experiments_plan": experiments,
        }

    # ------------------------------------------------------------------
    # (2) dispatch_ready：人类确认后，才执行 Send 分发逻辑
    # ------------------------------------------------------------------
    if stage == "dispatch_ready":
        run_id = state.get("run_id") or _make_run_id()
        user_intent = str(state.get("user_intent") or "").strip()
        if not user_intent:
            raise ValueError("user_intent invalid")

        experiments = state.get("experiments") or state.get("experiments_plan") or []
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("experiments invalid: nothing to dispatch")

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

        from langgraph.types import Command

        return Command(
            update={
                "stage": "collect",
                "run_id": run_id,
                "user_intent": user_intent,
                "experiments": experiments,
                "pending": len(experiments),
                "exam_results": [],
                "failed_results": [],
                "success_results": [],
            },
            goto=sends,
        )

    # ------------------------------------------------------------------
    # (3) collect：fan-in 聚合
    # ------------------------------------------------------------------
    if stage == "collect":
        pending = int(state.get("pending", 0) or 0)
        results = state.get("exam_results", []) or []
        _dbg(debug, f"[designer] fan-in pending={pending} results_len={len(results)}")

        if pending > 0:
            return {}

        results2: List[ExamResult] = state.get("exam_results", [])  # type: ignore
        succ, fail = _split_results(results2)
        _dbg(debug, f"[designer] collected results: success={len(succ)}, fail={len(fail)}")

        if fail:
            if retries < max_retries:
                _dbg(debug, "[designer] retry enabled -> redesign failed exps and re-dispatch")

                new_exps: List[ExperimentSpec] = []
                for r in fail:
                    exp_id = r["exp_id"]
                    orig = next(
                        (e for e in (state.get("experiments", []) or []) if e.get("exp_id") == exp_id),
                        None,
                    )
                    if not orig:
                        continue
                    patched = dict(orig)
                    patched["exp_id"] = exp_id + "_retry"
                    patched["description"] = (orig.get("description", "") + " (retry)")
                    patched_run_patch = dict(orig.get("run_patch", {}))
                    patched_run_patch["train.global_batch_size"] = 768
                    patched["run_patch"] = patched_run_patch

                    base_out = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "data", "runs", state["run_id"])
                    )
                    patched["out_dir"] = os.path.join(base_out, patched["exp_id"])
                    new_exps.append(patched)  # type: ignore

                from langgraph.types import Send, Command  # noqa

                sends = [
                    Send(
                        "exam_worker",
                        {"current_experiment": exp, "debug": debug, "run_id": state["run_id"]},
                    )
                    for exp in new_exps
                ]

                return Command(
                    update={
                        "stage": "collect",
                        "retries": retries + 1,
                        "failed_results": fail,
                        "success_results": succ,
                        "experiments": (state.get("experiments", []) or []) + new_exps,
                        "pending": len(new_exps),
                    },
                    goto=sends,
                )

            _dbg(debug, "[designer] retries exhausted -> need_redesign")
            return {
                "stage": "need_redesign",
                "failed_results": fail,
                "success_results": succ,
            }

        return {
            "stage": "analyze",
            "failed_results": [],
            "success_results": succ,
        }

    # ------------------------------------------------------------------
    # (4) need_redesign：兜底
    # ------------------------------------------------------------------
    if stage == "need_redesign":
        return {
            "stage": "done",
            "analyst_report": {
                "ok": False,
                "message": "Experiments failed and retries exhausted; please confirm variable fields / constraints.",
                "anomalies": [{"reason": "designer_need_more_info"}],
            },
        }

    return {"stage": "done"}
