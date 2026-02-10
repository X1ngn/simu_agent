from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Tuple, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from langgraph.graph import END

from backend.agent.state import GlobalState, ExperimentSpec, ExamResult
from backend.agent.tools.json_io import read_json
from backend.agent.llm_factory import get_llm_for_agent
from backend.agent.prompts.designer import designer_prompt
from backend.agent.memory import _get_mem_store


# -------------------------
# utils
# -------------------------

def _dbg(debug: bool, msg: str) -> None:
    if debug:
        print(msg)


def _make_run_id() -> str:
    return uuid.uuid4().hex[:8]


def _safe_json(obj: Any, max_chars: int = 8000) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) > max_chars:
        return s[: max_chars - 20] + "...(truncated)"
    return s


def _messages_tail(state: Dict[str, Any], n: int = 20) -> List[Dict[str, str]]:
    msgs = state.get("messages") or []
    tail = msgs[-n:]
    out: List[Dict[str, str]] = []
    for m in tail:
        out.append(
            {
                "type": getattr(m, "type", m.__class__.__name__),
                "content": getattr(m, "content", ""),
            }
        )
    return out


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    解决“结构化JSON后面跟自然语言”导致 json.loads 失败。
    1) 先整体 parse
    2) 失败则截取第一个 { 到最后一个 } 再 parse
    """
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        cand = text[i : j + 1]
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _dict_keys_topk(x: Any, k: int = 30) -> List[str]:
    """安全提取 dict 的 key 列表；非 dict 返回空。"""
    if not isinstance(x, dict):
        return []
    return [str(t) for t in list(x.keys())[:k]]


# -------------------------
# 0) summarize intent (LLM)  ✅ NEW (fixed)
# -------------------------

async def _llm_summarize_intent(
    *,
    llm: Any,
    system_prompt: str,
    state: Dict[str, Any],
    experiments_plan: List[ExperimentSpec],
    debug: bool,
) -> str:
    """
    在进入 need_human 前，把：
      - 初始 user_intent
      - 对话上下文（messages tail）
      - clarify_answers（如有）
      - 当前生成的 experiments_plan（仅做摘要参考）
    总结为“可长期记忆”的完整 user_intent，并写回 state["user_intent"]。

    返回：updated_user_intent（字符串）。失败则返回原 user_intent。
    """
    original_intent = str(state.get("user_intent") or "").strip()
    if not original_intent:
        return original_intent

    # 避免 prompt 膨胀：只给必要信息
    plan_compact: List[Dict[str, Any]] = []
    for e in (experiments_plan or [])[:12]:
        if not isinstance(e, dict):
            continue
        plan_compact.append(
            {
                "exp_id": e.get("exp_id"),
                "description": e.get("description"),
                "model_patch_keys": _dict_keys_topk(e.get("model_patch")),
                "run_patch_keys": _dict_keys_topk(e.get("run_patch")),   # ✅ FIX: [:30] 而不是 [:30,]
                "hw_patch_keys": _dict_keys_topk(e.get("hw_patch")),
            }
        )

    llm_req: Dict[str, Any] = {
        "task": "summarize_user_intent_for_long_term_memory",
        "rules": [
            "Return STRICT JSON ONLY. No markdown. No extra commentary.",
            "Output must be a JSON object with key: updated_user_intent (string).",
            "updated_user_intent must be a single concise but complete paragraph (Chinese).",
            "It must include: goal, controlled variables, sweep knobs + range (if known), hardware assumptions, failure boundary definition (if known), and any constraints mentioned by user.",
            "Do NOT include raw full experiment JSON; just summarize the intent and constraints.",
            "If some info is unknown, explicitly mark as '未指定'.",
        ],
        "context": {
            "original_user_intent": original_intent,
            "dialog_tail": _messages_tail(state, 30),
            "clarify_answers": state.get("clarify_answers", {}) or {},
            "last_rejection": state.get("last_rejection"),
            "last_failures": state.get("last_failures"),
            "experiments_plan_compact": plan_compact,
        },
        "output_schema": {"updated_user_intent": "string"},
    }

    try:
        resp = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(llm_req, ensure_ascii=False)),
            ]
        )
    except Exception as e:
        _dbg(debug, f"[designer][intent_summarize] ainvoke failed: {e}")
        return original_intent

    parsed = _extract_first_json_object(getattr(resp, "content", "") or "")
    if not parsed:
        _dbg(debug, f"[designer][intent_summarize] bad json, content={getattr(resp, 'content', '')[:300]}")
        return original_intent

    updated = str(parsed.get("updated_user_intent") or "").strip()
    if not updated:
        return original_intent

    # 轻量防抖：太短/明显无效则回退
    if len(updated) < max(20, len(original_intent) // 4):
        return original_intent

    if debug:
        _dbg(debug, f"[designer][intent_summarize] updated_user_intent={updated}")
    return updated


# -------------------------
# 1) clarify policy (LLM)
# -------------------------

async def _llm_need_clarify(
    *,
    llm: Any,
    system_prompt: str,
    user_intent: str,
    state: Dict[str, Any],
    memories: List[Dict[str, Any]],
    debug: bool,
) -> Dict[str, Any]:
    mem_compact: List[Dict[str, Any]] = []
    for m in (memories or [])[:5]:
        mem_compact.append(
            {
                "memory": m.get("memory"),
                "metadata": m.get("metadata", {}),
                "created_at": m.get("created_at"),
            }
        )

    llm_req: Dict[str, Any] = {
        "task": "clarify_user_intent_for_experiment_design",
        "rules": [
            "Return STRICT JSON ONLY. No markdown. No extra commentary.",
            "Output must be a JSON object with keys: need_more_info (bool), questions (list), notes (string).",
            "If need_more_info is false, questions must be an empty list.",
            "Each question must include: key, question, type. type in {text, choice, json}.",
            "If type is choice, include choices (list of strings).",
            "Ask the MINIMUM number of questions needed (typically 1-3).",
        ],
        "context": {
            "user_intent": user_intent,
            "dialog_tail": _messages_tail(state, 20),
            "clarify_answers": state.get("clarify_answers", {}) or {},
            "last_rejection": state.get("last_rejection"),
            "last_failures": state.get("last_failures"),
            "retrieved_memories": mem_compact,
            "hint": [
                "If hardware/model/runtime constraints are unclear, ask about them.",
                "If sweep variable is unclear, ask what to sweep and its range.",
                "If success metric/failure boundary definition is unclear, ask what to treat as failure.",
            ],
        },
        "output_schema": {
            "need_more_info": True,
            "questions": [
                {"key": "hardware", "question": "请提供硬件信息（GPU型号/数量）", "type": "text"},
            ],
            "notes": "short",
        },
    }

    resp = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(llm_req, ensure_ascii=False)),
        ]
    )
    parsed = _extract_first_json_object(getattr(resp, "content", "") or "")
    if not parsed:
        return {"need_more_info": False, "questions": [], "notes": "llm_output_unparseable"}

    need_more = bool(parsed.get("need_more_info", False))
    questions = parsed.get("questions") or []
    if not isinstance(questions, list):
        questions = []
    notes = str(parsed.get("notes") or "")

    norm_qs: List[Dict[str, Any]] = []
    for q in questions[:5]:
        if not isinstance(q, dict):
            continue
        key = str(q.get("key") or "").strip()
        question = str(q.get("question") or "").strip()
        qtype = str(q.get("type") or "text").strip()
        if not key or not question:
            continue
        if qtype not in ("text", "choice", "json"):
            qtype = "text"
        item: Dict[str, Any] = {"key": key, "question": question, "type": qtype}
        if qtype == "choice":
            choices = q.get("choices") or []
            if isinstance(choices, list):
                item["choices"] = [str(x) for x in choices if str(x).strip()]
        norm_qs.append(item)

    if not need_more:
        norm_qs = []

    out = {"need_more_info": need_more, "questions": norm_qs, "notes": notes}
    if debug:
        _dbg(debug, f"[designer][clarify] decision={_safe_json(out, 2000)}")
    return out


# -------------------------
# 2) generate experiments (LLM)
# -------------------------

async def gen_experiments(
    *,
    user_intent: str,
    run_id: str,
    debug: bool,
    state: Dict[str, Any],
    memories: List[Dict[str, Any]] | None = None,
) -> List[ExperimentSpec]:
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

    memories_compact: List[Dict[str, Any]] = []
    for m in (memories or [])[:5]:
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
            "If retrieved_memories or last_rejection/last_failures show bad patterns, avoid repeating them.",
            "Prefer simple sweep (1-2 knobs) + clear failure boundary probes.",
        ],
        "context": {
            "user_intent": user_intent,
            "dialog_tail": _messages_tail(state, 20),
            "clarify_answers": state.get("clarify_answers", {}) or {},
            "last_rejection": state.get("last_rejection"),
            "last_failures": state.get("last_failures"),
            "retrieved_memories": memories_compact,
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
        resp = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as e:
        _dbg(debug, f"[designer][llm] ainvoke failed: {e}")
        return _fallback_default()

    llm_json = _extract_first_json_object(getattr(resp, "content", "") or "")
    if not llm_json:
        _dbg(debug, f"[designer][llm] bad json (cannot parse), content={getattr(resp, 'content', '')[:300]}")
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


# -------------------------
# 3) designer node (async)
# -------------------------

def _split_results(results: List[ExamResult]) -> Tuple[List[ExamResult], List[ExamResult]]:
    succ, fail = [], []
    for r in results:
        if r.get("ok"):
            succ.append(r)
        else:
            fail.append(r)
    return succ, fail


async def designer_agent(state: GlobalState, config: RunnableConfig):
    debug = bool(state.get("debug", False))
    stage = state.get("stage", "init")
    retries = int(state.get("retries", 0) or 0)
    max_retries = int(state.get("max_retries", 1) or 1)

    _dbg(debug, f"[designer] enter stage={stage}, retries={retries}/{max_retries}")

    if stage in ("init", "design"):
        run_id = state.get("run_id") or _make_run_id()
        user_intent = str(state.get("user_intent") or "").strip()
        if not user_intent:
            raise ValueError("user_intent invalid")

        updates: Dict[str, Any] = {"run_id": run_id}

        if not state.get("_intent_logged"):
            updates["messages"] = [HumanMessage(content=user_intent)]
            updates["_intent_logged"] = True

        memories: List[Dict[str, Any]] = []
        store = _get_mem_store(config)
        if store is not None:
            try:
                memories = await asyncio.to_thread(
                    store.search_for_designer,
                    state=dict(state),
                    query=user_intent,
                    top_k=5,
                )
                _dbg(debug, f"[designer][mem0] retrieved memories: {len(memories)}")
            except Exception as e:
                _dbg(debug, f"[designer][mem0] search failed: {e}")

        llm = get_llm_for_agent("designer")

        clarify = await _llm_need_clarify(
            llm=llm,
            system_prompt=designer_prompt,
            user_intent=user_intent,
            state=dict(state),
            memories=memories,
            debug=debug,
        )

        if clarify.get("need_more_info"):
            questions = clarify.get("questions") or []
            updates["stage"] = "await_user"
            updates["clarify_questions"] = questions
            updates["clarify_notes"] = clarify.get("notes", "")

            q_text = "\n".join(
                [f"- ({q.get('key')}) {q.get('question')}" for q in questions if isinstance(q, dict)]
            )
            updates["messages"] = updates.get("messages", []) + [
                AIMessage(content=f"[CLARIFY] Need more info:\n{q_text}")
            ]
            return Command(goto=END, update=updates)

        experiments = await gen_experiments(
            user_intent=user_intent,
            run_id=run_id,
            debug=debug,
            state=dict(state),
            memories=memories,
        )

        updated_intent = await _llm_summarize_intent(
            llm=llm,
            system_prompt=designer_prompt,
            state=dict(state),
            experiments_plan=experiments,
            debug=debug,
        )

        updates.update(
            {
                "stage": "need_human",
                "run_id": run_id,
                "user_intent": updated_intent,
                "experiments_plan": experiments,
                "messages": updates.get("messages", []) + [
                    AIMessage(
                        content=(
                            f"[PLAN] generated {len(experiments)} experiments; "
                            f"user_intent consolidated; awaiting human review"
                        )
                    )
                ],
            }
        )
        return updates

    if stage == "dispatch_ready":
        run_id = state.get("run_id") or _make_run_id()
        user_intent = str(state.get("user_intent") or "").strip()
        if not user_intent:
            raise ValueError("user_intent invalid")

        experiments = state.get("experiments") or state.get("experiments_plan") or []
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("experiments invalid: nothing to dispatch")

        sends = [
            Send("exam_worker", {"current_experiment": exp, "debug": debug, "run_id": run_id})
            for exp in experiments
        ]

        _dbg(debug, f"[designer] dispatch {len(sends)} exam_worker jobs (human approved)")

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
                "messages": [AIMessage(content=f"[DISPATCH] {len(experiments)} workers launched")],
            },
            goto=sends,
        )

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
            fail_brief = {
                "failed_count": len(fail),
                "failed": [
                    {"exp_id": x.get("exp_id"), "reason": x.get("reason"), "message": x.get("message")}
                    for x in fail[:8]
                ],
            }
            if retries >= max_retries:
                return {
                    "stage": "need_redesign",
                    "failed_results": fail,
                    "success_results": succ,
                    "last_failures": fail_brief,
                    "messages": [AIMessage(content=f"[WORKER_FAIL] retries exhausted: {_safe_json(fail_brief, 2000)}")],
                }

            return {
                "stage": "design",
                "retries": retries + 1,
                "failed_results": fail,
                "success_results": succ,
                "last_failures": fail_brief,
                "messages": [AIMessage(content=f"[WORKER_FAIL] back to clarify/design: {_safe_json(fail_brief, 2000)}")],
            }

        return {
            "stage": "analyze",
            "failed_results": [],
            "success_results": succ,
            "messages": [AIMessage(content="[COLLECT_OK] all experiments succeeded -> analyze")],
        }

    if stage == "need_redesign":
        return {
            "stage": "done",
            "analyst_report": {
                "ok": False,
                "message": "Experiments failed and retries exhausted; please clarify constraints and redesign.",
                "anomalies": [{"reason": "designer_need_more_info"}],
            },
            "messages": [AIMessage(content="[DONE] need_redesign")],
        }

    return {"stage": "done"}
