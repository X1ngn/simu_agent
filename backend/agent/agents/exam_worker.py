from __future__ import annotations

import os
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig

from backend.agent.state import ExamResult, ExperimentSpec, GlobalState
from backend.agent.tools.json_io import read_json, write_json, apply_dotpatch
from backend.agent.tools.sim_engine import run_simulation, SimulationError


SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "samples")


def _dbg(debug: bool, msg: str) -> None:
    if debug:
        print(msg)


def exam_worker(state: GlobalState, config: RunnableConfig) -> Dict[str, Any]:
    """
    单个 exam_agent worker 节点逻辑（ReAct 范式：读样例 -> 修改 -> 写文件 -> 调引擎 -> 返回结果）
    约定：由 designer 派发时在 state 里带入一个临时字段 current_experiment
    """
    debug = bool(state.get("debug", False))
    exp: ExperimentSpec = state["current_experiment"]  # type: ignore

    exp_id = exp["exp_id"]
    out_dir = exp["out_dir"]

    logs: List[str] = []
    def log(s: str) -> None:
        logs.append(s)
        _dbg(debug, f"[exam_worker:{exp_id}] {s}")

    try:
        log("start")
        sample_model = read_json(os.path.join(SAMPLES_DIR, "model.json"))
        sample_run = read_json(os.path.join(SAMPLES_DIR, "run.json"))
        sample_hw = read_json(os.path.join(SAMPLES_DIR, "hw.json"))
        log("loaded sample configs")

        model_cfg = apply_dotpatch(sample_model, exp.get("model_patch", {}))
        run_cfg = apply_dotpatch(sample_run, exp.get("run_patch", {}))
        hw_cfg = apply_dotpatch(sample_hw, exp.get("hw_patch", {}))
        log(f"applied patches: model={len(exp.get('model_patch', {}))}, "
            f"run={len(exp.get('run_patch', {}))}, hw={len(exp.get('hw_patch', {}))}")

        model_path = os.path.join(out_dir, "model.json")
        run_path = os.path.join(out_dir, "run.json")
        hw_path = os.path.join(out_dir, "hw.json")
        write_json(model_path, model_cfg)
        write_json(run_path, run_cfg)
        write_json(hw_path, hw_cfg)
        log("wrote patched configs")

        csv_path = os.path.join(out_dir, "result.csv")
        run_simulation(
            model_json_path=model_path,
            run_json_path=run_path,
            hw_json_path=hw_path,
            out_csv_path=csv_path,
            debug=debug,
        )
        log("simulation finished")

        result: ExamResult = {
            "exp_id": exp_id,
            "ok": True,
            "out_dir": out_dir,
            "model_path": model_path,
            "run_path": run_path,
            "hw_path": hw_path,
            "csv_path": csv_path,
            "logs": logs,
        }
        return {
            "exam_results": [result],
            "pending": -1,
        }


    except SimulationError as e:
        log(f"simulation failed: {e}")
        result: ExamResult = {
            "exp_id": exp_id,
            "ok": False,
            "out_dir": out_dir,
            "error": str(e),
            "logs": logs,
        }
        return {
            "exam_results": [result],
            "pending": -1,
        }


    except Exception as e:
        log(f"unexpected error: {repr(e)}")
        result: ExamResult = {
            "exp_id": exp_id,
            "ok": False,
            "out_dir": out_dir,
            "error": f"unexpected: {repr(e)}",
            "logs": logs,
        }
        return {
            "exam_results": [result],
            "pending": -1,
        }

