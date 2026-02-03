from __future__ import annotations

import os
import random
import time
from typing import Dict, Any

from .csv_io import write_csv
from .json_io import read_json


class SimulationError(RuntimeError):
    pass


def run_simulation(
    model_json_path: str,
    run_json_path: str,
    hw_json_path: str,
    out_csv_path: str,
    debug: bool = False,
) -> str:
    """
    这里是仿真引擎工具的“适配层”：
    - 真实情况：你会在这里调用你已有的仿真引擎（Python API / CLI / RPC 都可以）
    - 当前实现：用 dummy 数据生成 CSV，模拟成功/失败
    """
    if debug:
        print(f"[sim_engine] model={model_json_path}")
        print(f"[sim_engine] run  ={run_json_path}")
        print(f"[sim_engine] hw   ={hw_json_path}")
        print(f"[sim_engine] out  ={out_csv_path}")

    model = read_json(model_json_path)
    run = read_json(run_json_path)
    hw = read_json(hw_json_path)

    # ---- 模拟运行耗时 ----
    time.sleep(0.05)

    # ---- 生成 dummy 指标：吞吐、显存、step_time 等 ----
    tp = int(run.get("tp_size", 1) or 1)
    pp = int(run.get("pp_size", 1) or 1)
    dp = int(run.get("dp_size", 1) or 1)
    cp = int(run.get("cp_size", 1) or 1)
    ep = int(run.get("ep_size", 1) or 1)
    gbs = int(run.get("gbs", 16) or 16)
    mbs = int(run.get("mbs", 1) or 1)
    
    hidden = int(model.get("arch", {}).get("hidden_size", 4096) or 4096)
    num_layers = int(sum([layer.get("repeat_number", 32) for layer in model.get("layer_order", {})]) or 32)
    mem = hw.get("mem1", {}).get("bandwidth_gbps", 0)

    base = (hidden * num_layers) / 1e6
    throughput = (dp * tp) * (2000.0 / (1.0 + base)) * random.uniform(0.9, 1.1)
    step_time_ms = 1000.0 / max(throughput, 1e-6)
    peak_mem_gb = (base * 0.8 + gbs * 0.002) * random.uniform(0.9, 1.15)

    if peak_mem_gb > mem:
        raise SimulationError(f"peak_mem_gb={peak_mem_gb:.2f}GB exceeds hw mem={mem}GB -> OOM (dummy)")

    rows = [
        {
            "tp": tp,
            "pp": pp,
            "cp": cp,
            "dp": dp,
            "ep": ep,
            "gbs": gbs,
            "mbs": mbs,
            "hidden": hidden,
            "layers": num_layers,
            "total_time_s": f"{step_time_ms * 1000:.3f}",
            "throughput_tokens_per_s": f"{throughput:.3f}",
            "mem": mem,
            "peak_mem_gb": f"{peak_mem_gb:.3f}",
            "status": "ok",
        },
        # 你真实引擎可能会输出更多行（不同候选配置/不同 kernel 选择等）
    ]

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    write_csv(out_csv_path, rows)

    return out_csv_path
