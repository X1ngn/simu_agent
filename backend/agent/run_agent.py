from __future__ import annotations

import os
from typing import Any, Dict

from graph import build_graph


def main():
    # debug 开关
    debug = True

    # 你也可以把 user_intent 换成从 CLI / Web / API 输入
    user_intent = "固定模型与硬件，扫 global_batch_size 对吞吐与显存峰值的影响，并找出失败边界。"

    app = build_graph()

    init_state: Dict[str, Any] = {
        "debug": debug,
        "stage": "init",
        "user_intent": user_intent,
        "retries": 0,
        "max_retries": 1,
    }

    # thread_id 用于 checkpoint / 多会话并行
    config = {"configurable": {"thread_id": "demo_thread"}}

    final_state = app.invoke(init_state, config=config)

    print("\n=== FINAL REPORT ===")
    report = final_state.get("analyst_report", {})
    for k, v in report.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    # 确保 data/runs 存在
    base = os.path.join(os.path.dirname(__file__), "data", "runs")
    os.makedirs(base, exist_ok=True)
    main()
