import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import pytest

from langchain_core.runnables import RunnableConfig

from backend.agent.graph import build_graph
from backend.agent.run_agent import handle_hitl_interrupts, HitlDecision

from tests.helpers.trace import TraceRecorder, wrap_node


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("inf")
    xs = sorted(values)
    idx = int(p * (len(xs) - 1))
    return xs[idx]


def _print_node_breakdown(trace: TraceRecorder) -> None:
    print("\n=== Node Latency Breakdown (ordered) ===")
    for c in trace.nodes:
        print(f"{c.node:<14} {c.duration * 1000:>9.2f} ms")

    agg = defaultdict(list)
    for c in trace.nodes:
        agg[c.node].append(c.duration)

    print("\n=== Node Latency Summary (per node) ===")
    for node, durs in agg.items():
        avg = sum(durs) / len(durs)
        mx = max(durs)
        print(
            f"{node:<14} count={len(durs):>2} avg={avg*1000:>8.2f}ms max={mx*1000:>8.2f}ms"
        )


class DummyMemStore:
    """
    ✅ 用于通过 human_review_node 的 mem_store 强制检查
    你真实 mem_store 可能还会被 designer.search_for_designer 用到，所以也提供该方法。
    """

    def search_for_designer(self, state: Dict[str, Any], query: str, top_k: int = 5):
        return []

    def __getattr__(self, name: str):
        # 兜底：human_review/其他模块如果调用 store.xxx(...)，不至于崩
        def _noop(*args, **kwargs):
            return None

        return _noop


@pytest.mark.e2e_judge
@pytest.mark.skipif(
    os.getenv("RUN_E2E_LATENCY_REAL") != "1",
    reason="Real E2E latency disabled by default",
)
def test_e2e_latency_breakdown_real_system():
    """
    真实系统端到端时延（含逐节点耗时拆解）：
    - 使用真实 designer/human_review/worker/analyst
    - 用 decider 自动 accept，跑完整链路（直到图自然结束或达到你的终止条件）
    - 统计 total latency + per-node breakdown

    环境变量：
      RUN_E2E_LATENCY_REAL=1      开启
      E2E_LAT_RUNS=3              跑几次
      E2E_LAT_PRINT_EACH=1        是否每次打印 breakdown
      E2E_LAT_P95_THRESHOLD_S=0   可选阈值（秒）
    """

    runs = int(os.getenv("E2E_LAT_RUNS", "3"))
    print_each = os.getenv("E2E_LAT_PRINT_EACH", "1") == "1"
    threshold_p95_s = float(os.getenv("E2E_LAT_P95_THRESHOLD_S", "0"))

    # ---- 真实节点（按你项目路径）
    from backend.agent.agents.designer import designer_agent
    from backend.agent.agents.human_review import human_review_node
    from backend.agent.agents.exam_worker import exam_worker
    from backend.agent.agents.analyst import analyst_agent

    trace = TraceRecorder()

    app = build_graph(
        designer=wrap_node(trace, "designer", designer_agent),
        human_review=wrap_node(trace, "human_review", human_review_node),
        worker=wrap_node(trace, "exam_worker", exam_worker),
        analyst=wrap_node(trace, "analyst", analyst_agent),
    )

    def decider_accept(payload: Dict[str, Any], state: Dict[str, Any]) -> HitlDecision:
        return HitlDecision(action="accept")

    total_latencies: List[float] = []
    per_node_latencies = defaultdict(list)

    for i in range(runs):
        trace.nodes.clear()
        trace.tools.clear()

        init_state = {
            "debug": False,
            "stage": "init",
            "user_intent": "固定模型与硬件，扫 global_batch_size 对吞吐与显存峰值的影响，并找出失败边界。",
            "retries": 0,
            "max_retries": 1,
        }

        mem_store = DummyMemStore()

        # ✅ 关键：用 RunnableConfig，不要用 dict
        config = RunnableConfig(
            configurable={
                "thread_id": f"e2e_latency_real_{i}",
                "mem_store": mem_store,
            }
        )

        # ✅ 关键：无论 LangGraph 是否注入 config，wrap_node 都会兜底用 trace.current_config
        trace.current_config = config

        t0 = time.perf_counter()
        s = app.invoke(init_state, config=config)

        # handle_hitl_interrupts 内部还会 app.invoke(...)，再次保证 current_config 可用
        trace.current_config = config
        final = handle_hitl_interrupts(app, s, config=config, decider=decider_accept)

        t1 = time.perf_counter()

        total = t1 - t0
        total_latencies.append(total)

        for c in trace.nodes:
            per_node_latencies[c.node].append(c.duration)

        print(f"\n--- Run {i+1}/{runs} total_latency={total:.4f}s ---")
        if print_each:
            _print_node_breakdown(trace)

    avg_total = sum(total_latencies) / len(total_latencies)
    p95_total = _percentile(total_latencies, 0.95)
    mx_total = max(total_latencies)

    print("\n=== REAL E2E Total Latency Metrics ===")
    print(f"runs            : {runs}")
    print(f"avg latency (s) : {avg_total:.4f}")
    print(f"p95 latency (s) : {p95_total:.4f}")
    print(f"max latency (s) : {mx_total:.4f}")

    print("\n=== REAL E2E Node Latency Metrics (aggregated) ===")
    for node, durs in per_node_latencies.items():
        avg = sum(durs) / len(durs)
        p95 = _percentile(durs, 0.95)
        mx = max(durs)
        print(
            f"{node:<14} count={len(durs):>3} avg={avg*1000:>8.2f}ms p95={p95*1000:>8.2f}ms max={mx*1000:>8.2f}ms"
        )

    if threshold_p95_s > 0:
        assert p95_total <= threshold_p95_s, (
            f"P95 total latency {p95_total:.4f}s exceeds threshold {threshold_p95_s:.4f}s"
        )

    # 最基本的 sanity：至少跑到 human_review
    assert any(c.node == "human_review" for c in trace.nodes), "human_review not executed"
