import os
import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed


@pytest.mark.perf
def test_load_with_variable_concurrency():
    """
    负载 / 压力测试（stubbed 版）

    验证目标：
    - 并发数 / 请求数由环境变量控制
    - 默认只跑 stubbed 逻辑（不触真实 LLM / RPC）
    - 统计并打印：
        * 吞吐（QPS）
        * 错误率
        * P95 延迟
    """

    # ------------------------------------------------------------
    # 1. 从环境变量读取压测参数（有安全默认值）
    # ------------------------------------------------------------
    concurrency = int(os.getenv("LOAD_CONCURRENCY", "8"))
    requests = int(os.getenv("LOAD_REQUESTS", "50"))

    assert concurrency > 0
    assert requests > 0

    # ------------------------------------------------------------
    # 2. stubbed request（未来可替换为 graph.invoke / worker）
    # ------------------------------------------------------------
    def one_req():
        """
        模拟一次请求：
        - sleep 表示 I/O / 推理 / RPC 延迟
        - 返回 True 表示成功
        """
        start = time.time()
        time.sleep(0.01)
        end = time.time()
        return {
            "ok": True,
            "latency": end - start,
        }

    # ------------------------------------------------------------
    # 3. 并发执行请求
    # ------------------------------------------------------------
    t0 = time.time()

    results = []
    latencies = []

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(one_req) for _ in range(requests)]

        # 用 as_completed：不依赖完成顺序
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r["ok"])
                latencies.append(r["latency"])
            except Exception:
                # 任何异常都算一次失败
                results.append(False)

    t1 = time.time()

    # ------------------------------------------------------------
    # 4. 统计指标
    # ------------------------------------------------------------
    total_time = t1 - t0
    success = sum(1 for r in results if r)
    failures = len(results) - success

    throughput = requests / total_time if total_time > 0 else 0.0
    error_rate = failures / requests if requests > 0 else 0.0

    # 延迟分位数（毫秒）
    p95 = statistics.quantiles(latencies, n=20)[-1] if latencies else float("inf")

    # ------------------------------------------------------------
    # 5. 输出关键指标（perf 测试允许 print）
    # ------------------------------------------------------------
    print("\n=== Load Test Metrics ===")
    print(f"Concurrency     : {concurrency}")
    print(f"Requests        : {requests}")
    print(f"Total time (s)  : {total_time:.4f}")
    print(f"Throughput (QPS): {throughput:.2f}")
    print(f"Error rate      : {error_rate:.2%}")
    print(f"P95 latency (s) : {p95:.4f}")

    # ------------------------------------------------------------
    # 6. 断言（只做“底线”校验，不做硬性能门槛）
    # ------------------------------------------------------------
    assert all(results), "Some requests failed in stubbed load test"
    assert error_rate == 0.0
    assert throughput > 0
