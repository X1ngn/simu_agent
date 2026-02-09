import pytest
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


@pytest.mark.perf
def test_stubbed_latency_concurrency():
    """
    性能 / 并发正确性测试（stub 级）：

    验证点：
    1) 不同延迟的任务可以并发执行
    2) 所有任务都会返回（无丢失、无死锁）
    3) 结果聚合不依赖完成顺序
    """

    # ------------------------------------------------------------
    # 1. 为“worker 内部工具”准备不同的 delay（模拟 I/O / RPC / 推理耗时）
    # ------------------------------------------------------------
    delays = [0.05, 0.2, 0.1, 0.3]

    def task(i: int) -> int:
        """
        stubbed worker / tool：
        - 用 sleep 模拟不同执行时长
        - 返回自身 index，方便后面校验是否全部返回
        """
        time.sleep(delays[i])
        return i

    # ------------------------------------------------------------
    # 2. 并发执行（max_workers >= 任务数，确保真正并发）
    # ------------------------------------------------------------
    start = time.time()

    with ThreadPoolExecutor(max_workers=len(delays)) as ex:
        futures = [ex.submit(task, i) for i in range(len(delays))]

        # as_completed：按“完成顺序”返回 future
        results = [f.result() for f in as_completed(futures)]

    elapsed = time.time() - start

    # ------------------------------------------------------------
    # 3. 断言：所有任务都返回（不依赖顺序）
    # ------------------------------------------------------------
    assert set(results) == set(range(len(delays))), (
        f"Expected results {set(range(len(delays)))} "
        f"but got {set(results)}"
    )

    # ------------------------------------------------------------
    # 4. （可选但很有价值）时间级断言：确实是并发而不是串行
    # ------------------------------------------------------------
    # 如果是串行执行，总耗时应接近 sum(delays)
    # 并发执行时，总耗时应接近 max(delays)
    assert elapsed < sum(delays) * 0.8, (
        f"Execution looks serial: elapsed={elapsed:.3f}s, "
        f"sum(delays)={sum(delays):.3f}s"
    )
