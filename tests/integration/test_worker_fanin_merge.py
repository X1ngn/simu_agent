# tests/integration/test_worker_fanin_merge.py
import pytest

@pytest.mark.integration
def test_worker_fanin_merge(app, init_state, config, fakes):
    """
    TODO:
    - 把 FakeDesigner 改成脚本化：
      1) 第一次 need_human
      2) accept 后 dispatch（如果你支持 Send 多 worker，就发多个）
      3) worker 返回 collect（多次）
      4) designer 聚合后 stage=analyze -> analyst
    - 断言：
      - exam_results 聚合齐全
      - 乱序返回不会影响最终聚合正确性
      - 如果要求幂等：重复返回不会重复累计
    """
    pass
