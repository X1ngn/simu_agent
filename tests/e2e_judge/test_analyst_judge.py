import os
import pytest

@pytest.mark.e2e_judge
@pytest.mark.skipif(os.getenv("RUN_E2E_JUDGE") != "1", reason="E2E judge disabled by default")
def test_analyst_report_consistent_with_results_and_intent():
    """
    TODO:
    - 给定 exam_results + user_intent，运行 analyst 得到 report
    - judge 检查 report 是否引用关键事实（OOM 边界、吞吐趋势）且结论一致
    """
    pass
