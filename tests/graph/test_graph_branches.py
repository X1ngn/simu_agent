import pytest

@pytest.mark.graph
def test_graph_covers_accept_edit_reject_and_analyst_fail():
    """
    TODO:
    - build_graph 传不同 stub 节点，覆盖：
      - accept：need_human -> dispatch/collect -> analyze -> ok -> END
      - edit：need_human -> edit -> need_redesign -> designer 再次生成
      - reject：need_human -> reject -> 终止
      - analyst not ok：analyst_report.ok=False -> need_redesign -> designer
    - 对每条路径用 trace 断言节点顺序/集合
    """
    pass
