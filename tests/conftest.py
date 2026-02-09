import os, sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.agent.graph import build_graph
from tests.helpers.trace import TraceRecorder, wrap_node
from tests.helpers import stubs

@pytest.fixture
def config(request):
    return {"configurable": {"thread_id": request.node.name}}

@pytest.fixture
def init_state():
    return {
        "debug": True,
        "stage": "init",
        "user_intent": "固定模型与硬件，扫 global_batch_size 对吞吐与显存峰值的影响，并找出失败边界。",
        "retries": 0,
        "max_retries": 1,
    }

@pytest.fixture
def trace():
    return TraceRecorder()

@pytest.fixture
def app_stubbed(trace):
    """
    返回一个“全 stub 节点”的图，用于 graph/perf 基础测试。
    TODO:
    - 你也可以在具体测试里自己 build_graph(...) 来覆盖不同分支
    """
    designer = wrap_node(trace, "designer", stubs.stub_designer_need_human)
    human = wrap_node(trace, "human_review", stubs.stub_human_review_interrupt)
    worker = wrap_node(trace, "exam_worker", stubs.stub_worker_collect)
    analyst = wrap_node(trace, "analyst", stubs.stub_analyst_ok)
    app = build_graph(designer=designer, human_review=human, worker=worker, analyst=analyst)
    return app
