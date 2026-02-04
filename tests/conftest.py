import pytest
from helpers.fakes import FakeDesigner, FakeHumanReview, FakeWorker, FakeAnalyst
from helpers.build_app import build_test_app

@pytest.fixture
def config():
    return {"configurable": {"thread_id": "pytest_thread"}}

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
def fakes():
    return {
        "designer": FakeDesigner(),
        "human_review": FakeHumanReview(),
        "worker": FakeWorker(),
        "analyst": FakeAnalyst(ok=True),
    }

@pytest.fixture
def app(fakes):
    return build_test_app(
        designer=fakes["designer"],
        human_review=fakes["human_review"],
        worker=fakes["worker"],
        analyst=fakes["analyst"],
    )
