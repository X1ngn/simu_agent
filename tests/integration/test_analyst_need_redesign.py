# tests/integration/test_analyst_need_redesign.py
import pytest
from tests.helpers.fakes import FakeAnalyst

@pytest.mark.integration
def test_analyst_not_ok_routes_to_redesign(fakes, init_state, config):
    """
    TODO:
    - analyst ok=False 时，应该走 need_redesign 分支回 designer
    - 你可以在 FakeDesigner 脚本里记录被再次调用次数
    """
    fakes["analyst"] = FakeAnalyst(ok=False)
    # TODO: 重新 build app，用 ok=False analyst
    # TODO: 跑图并断言 designer 被再次调用（重设计）
    pass
