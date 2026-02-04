# tests/integration/test_hitl_edit.py
import pytest
from backend.agent.run_agent import handle_hitl_interrupts
from tests.helpers.fakes import decider_edit

@pytest.mark.integration
def test_hitl_edit_flow(app, init_state, config):
    """
    TODO:
    - 断言 edit 时 experiments 被解析并通过 resume_value 写回 state
    - 断言图回到 designer 进入 need_redesign / 或你定义的 stage
    """
    state = app.invoke(init_state, config=config)
    final_state = handle_hitl_interrupts(app, state, config=config, decider=decider_edit)

    # TODO: assert final_state 中包含你约定的 experiments / feedback / stage
