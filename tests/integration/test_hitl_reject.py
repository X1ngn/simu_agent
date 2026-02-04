# tests/integration/test_hitl_reject.py
import pytest
from backend.agent.run_agent import handle_hitl_interrupts
from tests.helpers.fakes import decider_reject

@pytest.mark.integration
def test_hitl_reject_flow(app, init_state, config):
    """
    TODO:
    - 断言 reject 后 hitl_status == "rejected"（建议你按上面生产改造加上）
    - 断言不会继续跑 worker / analyst
    """
    state = app.invoke(init_state, config=config)
    final_state = handle_hitl_interrupts(app, state, config=config, decider=decider_reject)

    # TODO: assert final_state.get("hitl_status") == "rejected"
