# tests/integration/test_hitl_accept.py
import pytest
from backend.agent.run_agent import handle_hitl_interrupts
from tests.helpers.fakes import decider_accept

@pytest.mark.integration
def test_hitl_accept_flow(app, init_state, config, fakes):
    """
    TODO:
    1) state = app.invoke(init_state)
       - 断言进入 interrupt（__interrupt__ 存在）
    2) final_state = handle_hitl_interrupts(..., decider=decider_accept)
       - 断言最终 analyst_report ok=True
       - 断言调用路径：designer -> human_review -> designer -> ...
    """
    state = app.invoke(init_state, config=config)

    # TODO: assert "__interrupt__" in state

    final_state = handle_hitl_interrupts(app, state, config=config, decider=decider_accept)

    # TODO: assert final_state["analyst_report"]["ok"] is True
    # TODO: assert path by checking fakes["designer"].calls etc.
