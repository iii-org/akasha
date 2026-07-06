import pytest

from akasha.agent.agents import _is_final_action

pytestmark = pytest.mark.unit


def test_final_action_aliases_are_case_insensitive():
    assert _is_final_action("final answer") is True
    assert _is_final_action("FINAL") is True
    assert _is_final_action("answer") is True
    assert _is_final_action("tool") is False
    assert _is_final_action(None) is False
