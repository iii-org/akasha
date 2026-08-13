import pytest

from akasha.agent.agents import _as_message_list, _message_text

pytestmark = pytest.mark.unit


def test_agent_uses_chat_message_text_without_json_action_aliases():
    assert _message_text({"content": "final answer"}) == "final answer"
    assert _as_message_list([{"role": "user", "content": "hello"}]) == [
        {"role": "user", "content": "hello"}
    ]
