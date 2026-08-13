import pytest

from akasha.agent.agents import _as_message_list, _thinking_text
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.unit


def test_legacy_agent_history_is_converted_to_chat_messages():
    messages = _as_message_list(
        [
            {"role": "Action", "content": "called tool"},
            {"role": "Observation", "content": {"value": 1}},
        ]
    )
    assert messages == [
        {"role": "assistant", "content": "called tool"},
        {"role": "user", "content": {"value": 1}},
    ]


def test_thinking_is_read_from_standard_content_blocks():
    message = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "private reasoning"},
            {"type": "text", "text": "public answer"},
        ]
    )
    assert _thinking_text(message) == "private reasoning"
