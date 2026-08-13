import pytest

from akasha.helper.run_llm import content_to_text, content_to_thinking, normalize_chat_input

pytestmark = pytest.mark.unit


def test_reasoning_content_is_not_part_of_public_text():
    content = [
        {"type": "reasoning", "reasoning": "private"},
        {"type": "text", "text": "answer"},
    ]
    assert content_to_text(content) == "answer"
    assert content_to_thinking(content) == "private"


def test_legacy_gemini_parts_are_chat_message_content():
    assert normalize_chat_input(
        [{"role": "model", "parts": ["system text"]}]
    ) == [{"role": "system", "content": "system text"}]


def test_reasoning_content_from_provider_metadata_is_normalized():
    assert content_to_thinking(
        "", {"reasoning_content": "private provider reasoning"}
    ) == "private provider reasoning"
