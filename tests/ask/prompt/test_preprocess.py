import pytest

from akasha.helper import preprocess_prompts as prompts

pytestmark = pytest.mark.unit


def test_merge_history_and_prompt_without_history(monkeypatch):
    monkeypatch.setattr(prompts, "decide_auto_prompt_format_type", lambda model: "chat_gpt")
    monkeypatch.setattr(
        prompts,
        "format_sys_prompt",
        lambda system_prompt, prompt, prompt_format_type: [system_prompt, prompt, prompt_format_type],
    )

    result = prompts.merge_history_and_prompt([], "system", "hello", prompt_format_type="auto")

    assert result == ["system", "hello", "chat_gpt"]


def test_merge_history_and_prompt_for_chat_format(monkeypatch):
    monkeypatch.setattr(
        prompts,
        "format_sys_prompt",
        lambda system_prompt, prompt, prompt_format_type: [{"role": "system", "content": system_prompt or prompt, "fmt": prompt_format_type}],
    )
    monkeypatch.setattr(
        prompts,
        "format_history_prompt",
        lambda history_messages, prompt_format_type, user_tag, assistant_tag: [
            {"role": user_tag, "content": history_messages[0], "fmt": prompt_format_type},
            {"role": assistant_tag, "content": history_messages[1], "fmt": prompt_format_type},
        ],
    )

    result = prompts.merge_history_and_prompt(
        ["hi", "hello"],
        "system",
        "question",
        prompt_format_type="chat_gemini",
    )

    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert result[2]["role"] == "model"


def test_merge_history_and_prompt_for_string_format(monkeypatch):
    monkeypatch.setattr(
        prompts,
        "format_sys_prompt",
        lambda system_prompt, prompt, prompt_format_type: f"{system_prompt}|{prompt}|{prompt_format_type}",
    )

    result = prompts.merge_history_and_prompt(
        ["Q1", "A1", "Q2", "A2"],
        "system",
        "question",
        prompt_format_type="gpt",
    )

    assert "user: Q1" in result
    assert "assistant: A2" in result
    assert result.endswith("|gpt")


def test_retri_history_messages_limits_pairs_and_tokens(monkeypatch):
    monkeypatch.setattr(prompts.myTokenizer, "compute_tokens", lambda text, model_name: len(text))
    messages = [
        {"role": "User", "content": "Q1"},
        {"role": "Assistant", "content": "A1"},
        {"role": "User", "content": "Q2"},
        {"role": "Assistant", "content": "A2"},
    ]

    history, token_len = prompts.retri_history_messages(
        messages,
        pairs=1,
        max_input_tokens=100,
        model_name="openai:gpt-4o",
    )

    assert "User: Q2" in history
    assert "Assistant: A2" in history
    assert token_len > 0
