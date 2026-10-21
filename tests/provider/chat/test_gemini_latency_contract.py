"""Exercise token counting and empty-response handling through actual ask calls."""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from akasha.helper.token_counter import myTokenizer
from akasha.tools.ask import ask

pytestmark = pytest.mark.unit


class SequenceChatModel(BaseChatModel):
    responses: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self):
        return "gemini:offline-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def make_ask(responses):
    model = SequenceChatModel(responses=responses)
    qa = ask(model=model, system_prompt="Reply with only OK.",
             prompt_format_type="chat_gpt", max_input_tokens=1000)
    qa.model = "gemini:offline-test"
    return qa, model


def test_default_gemini_ask_does_not_call_remote_token_counter(monkeypatch):
    import akasha.utils.models.gemi as gemi

    calls = []
    monkeypatch.setattr(gemi, "calculate_token", lambda *a, **k: calls.append(1) or 8)
    qa, model = make_ask([AIMessage(content="OK")])
    assert qa("Reply OK.") == "OK"
    assert model.calls == 1
    assert calls == [], "default ask must not send a separate count_tokens request"


def test_empty_ask_does_not_silently_generate_again(monkeypatch):
    monkeypatch.setattr(myTokenizer, "compute_tokens", lambda *a, **k: 8)
    qa, model = make_ask([
        AIMessage(content="", response_metadata={"finish_reason": "MAX_TOKENS"}),
        AIMessage(content="OK"),
    ])
    with pytest.raises(RuntimeError, match="MAX_TOKENS") as caught:
        qa("Reply OK.")
    assert model.calls == 1
    assert caught.value.diagnostics["finish_reason"] == "MAX_TOKENS"


def test_gemini_remote_count_is_explicit_and_does_not_hide_errors(monkeypatch):
    import akasha.utils.models.gemi as gemi

    calls = []
    monkeypatch.setattr(gemi, "calculate_token", lambda text, model_name: calls.append(model_name) or 17)
    assert myTokenizer.compute_tokens("hello", "gemini:gemini-3.5-flash", exact=True) == 17
    assert calls == ["gemini-3.5-flash"]

    def fail(*a, **k):
        raise TimeoutError("offline token timeout")

    monkeypatch.setattr(gemi, "calculate_token", fail)
    with pytest.raises(TimeoutError):
        myTokenizer.compute_tokens("hello", "gemini:gemini-3.5-flash", exact=True)


@pytest.mark.parametrize("alias", ["gemini", "google", "gemi"])
def test_local_count_handles_special_token_text_and_aliases(monkeypatch, alias):
    def forbidden(*a, **k):
        pytest.fail("local budgeting must not construct a Google client")

    monkeypatch.setattr("google.genai.Client", forbidden)
    assert myTokenizer.compute_tokens("hello <|endoftext|>", f"{alias}:gemini-3.5-flash") > 0
    assert myTokenizer.compute_tokens("", f"{alias}:gemini-3.5-flash") == 0


@pytest.mark.parametrize("reason,content", [
    ("STOP", ""), ("STOP", "\n"), ("MAX_TOKENS", " "), ("SAFETY", []),
    ("STOP", [{"type": "thinking", "thinking": "PRIVATE_REASONING"}]),
])
def test_empty_response_preserves_status_without_private_payload(monkeypatch, reason, content):
    from akasha.helper import EmptyResponseError

    monkeypatch.setattr(myTokenizer, "compute_tokens", lambda *a, **k: 8)
    qa, model = make_ask([AIMessage(
        content=content,
        response_metadata={"finish_reason": reason,
                           "prompt_feedback": {"block_reason": "SAFETY", "text": "PRIVATE_FEEDBACK"},
                           "private": "PRIVATE_METADATA"},
        usage_metadata={"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
    )])
    with pytest.raises(EmptyResponseError) as caught:
        qa("PRIVATE_PROMPT")
    assert model.calls == 1
    assert caught.value.diagnostics["finish_reason"] == reason
    assert caught.value.diagnostics["block_reason"] == "SAFETY"
    assert caught.value.diagnostics["total_tokens"] == 11
    assert "PRIVATE" not in str(caught.value)
    assert "PRIVATE" not in str(caught.value.diagnostics)


def test_visible_text_survives_thinking_blocks_and_truncation(monkeypatch):
    monkeypatch.setattr(myTokenizer, "compute_tokens", lambda *a, **k: 8)
    qa, model = make_ask([AIMessage(
        content=[{"type": "thinking", "thinking": "private"},
                 {"type": "text", "text": "OK"}],
        response_metadata={"finish_reason": "MAX_TOKENS"},
    )])
    assert qa("Reply OK.") == "OK"
    assert model.calls == 1


def test_empty_tool_call_is_reported_without_replaying_tools(monkeypatch):
    from akasha.helper import EmptyResponseError

    monkeypatch.setattr(myTokenizer, "compute_tokens", lambda *a, **k: 8)
    qa, model = make_ask([AIMessage(
        content="", tool_calls=[{"name": "private_tool", "args": {"secret": "PRIVATE"}, "id": "1"}],
        response_metadata={"finish_reason": "STOP"},
    )])
    with pytest.raises(EmptyResponseError) as caught:
        qa("Reply OK.")
    assert model.calls == 1
    assert caught.value.diagnostics["tool_calls"] == 1
    assert "PRIVATE" not in str(caught.value)


def test_local_estimate_still_rejects_excessive_prompt_before_generation():
    qa, model = make_ask([AIMessage(content="OK")])
    qa.max_input_tokens = 1
    with pytest.raises(ValueError, match="larger than max_input_tokens"):
        qa("This prompt exceeds the configured budget.")
    assert model.calls == 0


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("vertex", [False, True])
def test_explicit_counter_closes_client_and_bounds_request(monkeypatch, fail, vertex):
    from types import SimpleNamespace
    import os
    import akasha.utils.models.gemi as gemi

    events = []
    monkeypatch.setenv("GEMINI_API_KEY", "offline")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", str(vertex).lower())
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "ambient-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "ambient-location")

    class Client:
        def __init__(self, **kwargs):
            events.append(kwargs)
            assert kwargs["vertexai"] is vertex
            assert kwargs["http_options"].timeout == 10000
            assert kwargs["http_options"].retry_options.attempts == 1
            if vertex:
                assert "GOOGLE_CLOUD_PROJECT" not in os.environ
                assert "GOOGLE_CLOUD_LOCATION" not in os.environ
            self.models = self

        def count_tokens(self, **kwargs):
            assert kwargs == {"model": "gemini-test", "contents": "text"}
            if fail:
                raise TimeoutError("offline")
            return SimpleNamespace(total_tokens=7)

        def close(self):
            events.append("closed")

    monkeypatch.setattr(gemi.genai, "Client", Client)
    if fail:
        with pytest.raises(TimeoutError):
            gemi.calculate_token("text", "gemini-test")
    else:
        assert gemi.calculate_token("text", "gemini-test") == 7
    assert events[-1] == "closed"
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "ambient-project"
    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "ambient-location"


def test_unknown_diagnostic_values_do_not_copy_provider_text():
    from akasha.helper import EmptyResponseError

    error = EmptyResponseError(AIMessage(
        content=[{"type": "PRIVATE_PAYLOAD"}],
        response_metadata={"finish_reason": "PRIVATE_STATUS",
                           "prompt_feedback": {"block_reason": "PRIVATE_REASON"}},
    ))
    assert error.diagnostics["finish_reason"] == "unknown"
    assert error.diagnostics["block_reason"] == "unknown"
    assert error.diagnostics["content_types"] == ["other"]
    assert "PRIVATE" not in str(error)
