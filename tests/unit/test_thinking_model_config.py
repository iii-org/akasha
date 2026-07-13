import pytest
import importlib
from langchain_core.messages import AIMessage

from akasha.utils.models.chat import build_chat_model

pytestmark = pytest.mark.unit


def test_gemini_thinking_settings_are_forwarded(monkeypatch):
    captured = {}

    class FakeGemini:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_google_genai

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeGemini)
    build_chat_model(
        "gemini",
        "gemini-2.5-flash",
        {"GEMINI_API_KEY": "test"},
        thinking=True,
        thinking_budget=1024,
    )

    assert captured["include_thoughts"] is True
    assert captured["thinking_budget"] == 1024


def test_thinking_budget_is_ignored_when_thinking_is_disabled(monkeypatch):
    captured = {}

    class FakeGemini:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_google_genai

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeGemini)
    build_chat_model(
        "gemini",
        "gemini-2.5-flash",
        {"GEMINI_API_KEY": "test"},
        thinking=False,
        thinking_budget=1024,
    )

    assert "thinking_budget" not in captured
    assert "include_thoughts" not in captured


def test_ask_and_agents_accept_thinking_settings(monkeypatch):
    class FakeGemini:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    import langchain_google_genai
    agents_module = importlib.import_module("akasha.agent.agents")
    from akasha.tools.ask import ask

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeGemini)
    monkeypatch.setattr(
        "akasha.helper.handle_objects._get_env_var",
        lambda env_file="": {"GEMINI_API_KEY": "test"},
    )
    monkeypatch.setattr(agents_module, "create_agent", lambda **kwargs: object())

    qa = ask(
        model="gemini:gemini-2.5-flash",
        thinking=True,
        thinking_budget=1024,
    )
    agent = agents_module.agents(
        model="gemini:gemini-2.5-flash",
        thinking=True,
        thinking_budget=1024,
    )

    assert qa.thinking is True
    assert qa.thinking_budget == 1024
    assert agent.thinking is True
    assert agent.thinking_budget == 1024


def test_agent_stream_enables_thinking_events_automatically(monkeypatch):
    class FakeGemini:
        def __init__(self, **kwargs):
            pass

    class FakeAgent:
        def stream(self, *args, **kwargs):
            yield {
                "model": {
                    "messages": [
                        AIMessage(
                            content=[
                                {"type": "thinking", "thinking": "private plan"},
                                {"type": "text", "text": "final answer"},
                            ]
                        )
                    ]
                }
            }

    import langchain_google_genai
    agents_module = importlib.import_module("akasha.agent.agents")

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeGemini)
    monkeypatch.setattr(
        "akasha.helper.handle_objects._get_env_var",
        lambda env_file="": {"GEMINI_API_KEY": "test"},
    )
    monkeypatch.setattr(agents_module, "create_agent", lambda **kwargs: FakeAgent())

    agent = agents_module.agents(
        model="gemini:gemini-2.5-flash",
        stream=True,
        thinking=True,
    )
    events = list(agent("question"))

    assert events == [
        {"type": "thinking", "data": "private plan"},
        {"type": "answer", "data": "final answer"},
    ]


def test_agent_stream_accepts_langgraph_v2_update_shape(monkeypatch):
    class FakeGemini:
        def __init__(self, **kwargs):
            pass

    class FakeAgent:
        def stream(self, *args, **kwargs):
            yield {
                "type": "updates",
                "data": {
                    "model": {
                        "messages": [AIMessage(content="final answer")]
                    }
                },
            }

    import langchain_google_genai
    agents_module = importlib.import_module("akasha.agent.agents")
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeGemini)
    monkeypatch.setattr(
        "akasha.helper.handle_objects._get_env_var",
        lambda env_file="": {"GEMINI_API_KEY": "test"},
    )
    monkeypatch.setattr(agents_module, "create_agent", lambda **kwargs: FakeAgent())

    agent = agents_module.agents(
        model="gemini:gemini-2.5-flash",
        stream=True,
    )
    assert list(agent("question")) == [
        {"type": "answer", "data": "final answer"}
    ]
