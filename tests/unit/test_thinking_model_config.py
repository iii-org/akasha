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
        thinking_budget="medium",
    )

    assert captured["include_thoughts"] is True
    assert captured["thinking_budget"] == 4096


@pytest.mark.parametrize(
    ("level", "max_output_tokens", "expected"),
    [
        ("low", 4096, 2048),
        ("medium", 16384, 16384),
        ("high", 65536, 131072),
    ],
)
def test_thinking_levels_scale_with_output_budget(
    level, max_output_tokens, expected
):
    from akasha.utils.models.thinking import normalize_thinking_budget

    assert (
        normalize_thinking_budget(
            level,
            thinking=True,
            max_output_tokens=max_output_tokens,
        )
        == expected
    )


def test_thinking_level_rejects_typo():
    with pytest.raises(ValueError, match="low, medium, high"):
        build_chat_model(
            "ollama",
            "gemma4:26b",
            {"OLLAMA_API_BASE": "https://ollama.example"},
            thinking=True,
            thinking_budget="mediu",
        )


def test_openai_thinking_level_maps_to_reasoning_effort(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    build_chat_model(
        "openai",
        "gpt-5.4",
        {"OPENAI_API_KEY": "test"},
        thinking=True,
        thinking_budget="high",
    )

    assert captured["reasoning_effort"] == "high"


def test_anthropic_does_not_send_temperature(monkeypatch):
    captured = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_anthropic

    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", FakeChatAnthropic)
    build_chat_model(
        "anthropic",
        "claude-sonnet-4-6",
        {"ANTHROPIC_API_KEY": "test"},
        temperature=0.0,
    )

    assert captured["model"] == "claude-sonnet-4-6"
    assert "temperature" not in captured


def test_anthropic_thinking_reserves_tokens_for_reasoning(monkeypatch):
    captured = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_anthropic

    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", FakeChatAnthropic)
    build_chat_model(
        "anthropic",
        "claude-sonnet-4-6",
        {"ANTHROPIC_API_KEY": "test"},
        max_output_tokens=256,
        thinking=True,
        thinking_budget="medium",
    )

    assert captured["max_tokens_to_sample"] == 5120
    assert captured["thinking"]["budget_tokens"] == 4096


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


def test_ollama_thinking_has_at_least_2048_prediction_tokens(monkeypatch):
    captured = {}

    class FakeOllama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_ollama

    monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeOllama)
    build_chat_model(
        "ollama",
        "gemma4:26b",
        {"OLLAMA_API_BASE": "https://ollama.example"},
        max_output_tokens=512,
        thinking=True,
        thinking_budget=256,
    )

    assert captured["reasoning"] is True
    assert captured["num_predict"] == 2048


def test_azure_openai_compatible_endpoint_uses_dedicated_environment(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        langchain_openai,
        "AzureChatOpenAI",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy Azure client used")),
    )

    build_chat_model(
        "openai",
        "gpt-4o-deployment",
        {
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_OPENAI_BASE_URL": "https://example.openai.azure.com/openai/v1/",
        },
    )

    assert captured["model"] == "gpt-4o-deployment"
    assert captured["api_key"] == "azure-key"
    assert captured["base_url"] == "https://example.openai.azure.com/openai/v1/"


def test_openai_uses_openai_base_url_without_azure_environment(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    build_chat_model(
        "openai",
        "gpt-4o-mini",
        {
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_BASE_URL": "https://api.openai.example/v1/",
        },
    )

    assert captured["api_key"] == "openai-key"
    assert captured["base_url"] == "https://api.openai.example/v1/"


def test_azure_environment_requires_key_and_base_url():
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        build_chat_model(
            "openai",
            "gpt-4o-deployment",
            {"AZURE_OPENAI_BASE_URL": "https://example.openai.azure.com/openai/v1/"},
        )


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
