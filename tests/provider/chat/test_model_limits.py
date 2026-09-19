"""Validate live budgets at the actual provider factory boundary."""
import pytest
from akasha.utils.models.chat import build_chat_model
from tests.support.model_limits import model_settings, output_token_budget

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model", [
    "openai:gpt-5.4", "azure:DeepSeek-V4-Flash",
    "gemini:gemini-3.5-flash", "anthropic:claude-sonnet-4-6",
])
@pytest.mark.parametrize("thinking", [False, True])
def test_live_budget_and_effective_request_fit_documented_limit(model, thinking):
    entry = model_settings()[model]
    budget = output_token_budget(model)
    ceiling = entry["capabilities"]["output_token_limit"]
    assert 256 < budget <= ceiling
    assert entry["capabilities"]["output_token_limit_source"].startswith("https://")
    provider, name = model.split(":", 1)
    env = {"OPENAI_API_KEY": "unused", "GEMINI_API_KEY": "unused",
           "ANTHROPIC_API_KEY": "unused"}
    if provider == "azure":
        env = {"AZURE_OPENAI_API_KEY": "unused",
               "AZURE_OPENAI_BASE_URL": "https://example.invalid/v1"}
    instance = build_chat_model(
        provider, name, env, max_output_tokens=budget,
        thinking=thinking, thinking_budget="medium",
    )
    effective = (instance.max_output_tokens if provider == "gemini"
                 else instance.max_tokens)
    assert budget <= effective <= ceiling
    if provider == "anthropic" and thinking:
        assert instance.thinking["budget_tokens"] < effective


def test_ollama_does_not_confuse_context_with_output_limit():
    entry = model_settings()["ollama:gemma4:26b"]
    assert entry["capabilities"]["output_token_limit"] is None
    assert output_token_budget(entry["id"]) == entry["test_max_output_tokens"]


@pytest.mark.parametrize("budget", [0, True, 10001])
def test_invalid_or_excessive_budget_fails_before_network(monkeypatch, budget):
    import tests.support.model_limits as limits
    monkeypatch.setattr(limits, "model_settings", lambda: {
        "test:model": {"test_max_output_tokens": budget,
                       "capabilities": {"output_token_limit": 10000}}
    })
    with pytest.raises(ValueError):
        limits.output_token_budget("test:model")
