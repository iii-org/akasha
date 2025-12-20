import asyncio
import os

import pytest

import importlib

agents_module = importlib.import_module("akasha.agent.agents")

# Live-model integration tests are opt-in to avoid accidental token usage.
RUN_LLM_TESTS = bool(os.getenv("RUN_LLM_TESTS"))

# Models to verify; each tuple is (model_name, required_env_key).
LIVE_MODELS = [
    ("openai:gpt-4o", "OPENAI_API_KEY"),
    ("gemini:gemini-2.5-flash", "GEMINI_API_KEY"),
]


@pytest.mark.integration
@pytest.mark.skipif(not RUN_LLM_TESTS, reason="LLM integration disabled")
@pytest.mark.parametrize("model_name, key_env", LIVE_MODELS)
def test_live_model_final_action_aliases(model_name: str, key_env: str):
    """
    Integration test: call real LLM to ensure it returns a final-action alias that the agent accepts.
    Skips automatically if the provider API key is missing or RUN_LLM_TESTS is not set.
    """
    if not os.getenv(key_env):
        pytest.skip(f"{key_env} not set")

    # Use a tiny prompt to reduce token cost; agent prompt enforces the JSON schema.
    agent = agents_module.agents(
        tools=[],
        model=model_name,
        keep_logs=False,
        max_round=2,
        temperature=1.0,
        env_file=os.getenv("ENV_FILE", ".env"),
    )

    response = asyncio.run(
        agent._run_agent(
            "Please follow the required JSON schema and respond with 'hi' as the final answer.",
            lambda tool, tool_input: None,
            messages=[],
        )
    )

    # Extract the recorded final action for visibility in test logs.
    final_action = None
    for msg in agent.messages[::-1]:
        if msg.get("role") == "Action":
            try:
                import json

                action_obj = json.loads(msg["content"])
                final_action = action_obj.get("action")
            except Exception:
                final_action = msg.get("content")
            break

    print(f"[integration] model={model_name} final_action={final_action}")

    assert isinstance(response, str) and response.strip(), "LLM did not return a final response string"
