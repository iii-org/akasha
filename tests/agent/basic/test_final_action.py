import asyncio
import os

import pytest

import importlib
from tests.support.live import require_keys

agents_module = importlib.import_module("akasha.agent.agents")

# Models to verify; each tuple is (model_name, required_env_key).
LIVE_MODELS = [
    ("openai:gpt-4o", "OPENAI_API_KEY"),
    ("gemini:gemini-2.5-flash", "GEMINI_API_KEY"),
]


@pytest.mark.live
@pytest.mark.requires_api
@pytest.mark.smoke
@pytest.mark.parametrize("model_name, key_env", LIVE_MODELS)
def test_live_model_final_action_aliases(model_name: str, key_env: str):
    """
    Integration test: call real LLM to ensure it returns a final-action alias that the agent accepts.
    Skips automatically if the provider API key is missing or live tests are disabled.
    """
    require_keys(key_env)

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
        agent.acall("Please respond with 'hi' as the final answer.", messages=[])
    )

    print(f"[integration] model={model_name} final_answer={response}")

    assert isinstance(response, str) and response.strip(), "LLM did not return a final response string"
