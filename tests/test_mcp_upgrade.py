import asyncio
import os
import sys
from pathlib import Path

import pytest


class _DummyModel:
    def get_num_tokens(self, _text: str) -> int:
        return 0


@pytest.mark.agent
def test_mcp_agent_load_tools_after_adapters_upgrade(monkeypatch: pytest.MonkeyPatch):
    import akasha
    import akasha.utils.atman as atman

    monkeypatch.setattr(atman, "handle_model", lambda *args, **kwargs: _DummyModel())

    agent = akasha.agents(
        model="gemini:gemini-2.5-flash",
        temperature=0.0,
        verbose=False,
        keep_logs=False,
        env_file=os.fspath(
            Path(__file__).resolve().parents[1] / "tests" / "test_upgrade" / ".env"
        ),
    )

    async def _fake_acall(_prompt: str):
        return "ok"

    monkeypatch.setattr(agent, "acall", _fake_acall)

    repo_root = Path(__file__).resolve().parents[1]
    server = repo_root / "tests" / "test_upgrade" / "mcp_math_server.py"
    assert server.exists()

    connection_info = {
        "math": {
            "command": sys.executable,
            "args": [os.fspath(server)],
            "transport": "stdio",
        }
    }

    result = agent.mcp_agent(connection_info, "2+3=?")

    assert result == "ok"
    assert "add" in agent.tools
    assert "multiply" in agent.tools
