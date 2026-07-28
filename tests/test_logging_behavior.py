import importlib
import io
import logging
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from akasha.utils.logging_config import configure_logging

pytestmark = pytest.mark.unit


def _get_console_handler():
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler):
            formatter = getattr(handler, "formatter", None)
            fmt = getattr(formatter, "_fmt", "")
            if "[akasha]" in fmt:
                return handler
    return None


def _get_file_handler():
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            return handler
    return None


def test_verbose_controls_console_output():
    configure_logging(verbose=False, keep_logs=False)
    logger = logging.getLogger("akasha.test")
    logger.setLevel(logging.INFO)
    handler = _get_console_handler()
    assert handler is not None
    stream = io.StringIO()
    original_stream = handler.stream
    handler.stream = stream
    try:
        logger.info("console-hidden")
        assert "console-hidden" not in stream.getvalue()

        configure_logging(verbose=True, keep_logs=False)
        logger.info("console-visible")
        assert "console-visible" in stream.getvalue()
    finally:
        handler.stream = original_stream


def test_keep_logs_writes_file_only(tmp_path):
    log_file = tmp_path / "akasha.log"
    configure_logging(verbose=False, keep_logs=str(log_file))
    logger = logging.getLogger("akasha.test")
    logger.setLevel(logging.INFO)
    handler = _get_console_handler()
    assert handler is not None
    stream = io.StringIO()
    original_stream = handler.stream
    handler.stream = stream
    try:
        logger.warning("file-visible")
        assert "file-visible" not in stream.getvalue()
    finally:
        handler.stream = original_stream

    file_handler = _get_file_handler()
    assert file_handler is not None
    file_handler.flush()
    assert log_file.exists()
    assert "file-visible" in log_file.read_text(encoding="utf-8")


def test_keep_logs_bool_creates_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    configure_logging(verbose=False, keep_logs=True)
    logger = logging.getLogger("akasha.test")
    logger.setLevel(logging.INFO)
    logger.error("default-path")

    log_file = tmp_path / "logs" / f"ak_{datetime.now():%Y%m%d}.log"
    assert log_file.exists()
    assert "default-path" in log_file.read_text(encoding="utf-8")


def _install_fake_agent(monkeypatch, fake_agent):
    agents_module = importlib.import_module("akasha.agent.agents")

    class FakeModel:
        def get_num_tokens(self, text):
            return len(text)

    monkeypatch.setattr(
        "akasha.utils.atman.handle_model",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        agents_module,
        "create_agent",
        lambda **kwargs: fake_agent,
    )
    return agents_module


def test_agent_verbose_prints_non_stream_tool_trace(monkeypatch, capsys):
    class FakeAgent:
        async def ainvoke(self, _payload, config=None):
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "load_skill",
                                "args": {"reference": "python-repl-skill"},
                                "id": "load-1",
                            }
                        ],
                    ),
                    ToolMessage(
                        content="Skill 'python-repl-skill' loaded.",
                        name="load_skill",
                        tool_call_id="load-1",
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "python_execute",
                                "args": {
                                    "skill": "python-repl-skill",
                                    "source": "total / len(values)",
                                },
                                "id": "python-1",
                            }
                        ],
                    ),
                    ToolMessage(
                        content="execution: repl\nstdout:\n5.0",
                        name="python_execute",
                        tool_call_id="python-1",
                    ),
                    AIMessage(content="The average is 5.0."),
                ]
            }

    agents_module = _install_fake_agent(monkeypatch, FakeAgent())
    agent = agents_module.agents(
        model="fake:model",
        verbose=True,
        keep_logs=False,
    )

    assert agent("calculate") == "The average is 5.0."
    output = capsys.readouterr().out
    assert "[akasha] tool call: load_skill" in output
    assert '"reference": "python-repl-skill"' in output
    assert "[akasha] tool result: load_skill" in output
    assert "[akasha] tool call: python_execute" in output
    assert '"source": "total / len(values)"' in output
    assert "execution: repl" in output

    quiet_agent = agents_module.agents(
        model="fake:model",
        verbose=False,
        keep_logs=False,
    )
    assert quiet_agent("calculate") == "The average is 5.0."
    assert "[akasha] tool" not in capsys.readouterr().out


def test_agent_keep_logs_writes_the_same_tool_trace_as_verbose(
    monkeypatch, tmp_path, capsys
):
    class FakeAgent:
        async def ainvoke(self, _payload, config=None):
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "python_execute",
                                "args": {"source": "2 + 3"},
                                "id": "python-1",
                            }
                        ],
                    ),
                    ToolMessage(
                        content="execution: repl\nstdout:\n5",
                        name="python_execute",
                        tool_call_id="python-1",
                    ),
                    AIMessage(content="5"),
                ]
            }

    log_file = tmp_path / "agent.log"
    agents_module = _install_fake_agent(monkeypatch, FakeAgent())
    agent = agents_module.agents(
        model="fake:model",
        verbose=True,
        keep_logs=str(log_file),
    )

    assert agent("calculate") == "5"
    console = capsys.readouterr().out
    file_handler = _get_file_handler()
    assert file_handler is not None
    file_handler.flush()
    recorded = log_file.read_text(encoding="utf-8")

    for text in (
        "tool call: python_execute",
        '"source": "2 + 3"',
        "tool result: python_execute",
        "execution: repl",
        "stdout:\n5",
    ):
        assert text in console
        assert text in recorded

    assert console.count("tool call: python_execute") == 1

def test_agent_verbose_prints_stream_tool_trace(monkeypatch, capsys):
    class FakeAgent:
        def stream(self, _payload, config=None, stream_mode=None):
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "python_execute",
                        "args": {"source": "2 + 3"},
                        "id": "python-1",
                    }
                ],
            )
            yield ToolMessage(
                content="execution: repl\nstdout:\n5",
                name="python_execute",
                tool_call_id="python-1",
            )
            yield AIMessageChunk(content="5")

    agents_module = _install_fake_agent(monkeypatch, FakeAgent())
    agent = agents_module.agents(
        model="fake:model",
        stream=True,
        verbose=True,
        keep_logs=False,
    )

    events = list(agent("calculate"))
    output = capsys.readouterr().out
    assert [event["type"] for event in events] == ["tool", "answer"]
    assert output.count("[akasha] tool call: python_execute") == 1
    assert "[akasha] tool result: python_execute" in output
    assert "execution: repl" in output
