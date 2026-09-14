import builtins

import pytest
from click.testing import CliRunner

from cli.glue import akasha

pytestmark = pytest.mark.unit


def test_toy_command_points_to_ui_extra_when_streamlit_is_missing(monkeypatch):
    original_import = builtins.__import__

    def block_streamlit(name, *args, **kwargs):
        if name.split(".", maxsplit=1)[0] == "streamlit":
            raise ImportError("streamlit missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_streamlit)

    result = CliRunner().invoke(akasha, ["toy"])

    assert result.exit_code == 1
    assert 'uv add "akasha-terminal[ui]"' in result.output
    assert 'pip install "akasha-terminal[ui]"' in result.output
