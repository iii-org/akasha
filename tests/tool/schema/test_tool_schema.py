import pytest

from akasha.agent.base import create_tool

pytestmark = pytest.mark.unit


def test_create_tool_exposes_structured_arguments():
    def add(left: int, right: int) -> int:
        return left + right

    tool = create_tool("Add two numbers", add, "add")
    assert tool.name == "add"
    assert "left" in tool.args
    assert "right" in tool.args
    assert tool.invoke({"left": 2, "right": 3}) == 5
