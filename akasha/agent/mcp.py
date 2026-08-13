"""Public helpers for adapting MCP tools to the Akasha agent contract."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool


def normalize_mcp_result(result: Any) -> Any:
    """Normalize common MCP content results without discarding content blocks."""
    if isinstance(result, dict):
        return result
    if not isinstance(result, (list, tuple)):
        return result

    blocks = list(result)
    if len(blocks) == 1 and isinstance(blocks[0], dict):
        block = blocks[0]
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text = block["text"]
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return blocks


def normalize_mcp_tools(tools: Iterable[BaseTool]) -> list[BaseTool]:
    """Wrap discovered MCP tools with the normalized Akasha result contract."""
    normalized: list[BaseTool] = []
    for tool in tools:
        invoke = _bind_tool(tool)

        normalized.append(
            StructuredTool.from_function(
                coroutine=invoke,
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema,
            )
        )
    return normalized


def _bind_tool(source_tool: BaseTool) -> Any:
    async def bound(**kwargs: Any) -> Any:
        result = await source_tool.ainvoke(kwargs)
        return normalize_mcp_result(result)

    return bound
