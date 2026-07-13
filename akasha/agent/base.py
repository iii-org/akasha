from typing import Callable, Dict, List, Union
import inspect
import logging

from langchain_core.tools import BaseTool, StructuredTool


def get_tool_explaination(tools: List[BaseTool]) -> Dict[str, str]:
    """Return a compact description of tools for logs and diagnostics."""
    ret = {}
    for tool in tools:
        description = tool.description
        try:
            args = ", ".join(
                f"{name}:{schema.get('type', 'value')}"
                for name, schema in tool.args.items()
            )
            description = f"{description}, args: {{{args}}}"
        except Exception:
            pass
        ret[tool.name] = description
    return ret


def create_tool(
    tool_description: str, func: Callable, tool_name: Union[str, None] = None
) -> Union[BaseTool, None]:
    """Create a LangChain structured tool with an inferred argument schema."""
    try:
        if tool_name is None:
            tool_name = func.__name__
        return StructuredTool.from_function(
            func=None if inspect.iscoroutinefunction(func) else func,
            coroutine=func if inspect.iscoroutinefunction(func) else None,
            name=tool_name,
            description=tool_description,
        )
    except Exception as exc:
        logging.error("Cannot create tool correctly, %s\n", exc)
        raise
