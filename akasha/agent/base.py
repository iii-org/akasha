from typing import List, Dict, Callable, Union
import logging, inspect
from langchain.tools import BaseTool


def _get_tool_explaination(tools: List[BaseTool]) -> Dict[str, str]:
    """get the explaination and parameter name, type of tools"""
    ret = {}
    for tool in tools:
        name = tool.name
        dscp = tool.description
        para_des = ""
        try:
            para_des = tool.para_des
        except:
            try:
                para_des = "args: {"
                dic = tool.args
                for k, v in dic.items():
                    para_des += f"{k}: {v['type']},"
                para_des += "}"

            except:
                para_des = ""
        ret[name] = dscp + ", " + para_des

    return ret


def create_tool(tool_name: str, tool_description: str,
                func: Callable) -> Union[BaseTool, None]:
    """input function to create a tool

    Args:
        tool_name (str): name of the tool
        tool_description (str): description of the function and the parameters of the tool
        func (Callable): callable function of the tool

    Returns:
        Union[BaseTool, None]: return the tool if success, return None if failed
    """
    try:

        class custom_tool(BaseTool):
            name: str
            description: str
            para_des: str

            async def ainvoke(self, tool_input: dict, **kwargs):
                """Asynchronous invocation of the tool."""
                if inspect.iscoroutinefunction(func):
                    # If func is asynchronous, await it
                    return await func(**tool_input, **kwargs)
                else:
                    # If func is synchronous, run it directly
                    return func(**tool_input, **kwargs)

            def _run(self, *args, **kwargs):
                return func(*args, **kwargs)

            def run(self, *args, **kwargs):
                return self._run(*args, **kwargs)

        sig = inspect.signature(func)
        params = sig.parameters

        prm = "args: {"
        for n, p in params.items():
            if p.annotation is not inspect.Parameter.empty:
                prm += f" {n}: {p.annotation},"
        prm += "}"

    except Exception as e:

        logging.error(f"Cannot create tool correctly, {e}\n\n")
        raise e

    return custom_tool(name=tool_name,
                       description=tool_description,
                       para_des=prm)
