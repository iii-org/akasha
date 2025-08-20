from typing import List, Dict, Callable, Union
import logging
import inspect
from langchain.tools import BaseTool


def get_tool_explaination(tools: List[BaseTool]) -> Dict[str, str]:
    """get the explaination and parameter name, type of tools"""
    ret = {}
    for tool in tools:
        name = tool.name
        dscp = tool.description
        para_des = ""
        try:
            para_des = tool.para_des
        except Exception:
            try:
                para_des = "args: {"
                dic = tool.args
                for k, v in dic.items():
                    para_des += f"{k}:{v['type']}, "
                para_des = para_des[:-2]  # remove the last comma and space
                para_des += "}"

            except Exception:
                para_des = ""
        ret[name] = dscp + ", " + para_des

    return ret


def create_tool(
    tool_description: str, func: Callable, tool_name: Union[str, None] = None
) -> Union[BaseTool, None]:
    """input function to create a tool

    Args:
        tool_description (str): description of the function and the parameters of the tool
        func (Callable): callable function of the tool
        tool_name (str, optional): name of the tool. If not provided, the function's name will be used.

    Returns:
        Union[BaseTool, None]: return the tool if success, return None if failed
    """
    try:
        # Check if tool_name is provided, otherwise use the function name
        if tool_name is None:
            tool_name = func.__name__

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

    return custom_tool(name=tool_name, description=tool_description, para_des=prm)


def get_REACT_PROMPT(tool_explain_str: str, tool_name_str: str) -> str:
    """get the REACT prompt for the tool"""

    ret = f"""Respond to the human as helpfully and accurately as possible. You have access to the following tools:\n\n{tool_explain_str}\n
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\n
Valid "action" values: "Answer" or {tool_name_str}\n\n
Provide only ONE action per $JSON_BLOB, as shown:\n{{\n```\n\n  "action": $TOOL_NAME,\n  "action_input": $INPUT\n}}\n```$INPUT is a dictionary that contains tool parameters and their values\n\n
the meaning of each format:\n
Question: input question to answer\nThought: consider previous and subsequent steps\nAction:\n```\n$JSON_BLOB\n```\nObservation: action result\n
... (repeat Thought/Action N times)\nThought: I know what to respond\nAction:\n```\n{{\n  "action": "Answer",\n  "action_input": "Final response to human"\n}}\n```\n\n
Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Thought: then Action:```$JSON_BLOB```.\n
"""

    return ret


DEFAULT_REMEMBER_PROMPT = (
    "**Remember, Format is Thought: then Action:```$JSON_BLOB```\n\n"
)


DEFAULT_OBSERVATION_PROMPT = "\n\nBelow are your previous work, check them carefully and provide the next action and thought,**do not ask same question repeatedly: "
DEFAULT_RETRI_OBSERVATION_PROMPT = "User will give you Question, Thought and Observation, return the information from Observation that you think is most relevant to the Question or Thought, if you can't find the information, return None."
