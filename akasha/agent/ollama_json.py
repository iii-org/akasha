"""Bridge LangChain's official JSON decision agent to its modern tool runtime.

The structured-chat factory renders tool schemas and parses AgentAction/Finish.
The outer create_agent graph owns tool execution, validation, skills and limits.
"""

from __future__ import annotations

import json
from uuid import uuid4

from langchain_classic.agents import create_structured_chat_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from akasha.helper.run_llm import content_to_text


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是可以多輪使用工具的助理。\n{instructions}\n可用工具：\n{tools}\n"
     "每輪只能輸出一個合法JSON物件，包含action和action_input，不要解釋或Markdown。"
     "合法action為{tool_names}或Final Answer。"
     '需要工具時：{{"action":"工具名稱","action_input":{{"參數名稱":"參數值"}}}}。'
     '完成任務時：{{"action":"Final Answer","action_input":"回答原始問題"}}。'
     "根據原始問題與已執行工具的真實結果決定下一步，每輪只選一個工具。"
     "不要捏造工具結果。根據前一步真實結果填寫下一步參數，不要放運算式或變數名稱。"
     "檢查歷史紀錄，不要重做已完成的步驟；所有步驟完成後才輸出Final Answer。"
     "最後回答必須忠實引用工具結果，不得自行重新排列或改寫數值，並回答完整過程與結果。"
     "使用使用者的語言。若缺少必要資訊，在最終回答說明需要補充什麼，不要猜測參數。"),
    MessagesPlaceholder("chat_history"),
    ("human", "原始問題：{input}\n\n已執行工具與真實結果：\n{agent_scratchpad}"),
])


class _ResponseCapture(BaseCallbackHandler):
    """Retain provider metadata without publishing internal JSON token events."""

    def __init__(self):
        self.message = None

    def on_llm_end(self, response, **kwargs):
        self.message = response.generations[-1][-1].message


def _payload(messages):
    current = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    history = []
    for message in messages[:current]:
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, ToolMessage):
            history.append(HumanMessage(content=f"Tool {message.name}: {message.content}"))
        elif isinstance(message, AIMessage) and message.tool_calls:
            history.append(AIMessage(content=json.dumps(message.tool_calls, ensure_ascii=False)))
        else:
            history.append(message)
    calls = {}
    steps = []
    for message in messages[current + 1:]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls[call["id"]] = AgentAction(
                    tool=call["name"], tool_input=call["args"],
                    log=json.dumps({"action": call["name"], "action_input": call["args"]}, ensure_ascii=False),
                )
        elif isinstance(message, ToolMessage):
            steps.append((calls[message.tool_call_id], content_to_text(message.content)))
    return {
        "instructions": "\n\n".join(content_to_text(m.content) for m in messages if isinstance(m, SystemMessage)),
        "input": content_to_text(messages[current].content),
        "chat_history": history,
        "intermediate_steps": steps,
    }


class OllamaJSONModel(BaseChatModel):
    """Translate official structured-chat decisions into standard AI messages."""

    client: BaseChatModel

    @property
    def _llm_type(self):
        return "akasha-ollama-structured-chat"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if tool_choice not in (None, "auto"):
            raise ValueError("Ollama JSON agents support automatic tool selection only")
        return self.bind(tools=list(tools), **kwargs)

    def _decision_agent(self, tools, stop, kwargs):
        descriptors = []
        for tool in tools:
            definition = convert_to_openai_tool(tool)["function"]
            # Descriptors are only rendered; the outer graph executes originals.
            descriptors.append(StructuredTool(
                name=definition["name"], description=definition.get("description", ""),
                args_schema=definition.get("parameters", {"type": "object", "properties": {}}),
            ))
        return create_structured_chat_agent(
            self.client.bind(**{**kwargs, "format": "json"}), descriptors, _PROMPT,
            stop_sequence=stop or True,
        )

    @staticmethod
    def _result(decision, capture):
        if isinstance(decision, AgentAction):
            if not isinstance(decision.tool_input, dict):
                raise ValueError("Ollama JSON tool arguments must be an object")
            message = AIMessage(content="", tool_calls=[{
                "name": decision.tool, "args": decision.tool_input, "id": "call_" + uuid4().hex,
            }])
        elif isinstance(decision, AgentFinish):
            message = AIMessage(content=decision.return_values["output"])
        else:
            raise TypeError("Structured chat returned neither an action nor a final answer")
        if capture.message is not None:
            message.additional_kwargs = capture.message.additional_kwargs
            message.usage_metadata = capture.message.usage_metadata
            message.response_metadata = capture.message.response_metadata
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        tools = kwargs.pop("tools", None)
        if not tools:
            return self.client._generate(messages, stop, run_manager, **kwargs)
        capture = _ResponseCapture()
        decision = self._decision_agent(tools, stop, kwargs).invoke(
            _payload(messages), config={"callbacks": [capture]},
        )
        return self._result(decision, capture)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        tools = kwargs.pop("tools", None)
        if not tools:
            return await self.client._agenerate(messages, stop, run_manager, **kwargs)
        capture = _ResponseCapture()
        decision = await self._decision_agent(tools, stop, kwargs).ainvoke(
            _payload(messages), config={"callbacks": [capture]},
        )
        return self._result(decision, capture)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        result = self._generate(messages, stop, **kwargs)
        data = result.generations[0].message.model_dump(exclude={"type"})
        yield ChatGenerationChunk(message=AIMessageChunk(**data, chunk_position="last"))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        result = await self._agenerate(messages, stop, **kwargs)
        data = result.generations[0].message.model_dump(exclude={"type"})
        yield ChatGenerationChunk(message=AIMessageChunk(**data, chunk_position="last"))
