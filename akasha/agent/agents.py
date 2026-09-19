"""LangChain-native Agent facade used by Akasha."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import time
from typing import Any, Generator, List, Sequence, Union

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import BaseTool

from akasha.agent.skills import (
    DynamicSkillMiddleware,
    Skill,
    SkillContext,
    SkillToolContext,
)
from akasha.helper.base import get_doc_length
from akasha.helper.run_llm import content_to_text, content_to_thinking
from akasha.utils.atman import basic_llm
from akasha.utils.models.thinking import ThinkingBudget
from akasha.utils.base import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
)

logger = logging.getLogger("akasha.agent")

_PROGRESS_PROMPT = """User-visible progress reporting:
When using tools, you may briefly explain the purpose of the operation.
Use the user's language. Describe observable actions, not private reasoning.
Do not invent results or claim success before a tool completes.
Your answer may include concise progress and verified findings together with
the requested result. Directly answer the user's question in the current
response; do not merely announce completion or promise to answer later.
Do not split an otherwise complete answer into separate messages just to
separate progress from the answer. Akasha handles progress, tool, and answer
labels; do not add these labels yourself.
These reporting instructions do not change the user's task or output constraints.
"""


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        return content_to_text(message.get("content", ""))
    return content_to_text(getattr(message, "content", message))


def _message_dump(message: Any) -> Any:
    try:
        return message.model_dump(mode="json")
    except Exception:
        try:
            return message.dict()
        except Exception:
            return {"type": type(message).__name__, "content": _message_text(message)}


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)


def _trace_text(value: Any, limit: int = 4_000) -> str:
    safe = _json_safe(value)
    if isinstance(safe, str):
        text = safe
    else:
        text = json.dumps(safe, ensure_ascii=False, indent=2)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [verbose output truncated]"


def _thinking_text(message: Any) -> str:
    blocks = getattr(message, "content_blocks", None) or []
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") in {"reasoning", "thinking"}:
            value = block.get("reasoning") or block.get("thinking") or block.get("text")
            if value:
                parts.append(str(value))
    if parts:
        return "".join(parts)

    value = content_to_thinking(getattr(message, "content", None))
    if value:
        return value

    extra = getattr(message, "additional_kwargs", {}) or {}
    for key in ("reasoning_content", "thinking", "reasoning"):
        value = extra.get(key)
        if value:
            return value if isinstance(value, str) else str(value)
    return ""


def _as_message_list(messages: List[dict] | None) -> list:
    """Convert legacy history roles to messages accepted by create_agent."""
    if not messages:
        return []
    converted = []
    for message in messages:
        if not isinstance(message, dict):
            converted.append(message)
            continue
        role = message.get("role", "user")
        content = message.get("content", "")
        if role in {"Action", "Observation"}:
            role = "assistant" if role == "Action" else "user"
        converted.append({"role": role, "content": content})
    return converted


def _extract_messages(result: Any) -> list:
    if isinstance(result, dict):
        messages = result.get("messages", [])
        return messages if isinstance(messages, list) else list(messages)
    return []


def _last_answer(messages: list) -> str:
    for message in reversed(messages):
        if getattr(message, "tool_calls", None):
            return ""
        if isinstance(message, (AIMessage, AIMessageChunk)):
            text = _message_text(message)
            if text:
                return text
        elif getattr(message, "type", None) == "ai":
            text = _message_text(message)
            if text:
                return text
    return ""


def _loaded_skill_events(messages: list) -> list[dict[str, str]]:
    """Extract completed ``load_skill`` calls from agent messages."""
    references: dict[str, str] = {}
    events = []
    for message in messages:
        if isinstance(message, (AIMessage, AIMessageChunk)):
            for call in getattr(message, "tool_calls", None) or []:
                if call.get("name") == "load_skill":
                    reference = call.get("args", {}).get("reference")
                    if reference:
                        references[str(call.get("id", ""))] = str(reference)
        elif isinstance(message, ToolMessage) and getattr(message, "name", None) == "load_skill":
            reference = references.get(str(getattr(message, "tool_call_id", "")))
            if reference:
                events.append(
                    {"reference": reference, "message": _message_text(message)}
                )
    return events


def _stream_update_messages(update: Any) -> list:
    """Read messages from LangGraph v1/v2 update stream shapes."""
    payload = update
    if (
        isinstance(update, dict)
        and update.get("type") == "updates"
        and isinstance(update.get("data"), dict)
    ):
        payload = update["data"]

    if not isinstance(payload, dict):
        return []

    messages = []
    for node_update in payload.values():
        if isinstance(node_update, dict):
            node_messages = node_update.get("messages", [])
            if isinstance(node_messages, list):
                messages.extend(node_messages)
        elif isinstance(node_update, (AIMessage, AIMessageChunk, ToolMessage)):
            messages.append(node_update)
    return messages


def _stream_message_chunk(update: Any) -> Any:
    """Extract the message chunk from LangGraph ``messages`` stream output."""
    if isinstance(update, tuple) and update:
        return update[0]
    if isinstance(update, (AIMessage, AIMessageChunk, ToolMessage)):
        return update
    return None


def _count_tokens(model: Any, text: str) -> int:
    try:
        return model.get_num_tokens(text)
    except Exception:
        return len(text)


class _ProgressCallbacks(BaseCallbackHandler):
    """Observe completed agent model turns before LangGraph executes tools."""

    run_inline = True

    def __init__(self, agent, history=None):
        self.agent = agent
        self.model_runs = set()
        self.calls = set()
        self.results = set()
        for message in _as_message_list(history):
            for call in getattr(message, "tool_calls", None) or []:
                self.calls.add(call.get("id"))
            if isinstance(message, ToolMessage):
                self.results.add(message.tool_call_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        if (metadata or {}).get("langgraph_node") == "model":
            self.model_runs.add(run_id)

    def on_llm_end(self, response, *, run_id, **kwargs):
        if run_id not in self.model_runs:
            return
        self.model_runs.discard(run_id)
        for generations in response.generations:
            for generation in generations:
                self.message(getattr(generation, "message", None))

    def message(self, message):
        calls = getattr(message, "tool_calls", None) or []
        fresh = [call for call in calls if call.get("id") not in self.calls]
        if fresh:
            for text in self.agent._progress_for_message(message):
                self.agent._display_progress(text)
            for call in fresh:
                self.calls.add(call.get("id"))
                self.agent._display_tool_call(call)
        if isinstance(message, ToolMessage) and message.tool_call_id not in self.results:
            self.results.add(message.tool_call_id)
            self.agent._display_tool_result(message)

    def on_tool_end(self, output, **kwargs):
        if isinstance(output, ToolMessage):
            self.message(output)


class agents(basic_llm):
    """Reusable tool-calling agent facade over LangChain ``create_agent``."""

    def __init__(
        self,
        tools: Union[BaseTool, List] | None = None,
        skills: str | Skill | Sequence[str | Skill] | None = None,
        model: str = DEFAULT_MODEL,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        prompt_format_type: str = "auto",
        max_round: int = 20,
        max_past_observation: int = 10,
        language: str = "ch",
        record_exp: str = "",
        system_prompt: str = "",
        retri_observation: bool = False,
        keep_logs: bool = True,
        verbose: bool = False,
        stream: bool = False,
        env_file: str = "",
        thinking: bool = False,
        thinking_budget: ThinkingBudget = None,
        max_resource_bytes: int = 128 * 1024,
    ) -> None:
        super().__init__(
            model=model,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            language=language,
            record_exp=record_exp,
            system_prompt=system_prompt,
            keep_logs=keep_logs,
            verbose=verbose,
            env_file=env_file,
            thinking=thinking,
            thinking_budget=thinking_budget,
        )
        self.stream = stream
        self.prompt_format_type = prompt_format_type
        self.max_round = max_round
        self.max_past_observation = max_past_observation
        self.retri_observation = retri_observation
        self.messages: list = []
        self.thoughts: list = []
        self.progress: list[str] = []
        self.tool_calls: list = []
        self.tokens = 0
        self.input_len = 0
        self.question = ""
        if tools is None:
            tools = []
        if isinstance(tools, BaseTool):
            tools = [tools]
        self.tools = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}
        if len(self.tools) != len(tools):
            logger.warning("tools should be a list of BaseTool")
        self.tool_name_str = ", ".join(f'"{name}"' for name in self.tools)
        self.tool_explaination = {
            name: tool.description for name, tool in self.tools.items()
        }
        self.max_resource_bytes = max_resource_bytes
        self.skill_references = skills
        self._skills_enabled = self._has_skill_references(skills)
        self.skill_context: SkillContext = SkillContext()
        self.skill_tools = ()
        self.skill_tool_names: dict[str, tuple[str, ...]] = {}
        self.skill_middleware: DynamicSkillMiddleware | None = None
        self._agent = self._build_agent()

    def _set_model(self, **kwargs):
        previous = getattr(self, "model_obj", None)
        super()._set_model(**kwargs)
        if getattr(self, "model_obj", None) is not previous:
            self._agent = self._build_agent()

    @staticmethod
    def _has_skill_references(skills) -> bool:
        if skills is None:
            return False
        if isinstance(skills, (str, Skill)):
            return True
        return bool(skills)

    def _build_agent(self):
        effective_prompt = "\n\n".join(
            part for part in (self.system_prompt.strip(), _PROGRESS_PROMPT) if part
        )
        kwargs = {
            "model": self.model_obj,
            "tools": list(self.tools.values()),
        }
        if self._skills_enabled:
            self.skill_tool_context = SkillToolContext(
                env_file=self.env_file,
                language=self.language,
                model=self.model,
            )
            self.skill_middleware = DynamicSkillMiddleware(
                self.skill_references,
                base_prompt=effective_prompt,
                tool_context=self.skill_tool_context,
                existing_tools=list(self.tools.values()),
                max_resource_bytes=self.max_resource_bytes,
            )
            self.skill_context = self.skill_middleware.available_context
            kwargs["middleware"] = [self.skill_middleware]
        else:
            kwargs["system_prompt"] = effective_prompt
        return create_agent(**kwargs)
    def _display_thinking_info(self) -> None:
        message = (
            "Thinking: %s, Thinking budget level: %s, "
            "Effective thinking budget: %s"
        )
        if self.verbose or self.keep_logs:
            self._emit_trace(
                message
                % (
                    self.thinking,
                    self.thinking_budget_level,
                    self.effective_thinking_budget,
                )
            )

    def _emit_trace(self, text: str) -> None:
        """Write one complete agent trace line to enabled debug channels."""
        if self.verbose:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prefix = f"\033[32m[akasha {timestamp}]\033[0m"
            trace_label = "\033[33m[agent.trace]\033[0m "
            compatibility_label = "[akasha] " if text.startswith("tool ") else ""
            console_text = trace_label + compatibility_label + text
            if "stderr:\n" in console_text:
                before, error = console_text.split("stderr:\n", 1)
                console_text = (
                    before
                    + "\033[31mstderr:\n"
                    + error
                    + "\033[0m"
                )
            print(f"{prefix} {console_text}")
        if self.keep_logs:
            logging.getLogger("akasha.agent").info(
                f"[agent.logger] {text}",
                extra={"akasha_trace": True},
            )
    def _record_timing(
        self,
        timestamp: str,
        started_at: datetime.datetime,
        ended_at: datetime.datetime,
        elapsed: float,
    ) -> None:
        if self.keep_logs:
            self.logs[timestamp].update(
                {
                    "start_time": started_at.isoformat(timespec="milliseconds"),
                    "end_time": ended_at.isoformat(timespec="milliseconds"),
                    "elapsed_seconds": elapsed,
                }
            )

    def _display_timing(
        self,
        phase: str,
        moment: datetime.datetime,
        elapsed: float | None = None,
    ) -> None:
        if not (self.verbose or self.keep_logs):
            return
        message = f"{phase}: {moment.isoformat(timespec='milliseconds')}"
        if elapsed is not None:
            message += f" elapsed={elapsed:.3f}s"
        self._emit_trace(message)
    def _display_stream_event(self, event_type: str, data: Any) -> None:
        """Display consumed stream data when verbose mode is enabled."""
        if not self.verbose:
            return
        if event_type == "thinking":
            self._emit_trace(f"stream.thinking: {data}")
        elif event_type == "answer":
            print(str(data), end="", flush=True)
    def _display_tool_call(self, call: Any) -> None:
        if not (self.verbose or self.keep_logs):
            return
        call = _json_safe(call)
        name = (
            call.get("name", "unknown")
            if isinstance(call, dict)
            else "unknown"
        )
        args = call.get("args", {}) if isinstance(call, dict) else call
        self._emit_trace(f"tool call: {name}\n[tool] 呼叫 {name}\n  args: {_trace_text(args)}")

    def _display_tool_result(self, message: ToolMessage) -> None:
        if not (self.verbose or self.keep_logs):
            return
        name = getattr(message, "name", None) or "unknown"
        self._emit_trace(
            f"tool result: {name}\n[tool] {name} 回傳結果\n  result: {_trace_text(_message_text(message))}"
        )

    def _progress_for_message(self, message: Any) -> list[str]:
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            return []
        text = _message_text(message).strip()
        return [text] if text else [
            f"準備呼叫工具 {call['name']}。" for call in calls
        ]

    def _display_progress(self, text: str) -> None:
        self.progress.append(text)
        self._emit_trace(f"[progress] {text}")

    def _display_tool_trace(self, messages: list) -> None:
        if not (self.verbose or self.keep_logs):
            return
        for message in messages:
            if isinstance(message, (AIMessage, AIMessageChunk)):
                for call in getattr(message, "tool_calls", None) or []:
                    self._display_tool_call(call)
            elif isinstance(message, ToolMessage):
                self._display_tool_result(message)
        for event in _loaded_skill_events(messages):
            self._emit_trace(f"skill loaded: {event['reference']}")

    def _record_result(self, timestamp: str, result: Any, elapsed: float) -> None:
        if self.skill_middleware is not None:
            self.skill_tool_names = self.skill_middleware.loaded_skill_tools
            self.skill_tools = tuple(self.skill_middleware._loaded_tools.values())
        messages = _extract_messages(result)
        self.messages = [_message_dump(message) for message in messages]
        self.tool_calls = []
        thinking = []
        for message in messages:
            if isinstance(message, (AIMessage, AIMessageChunk)):
                if getattr(message, "tool_calls", None):
                    self.tool_calls.extend(message.tool_calls)
                value = _thinking_text(message)
                if value:
                    thinking.append(value)
        self.thoughts = thinking
        self.response = _last_answer(messages)
        if self.keep_logs:
            self.logs[timestamp].update(
                {
                    "time": elapsed,
                    "messages": self.messages,
                    "tool_calls": _json_safe(self.tool_calls),
                    "thinking": "".join(thinking),
                    "response": self.response,
                    "progress": list(self.progress),
                    "model": self.model,
                    "provider": self.model.split(":", 1)[0],
                    "tokens": self.tokens,
                    "input_len": self.input_len,
                    "loaded_skills": _loaded_skill_events(messages),
                }
            )

    def _payload(self, question: str, messages: List[dict] | None) -> dict:
        history = _as_message_list(messages)
        history.append({"role": "user", "content": question})
        return {"messages": history}

    def __call__(
        self,
        question: str,
        messages: List[dict[str, Any]] | None = None,
        include_thinking: bool | None = None,
    ) -> str | Generator[dict[str, Any], None, None]:
        self.question = question
        if self.stream:
            self._ensure_stream_supported()
            return self._stream(question, messages, include_thinking)
        return asyncio.run(self._ainvoke(question, messages))

    async def acall(
        self,
        question: str,
        messages: List[dict[str, Any]] | None = None,
        include_thinking: bool | None = None,
    ) -> str | Generator[dict[str, Any], None, None]:
        self.question = question
        if self.stream:
            self._ensure_stream_supported()
            return self._stream(question, messages, include_thinking)
        return await self._ainvoke(question, messages)

    async def _ainvoke(self, question: str, messages: List[dict] | None):
        self.progress = []
        callbacks = _ProgressCallbacks(self, messages)
        self._display_thinking_info()
        started_at = datetime.datetime.now()
        start = time.time()
        timestamp = started_at.strftime("%Y/%m/%d, %H:%M:%S")
        self._display_timing("start", started_at)
        if self.keep_logs:
            self.timestamp_list.append(timestamp)
            self.logs[timestamp] = {
                "fn_type": "agent_call",
                "start_time": started_at.isoformat(timespec="milliseconds"),
                "question": question,
                "model": self.model,
                "tools": list(self.tools),
                "skills": self.skill_context.names,
                "skill_versions": self.skill_context.versions,
                "skill_tools": self.skill_tool_names,
                "thinking": self.thinking,
                "thinking_budget_level": self.thinking_budget_level,
                "effective_thinking_budget": self.effective_thinking_budget,
            }
        self.input_len = get_doc_length(self.language, question)
        self.tokens = _count_tokens(self.model_obj, question)
        result = await self._agent.ainvoke(
            self._payload(question, messages),
            config={
                "recursion_limit": max(3, self.max_round * 2 + 1),
                "callbacks": [callbacks],
            },
        )
        result_messages = _extract_messages(result)
        for message in result_messages:
            callbacks.message(message)
        for event in _loaded_skill_events(result_messages):
            self._emit_trace(f"skill loaded: {event['reference']}")
        elapsed = time.time() - start
        ended_at = datetime.datetime.now()
        self._record_result(timestamp, result, elapsed)
        self._record_timing(timestamp, started_at, ended_at, elapsed)
        self._display_timing("end", ended_at, elapsed)
        if not self.response:
            raise RuntimeError("LangChain agent returned no final answer")
        if self.verbose:
            print(f"[answer] {self.response}")
        return self.response

    def _stream(
        self, question: str, messages: List[dict] | None, include_thinking: bool | None
    ) -> Generator[dict, None, None]:
        self._display_thinking_info()
        started_at = datetime.datetime.now()
        start = time.time()
        timestamp = started_at.strftime("%Y/%m/%d, %H:%M:%S")
        self._display_timing("start", started_at)
        collected = []
        answer_parts = []
        thinking_parts = []
        self.progress = []
        pending = None

        def finish_turn():
            nonlocal pending
            if pending is None:
                return
            message, pending = pending, None
            collected.append(message)
            calls = getattr(message, "tool_calls", None) or []
            if calls:
                for text in self._progress_for_message(message):
                    self._display_progress(text)
                    yield {"type": "progress", "data": text}
                for call in calls:
                    self._display_tool_call(call)
            else:
                text = _message_text(message)
                if text:
                    if self.verbose:
                        print("\033[33m[agent.answer] [answer]\033[0m ", end="", flush=True)
                    answer_parts.append(text)
                    self._display_stream_event("answer", text)
                    if self.verbose:
                        print()
                    yield {"type": "answer", "data": text}
        include_thinking = self.thinking if include_thinking is None else include_thinking
        if self.keep_logs:
            self.timestamp_list.append(timestamp)
            self.logs[timestamp] = {
                "fn_type": "agent_call",
                "start_time": started_at.isoformat(timespec="milliseconds"),
                "question": question,
                "model": self.model,
                "tools": list(self.tools),
                "skills": self.skill_context.names,
                "skill_versions": self.skill_context.versions,
                "skill_tools": self.skill_tool_names,
            }
        try:
            stream_kwargs = {
                "input": self._payload(question, messages),
                "config": {"recursion_limit": max(3, self.max_round * 2 + 1)},
                "stream_mode": "messages",
            }
            updates = self._agent.stream(
                stream_kwargs["input"],
                config=stream_kwargs["config"],
                stream_mode=stream_kwargs["stream_mode"],
            )

            # Buffer assistant text until tool-call intent is known. Reasoning
            # events remain incremental and independent from progress.
            for update in updates:
                message = _stream_message_chunk(update)
                messages = [message] if message is not None else _stream_update_messages(update)
                for message in messages:
                    if isinstance(message, ToolMessage):
                        yield from finish_turn()
                        collected.append(message)
                        self._display_tool_result(message)
                        yield {"type": "tool", "data": _message_dump(message)}
                        continue
                    if not isinstance(message, (AIMessage, AIMessageChunk)):
                        continue
                    if (
                        isinstance(update, tuple) and len(update) > 1
                        and isinstance(update[1], dict)
                        and update[1].get("langgraph_node", "model") != "model"
                    ):
                        continue
                    if pending is not None and (
                        not isinstance(message, AIMessageChunk)
                        or (message.id and pending.id and message.id != pending.id)
                    ):
                        yield from finish_turn()
                    pending = message if pending is None else pending + message
                    thinking = _thinking_text(message)
                    if include_thinking and thinking:
                        thinking_parts.append(thinking)
                        self._display_stream_event("thinking", thinking)
                        yield {"type": "thinking", "data": thinking}
                    if not isinstance(message, AIMessageChunk) or getattr(message, "chunk_position", None) == "last":
                        yield from finish_turn()
            yield from finish_turn()
            result = {"messages": collected}
            elapsed = time.time() - start
            ended_at = datetime.datetime.now()
            self._record_result(timestamp, result, elapsed)
            self._record_timing(timestamp, started_at, ended_at, elapsed)
            self._display_timing("end", ended_at, elapsed)
            self.response = "".join(answer_parts)
            if self.verbose and self.response:
                print()
            self.thoughts = thinking_parts
            if self.keep_logs:
                self.logs[timestamp]["response"] = self.response
                self.logs[timestamp]["thinking"] = "".join(thinking_parts)
            if not self.response:
                raise RuntimeError("LangChain agent returned no final answer")
        except Exception:
            logger.exception("LangChain agent streaming failed")
            raise

    def _has_async_only_tools(self) -> bool:
        """Return whether the agent contains tools that cannot sync-invoke.

        MCP tools produced by ``langchain-mcp-adapters`` expose only a
        coroutine. LangGraph's synchronous ToolNode calls ``tool.invoke`` and
        therefore raises for those tools.
        """
        return any(
            getattr(tool, "coroutine", None) is not None
            and getattr(tool, "func", None) is None
            for tool in self.tools.values()
        )

    def _ensure_stream_supported(self) -> None:
        """Reject sync streaming when the agent contains async-only tools."""
        if self._has_async_only_tools():
            raise ValueError(
                "MCP tools are async-only; construct the agent with stream=False "
                "so the agent can await the complete tool result via ainvoke()."
            )
