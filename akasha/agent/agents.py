"""LangChain-native Agent facade used by Akasha."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import time
from typing import Any, Generator, List, Sequence, Union

from langchain.agents import create_agent
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
        if isinstance(message, (AIMessage, AIMessageChunk)):
            text = _message_text(message)
            if text:
                return text
        elif getattr(message, "type", None) == "ai":
            text = _message_text(message)
            if text:
                return text
    return ""


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


class agents(basic_llm):
    """Public Akasha facade over LangChain 1.3+ ``create_agent``."""

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
    ):
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
                base_prompt=self.system_prompt,
                tool_context=self.skill_tool_context,
                existing_tools=list(self.tools.values()),
                max_resource_bytes=self.max_resource_bytes,
            )
            self.skill_context = self.skill_middleware.available_context
            kwargs["middleware"] = [self.skill_middleware]
        elif self.system_prompt.strip():
            kwargs["system_prompt"] = self.system_prompt
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
            console_text = (
                "\033[33m[agent.trace]\033[0m "
                + text
            )
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
        self._emit_trace(f"tool call: {name}\n  args: {_trace_text(args)}")

    def _display_tool_result(self, message: ToolMessage) -> None:
        if not (self.verbose or self.keep_logs):
            return
        name = getattr(message, "name", None) or "unknown"
        self._emit_trace(
            f"tool result: {name}\n  result: {_trace_text(_message_text(message))}"
        )

    def _display_tool_trace(self, messages: list) -> None:
        if not (self.verbose or self.keep_logs):
            return
        for message in messages:
            if isinstance(message, (AIMessage, AIMessageChunk)):
                for call in getattr(message, "tool_calls", None) or []:
                    self._display_tool_call(call)
            elif isinstance(message, ToolMessage):
                self._display_tool_result(message)

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
                    "model": self.model,
                    "provider": self.model.split(":", 1)[0],
                    "tokens": self.tokens,
                    "input_len": self.input_len,
                }
            )

    def _payload(self, question: str, messages: List[dict] | None) -> dict:
        history = _as_message_list(messages)
        history.append({"role": "user", "content": question})
        return {"messages": history}

    def __call__(
        self,
        question: str,
        messages: List[dict] = None,
        include_thinking: bool | None = None,
    ):
        self.question = question
        if self.stream:
            self._ensure_stream_supported()
            return self._stream(question, messages, include_thinking)
        return asyncio.run(self._ainvoke(question, messages))

    async def acall(
        self,
        question: str,
        messages: List[dict] = None,
        include_thinking: bool | None = None,
    ):
        self.question = question
        if self.stream:
            self._ensure_stream_supported()
            return self._stream(question, messages, include_thinking)
        return await self._ainvoke(question, messages)

    async def _ainvoke(self, question: str, messages: List[dict] | None):
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
            config={"recursion_limit": max(3, self.max_round * 2 + 1)},
        )
        self._display_tool_trace(_extract_messages(result))
        elapsed = time.time() - start
        ended_at = datetime.datetime.now()
        self._record_result(timestamp, result, elapsed)
        self._record_timing(timestamp, started_at, ended_at, elapsed)
        self._display_timing("end", ended_at, elapsed)
        if not self.response:
            raise RuntimeError("LangChain agent returned no final answer")
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
        displayed_tool_calls = set()
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

            # ``messages`` yields AIMessageChunk/ToolMessage chunks, so
            # callers receive token-level answer/thinking events.
            for update in updates:
                message = _stream_message_chunk(update)
                messages = [message] if message is not None else _stream_update_messages(update)
                for message in messages:
                    if isinstance(message, ToolMessage):
                        collected.append(message)
                        self._display_tool_result(message)
                        yield {"type": "tool", "data": _message_dump(message)}
                        continue
                    collected.append(message)
                    for call in getattr(message, "tool_calls", None) or []:
                        safe_call = _json_safe(call)
                        identity = json.dumps(
                            safe_call, ensure_ascii=False, sort_keys=True
                        )
                        if identity not in displayed_tool_calls:
                            displayed_tool_calls.add(identity)
                            self._display_tool_call(call)
                    thinking = _thinking_text(message)
                    if include_thinking and thinking:
                        thinking_parts.append(thinking)
                        self._display_stream_event("thinking", thinking)
                        yield {"type": "thinking", "data": thinking}
                    text = _message_text(message)
                    if text:
                        if self.verbose and not answer_parts:
                            print("\033[33m[agent.answer]\033[0m ", end="", flush=True)
                        answer_parts.append(text)
                        self._display_stream_event("answer", text)
                        yield {"type": "answer", "data": text}
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
