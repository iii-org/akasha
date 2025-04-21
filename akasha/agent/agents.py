from langchain.tools import BaseTool
from typing import Union, List, Generator
import json
import datetime
import time
import logging
import asyncio
from akasha.utils.atman import basic_llm

from akasha.utils.base import (
    DEFAULT_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_INPUT_TOKENS,
)
from akasha.helper.preprocess_prompts import retri_history_messages
from akasha.helper.base import get_doc_length, extract_json
from akasha.utils.prompts.gen_prompt import format_sys_prompt
from akasha.helper.run_llm import call_model
from .base import get_tool_explaination


class agents(basic_llm):
    """basic class for akasha agent, implement _change_variables, _check_db, add_log and save_logs function."""

    def __init__(
        self,
        tools: Union[BaseTool, List],
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
    ):
        """initials of agent class

        Args:
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
            **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
            **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
                max token size of llm input.\n
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n
            **keep_logs (bool, optional)**: record logs or not. Defaults to False.\n
            ** max_round (int, optional)**: the maximum round of the conversation. Defaults to 20.\n
            ** max_doc_len (int, optional)**: the maximum length of the past thoughts and observations that will send to agent. Defaults to 1500.\n (deprecated in future 1.0.0 version)\n
            ** max_past_observation (int, optional)**: the maximum round of the past thoughts and observations that will send to agent. Defaults to 10.\n
            ** prompt_format_type (str, optional)**: the prompt and system prompt format for the language model, including auto, gpt, llama, chat_gpt, chat_mistral, chat_gemini . Defaults to "auto".
            ** retri_observation (bool, optional)**: if True, agent will ask LLM to retrieve the information from the past thoughts and observations. Defaults to False.\n
            **max_output_tokens (int, optional)**: max output tokens of llm model. Defaults to 1024.\n
            **max_input_tokens (int, optional)**: max input tokens of llm model. Defaults to 3600.\n
        """
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
        )
        self.stream = stream
        self.prompt_format_type = prompt_format_type
        self.max_round = max_round
        self.max_past_observation = max_past_observation
        self.retri_observation = retri_observation

        ## var ##
        self.messages = []
        self.thoughts = []

        self.tokens = 0
        self.input_len = 0
        self.question = ""

        self.tool_explaination = get_tool_explaination(tools)
        if not isinstance(tools, List):
            tools = [tools]
        self.tools = {}
        for tool in tools:
            if not isinstance(tool, BaseTool):
                logging.warning("tools should be a list of BaseTool")
                continue
            tool_name = tool.name
            self.tools[tool_name] = tool

        tool_explain_str = "\n".join(
            [
                f"{tool_name}: {tool_des}"
                for tool_name, tool_des in self.tool_explaination.items()
            ]
        )
        tool_name_str = ", ".join(
            [f'"{tool_name}"' for tool_name in self.tool_explaination.keys()]
        )
        self.tool_name_str = tool_name_str

        self.REACT_PROMPT = f"""Respond to the human as helpfully and accurately as possible. You have access to the following tools:\n\n{tool_explain_str}\n
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\n
Valid "action" values: "Answer" or {tool_name_str}\n\n
Provide only ONE action per $JSON_BLOB, as shown:\n{{\n```\n\n  "action": $TOOL_NAME,\n  "action_input": $INPUT\n}}\n```$INPUT is a dictionary that contains tool parameters and their values\n\n
the meaning of each format:\n
Question: input question to answer\nThought: consider previous and subsequent steps\nAction:\n```\n$JSON_BLOB\n```\nObservation: action result\n
... (repeat Thought/Action N times)\nThought: I know what to respond\nAction:\n```\n{{\n  "action": "Answer",\n  "action_input": "Final response to human"\n}}\n```\n\n
Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Thought: then Action:```$JSON_BLOB```.\n
"""
        self.REMEMBER_PROMPT = (
            "**Remember, Format is Thought: then Action:```$JSON_BLOB```\n\n"
        )
        self.OBSERVATION_PROMPT = "\n\nBelow are your previous work, check them carefully and provide the next action and thought,**do not ask same question repeatedly: "
        self.RETRI_OBSERVATION_PROMPT = "User will give you Question, Thought and Observation, return the information from Observation that you think is most relevant to the Question or Thought, if you can't find the information, return None."

    def _add_basic_log(self, timestamp: str, fn_type: str) -> bool:
        """add pre-process log to self.logs

        Args:
            timestamp (str): timestamp of this run
            fn_type (str): function type of this run
        """
        if super()._add_basic_log(timestamp, fn_type) is False:
            return False

        self.logs[timestamp]["question"] = self.question
        self.logs[timestamp]["max_round"] = self.max_round
        self.logs[timestamp]["max_input_tokens"] = self.max_input_tokens
        self.logs[timestamp]["max_past_observation"] = self.max_past_observation
        return True

    def _add_result_log(self, timestamp: str, time: float) -> bool:
        """add post-process log to self.logs

        Args:
            timestamp (str): timestamp of this run
            time (float): spent time of this run
        """

        if self.keep_logs is False:
            return False

        ### add token information ###
        tool_list = []
        for name, tool in self.tools.items():
            tool_list.append(name)

        self.logs[timestamp]["time"] = time
        self.logs[timestamp]["tools"] = tool_list
        self.logs[timestamp]["messages"] = self.messages
        self.logs[timestamp]["thoughts"] = self.thoughts
        self.logs[timestamp]["response"] = self.response
        self.logs[timestamp]["tokens"] = self.tokens
        self.logs[timestamp]["input_len"] = self.input_len
        return True

    def _display_info(self, batch: int = 1) -> bool:
        """display the information of the parameters if verbose is True"""
        if self.verbose is False:
            return False
        print(f"Model: {self.model}, Temperature: {self.temperature}")
        print("Tool: ", self.tool_name_str)
        print(
            f"Prompt format type: {self.prompt_format_type}, Max input tokens: {self.max_input_tokens}"
        )

        return True

    def __call__(self, question: str, messages: List[dict] = None):
        """Synchronous version of agent."""

        async def collect_stream():
            if self.stream:
                # Handle streaming case
                results = []
                async for chunk in self._run_agent_stream(
                    question,
                    tool_runner=lambda tool, tool_input: tool._run(**tool_input),
                    messages=messages,
                ):
                    results.append(chunk)
                return results
            else:
                # Handle non-streaming case
                return await self._run_agent(
                    question,
                    tool_runner=lambda tool, tool_input: tool._run(**tool_input),
                    messages=messages,
                )

        # Consume the async generator or coroutine and return the result
        return asyncio.run(collect_stream())

    async def acall(self, question: str, messages: List[dict] = None):
        """Asynchronous version of agent."""
        return await self._run_agent(
            question,
            tool_runner=lambda tool, tool_input: tool.ainvoke(tool_input),
            messages=messages,
        )

    async def _run_agent(
        self, question: str, tool_runner: callable, messages: List[dict] = None
    ):
        """run agent to get response"""
        if self.stream:
            return self._run_agent_stream(question, tool_runner, messages)

        start_time = time.time()
        round_count = self.max_round
        self.response = ""
        if messages is None:
            self.messages = []
        else:
            self.messages = messages
        self.thoughts = []
        observation = ""
        thought = ""
        retri_messages = ""
        self._display_info()
        ### call model to get response ###
        retri_messages, messages_len = retri_history_messages(
            self.messages,
            self.max_past_observation,
            self.max_input_tokens,
            self.model,
            "Action",
            "Observation",
        )
        if retri_messages != "":
            retri_messages = self.OBSERVATION_PROMPT + retri_messages + "\n\n"

        text_input = format_sys_prompt(
            self.REACT_PROMPT,
            "Question: " + question + retri_messages + self.REMEMBER_PROMPT,
            self.prompt_format_type,
            self.model,
        )

        response = call_model(self.model_obj, text_input)

        txt = (
            "Question: "
            + " think step by step"
            + question
            + self.REMEMBER_PROMPT
            + self.REACT_PROMPT
            + retri_messages
        )
        self.input_len = get_doc_length(self.language, txt)
        self.tokens = self.model_obj.get_num_tokens(txt)

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        if self.keep_logs is True:
            self.timestamp_list.append(timestamp)
            self._add_basic_log(timestamp, "agent_call")

        ### start to run agent ###
        while round_count > 0:
            try:
                cur_action = extract_json(response)
                if (not isinstance(cur_action["action"], str)) or (
                    not isinstance(cur_action["action_input"], dict)
                    and (cur_action["action"] != "Answer")
                ):
                    raise ValueError("Cannot find correct action from response")
            except Exception:
                logging.warning(
                    "Cannot extract JSON format action from response, retry."
                )
                text_input = format_sys_prompt(
                    self.REACT_PROMPT,
                    "Question: " + question + retri_messages + self.REMEMBER_PROMPT,
                    self.prompt_format_type,
                    self.model,
                )
                response = call_model(self.model_obj, text_input)
                round_count -= 1
                txt = (
                    "Question: "
                    + question
                    + retri_messages
                    + self.REACT_PROMPT
                    + self.REMEMBER_PROMPT
                )
                self.input_len += get_doc_length(self.language, txt)
                self.tokens += self.model_obj.get_num_tokens(txt)
                continue

            ### get thought from response ###
            thought = "".join(response.split("Thought:")[1:]).split("Action:")[0]
            if thought.replace(" ", "").replace("\n", "") == "":
                thought = "None."
            self.thoughts.append(thought)

            if cur_action is None:
                raise ValueError("Cannot find correct action from response")
            if cur_action["action"].lower() in [
                "final answer",
                "final_answer",
                "final",
                "answer",
            ]:
                if isinstance(cur_action["action_input"], (dict, list)):
                    response = json.dumps(
                        cur_action["action_input"], ensure_ascii=False
                    )
                else:
                    response = str(cur_action["action_input"])

                self.messages.append(
                    {
                        "role": "Action",
                        "content": json.dumps(cur_action, ensure_ascii=False),
                    }
                )

                break

            elif cur_action["action"] in self.tools:
                tool_name = cur_action["action"]
                tool_input = cur_action["action_input"]

                tool = self.tools[tool_name]
                result = tool_runner(tool, tool_input)
                if asyncio.iscoroutine(result):
                    firsthand_observation = await result  # Await async result
                else:
                    firsthand_observation = result  # Sync result

                if self.retri_observation:
                    text_input = format_sys_prompt(
                        self.RETRI_OBSERVATION_PROMPT,
                        "Question: "
                        + question
                        + "\n\nThought: "
                        + thought
                        + "\n\nObservation: "
                        + firsthand_observation,
                        self.prompt_format_type,
                        self.model,
                    )
                    observation = call_model(self.model_obj, text_input)
                    txt = (
                        "Question: "
                        + question
                        + "\n\nThought: "
                        + thought
                        + "\n\nObservation: "
                        + firsthand_observation
                        + self.RETRI_OBSERVATION_PROMPT
                    )
                    self.input_len += get_doc_length(self.language, txt)
                    self.tokens += self.model_obj.get_num_tokens(txt)
                else:
                    observation = firsthand_observation

                if self.verbose:
                    print("\nObservation: " + observation)

            else:
                raise ValueError(f"Cannot find tool {cur_action['action']}")

            cur_action["action_input"].pop("run_manager", None)
            self.messages.append(
                {
                    "role": "Action",
                    "content": json.dumps(cur_action, ensure_ascii=False),
                }
            )
            self.messages.append({"role": "Observation", "content": observation})

            retri_messages, messages_len = retri_history_messages(
                self.messages,
                self.max_past_observation,
                self.max_input_tokens,
                self.model,
                "Action",
                "Observation",
            )
            if retri_messages != "":
                retri_messages = self.OBSERVATION_PROMPT + retri_messages + "\n\n"

            text_input = format_sys_prompt(
                self.REACT_PROMPT,
                "Question: " + question + retri_messages,
                self.prompt_format_type,
                self.model,
            )
            response = call_model(self.model_obj, text_input)
            txt = "Question: " + question + retri_messages + self.REACT_PROMPT
            self.input_len += get_doc_length(self.language, txt)
            self.tokens += self.model_obj.get_num_tokens(txt)

            round_count -= 1

        end_time = time.time()
        print(
            "\n-------------------------------------\nSpend Time: ",
            end_time - start_time,
            "s\n",
        )
        self.response = response
        self._add_result_log(timestamp, end_time - start_time)

        return response

    async def _run_agent_stream(
        self, question: str, tool_runner: callable, messages: List[dict] = None
    ):
        """run agent stream to get response"""

        start_time = time.time()
        round_count = self.max_round
        self.response = ""
        if messages is None:
            self.messages = []
        else:
            self.messages = messages
        self.thoughts = []
        observation = ""
        thought = ""
        retri_messages = ""
        self._display_info()
        ### call model to get response ###
        retri_messages, messages_len = retri_history_messages(
            self.messages,
            self.max_past_observation,
            self.max_input_tokens,
            self.model,
            "Action",
            "Observation",
        )
        if retri_messages != "":
            retri_messages = self.OBSERVATION_PROMPT + retri_messages + "\n\n"

        text_input = format_sys_prompt(
            self.REACT_PROMPT,
            "Question: " + question + retri_messages + self.REMEMBER_PROMPT,
            self.prompt_format_type,
            self.model,
        )

        response = call_model(self.model_obj, text_input)

        txt = (
            "Question: "
            + " think step by step"
            + question
            + self.REMEMBER_PROMPT
            + self.REACT_PROMPT
            + retri_messages
        )
        self.input_len = get_doc_length(self.language, txt)
        self.tokens = self.model_obj.get_num_tokens(txt)

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        if self.keep_logs is True:
            self.timestamp_list.append(timestamp)
            self._add_basic_log(timestamp, "agent_call")

        ### start to run agent ###
        while round_count > 0:
            try:
                cur_action = extract_json(response)
                if (not isinstance(cur_action["action"], str)) or (
                    not isinstance(cur_action["action_input"], dict)
                    and (cur_action["action"] != "Answer")
                ):
                    raise ValueError("Cannot find correct action from response")
            except Exception:
                logging.warning(
                    "Cannot extract JSON format action from response, retry."
                )
                text_input = format_sys_prompt(
                    self.REACT_PROMPT,
                    "Question: " + question + retri_messages + self.REMEMBER_PROMPT,
                    self.prompt_format_type,
                    self.model,
                )
                response = call_model(self.model_obj, text_input)
                round_count -= 1
                txt = (
                    "Question: "
                    + question
                    + retri_messages
                    + self.REACT_PROMPT
                    + self.REMEMBER_PROMPT
                )
                self.input_len += get_doc_length(self.language, txt)
                self.tokens += self.model_obj.get_num_tokens(txt)
                continue

            ### get thought from response ###
            thought = "".join(response.split("Thought:")[1:]).split("Action:")[0]
            if thought.replace(" ", "").replace("\n", "") == "":
                thought = "None."
            self.thoughts.append(thought)

            # yield thought #
            intermed_stream_text = "\nThought: " + str(thought) + "\n"
            yield intermed_stream_text

            if cur_action is None:
                raise ValueError("Cannot find correct action from response")
            if cur_action["action"].lower() in [
                "final answer",
                "final_answer",
                "final",
                "answer",
            ]:
                if isinstance(cur_action["action_input"], (dict, list)):
                    response = json.dumps(
                        cur_action["action_input"], ensure_ascii=False
                    )
                else:
                    response = str(cur_action["action_input"])

                self.messages.append(
                    {
                        "role": "Action",
                        "content": json.dumps(cur_action, ensure_ascii=False),
                    }
                )

                yield response

                break

            elif cur_action["action"] in self.tools:
                tool_name = cur_action["action"]
                tool_input = cur_action["action_input"]

                # yield action #
                intermed_stream_text = (
                    "\nAction: "
                    + tool_name
                    + ", "
                    + json.dumps(cur_action, ensure_ascii=False)
                    + "\n"
                )
                yield intermed_stream_text

                tool = self.tools[tool_name]
                result = tool_runner(tool, tool_input)
                if asyncio.iscoroutine(result):
                    firsthand_observation = await result  # Await async result
                else:
                    firsthand_observation = result  # Sync result

                if self.retri_observation:
                    text_input = format_sys_prompt(
                        self.RETRI_OBSERVATION_PROMPT,
                        "Question: "
                        + question
                        + "\n\nThought: "
                        + thought
                        + "\n\nObservation: "
                        + firsthand_observation,
                        self.prompt_format_type,
                        self.model,
                    )
                    observation = call_model(self.model_obj, text_input)
                    txt = (
                        "Question: "
                        + question
                        + "\n\nThought: "
                        + thought
                        + "\n\nObservation: "
                        + firsthand_observation
                        + self.RETRI_OBSERVATION_PROMPT
                    )
                    self.input_len += get_doc_length(self.language, txt)
                    self.tokens += self.model_obj.get_num_tokens(txt)
                else:
                    observation = firsthand_observation

                if self.verbose:
                    print("\nObservation: " + observation)
                # yield observation #
                intermed_stream_text = "\nObservation: " + str(observation) + "\n"
                yield intermed_stream_text

            else:
                raise ValueError(f"Cannot find tool {cur_action['action']}")

            cur_action["action_input"].pop("run_manager", None)
            self.messages.append(
                {
                    "role": "Action",
                    "content": json.dumps(cur_action, ensure_ascii=False),
                }
            )
            self.messages.append({"role": "Observation", "content": observation})

            retri_messages, messages_len = retri_history_messages(
                self.messages,
                self.max_past_observation,
                self.max_input_tokens,
                self.model,
                "Action",
                "Observation",
            )
            if retri_messages != "":
                retri_messages = self.OBSERVATION_PROMPT + retri_messages + "\n\n"

            text_input = format_sys_prompt(
                self.REACT_PROMPT,
                "Question: " + question + retri_messages,
                self.prompt_format_type,
                self.model,
            )
            response = call_model(self.model_obj, text_input)
            txt = "Question: " + question + retri_messages + self.REACT_PROMPT
            self.input_len += get_doc_length(self.language, txt)
            self.tokens += self.model_obj.get_num_tokens(txt)

            round_count -= 1

        end_time = time.time()
        print(
            "\n-------------------------------------\nSpend Time: ",
            end_time - start_time,
            "s\n",
        )
        self.response = response
        self._add_result_log(timestamp, end_time - start_time)

        return

    def _final_ronud_stream(self, response: str) -> Generator[str, None, None]:
        """final round stream"""
        for c in response:
            self.response += c
            yield c

    def _display_stream(self, text: str) -> Generator[str, None, None]:
        """display stream"""
        for c in text:
            yield c
