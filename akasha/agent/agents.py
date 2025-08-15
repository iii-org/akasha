from langchain.tools import BaseTool
from typing import Union, List
import json
import datetime
import time
import logging
import asyncio
import queue
import threading
from langchain_mcp_adapters.client import MultiServerMCPClient
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
from .base import (
    get_tool_explaination,
    get_REACT_PROMPT,
    DEFAULT_REMEMBER_PROMPT,
    DEFAULT_OBSERVATION_PROMPT,
    DEFAULT_RETRI_OBSERVATION_PROMPT,
)


class agents(basic_llm):
    """basic class for akasha agent, implement _change_variables, _check_db, add_log and save_logs function."""

    def __init__(
        self,
        tools: Union[BaseTool, List] = [],
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
        self.tools = {}
        ## if tools is mcp connection info, connect mcp client to get tools when acall ##

        if isinstance(tools, BaseTool):
            tools = [tools]

        self.tool_explaination = get_tool_explaination(tools)
        for tool in tools:
            if not isinstance(tool, BaseTool):
                logging.warning("tools should be a list of BaseTool")
                print("WARNING. tools should be a list of BaseTool\n\n")
                continue
            tool_name = tool.name
            self.tools[tool_name] = tool

        self.REACT_PROMPT = self._merge_tool_explaination_and_react()
        self.REMEMBER_PROMPT = DEFAULT_REMEMBER_PROMPT
        self.OBSERVATION_PROMPT = DEFAULT_OBSERVATION_PROMPT
        self.RETRI_OBSERVATION_PROMPT = DEFAULT_RETRI_OBSERVATION_PROMPT

    def _merge_tool_explaination_and_react(self) -> str:
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

        return get_REACT_PROMPT(tool_explain_str, tool_name_str)

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

        self.question = question

        if not self.stream:
            # Non-streaming: just run and return the result
            async def collect_non_stream():
                return await self._run_agent(
                    question,
                    tool_runner=lambda tool, tool_input: tool._run(**tool_input),
                    messages=messages,
                )

            return asyncio.run(collect_non_stream())

        # Streaming: yield results in real time
        message_queue = queue.Queue()
        stop_event = threading.Event()

        def run_async_stream():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def process_stream():
                try:
                    async for chunk in self._run_agent_stream(
                        question,
                        tool_runner=lambda tool, tool_input: tool._run(**tool_input),
                        messages=messages,
                    ):
                        message_queue.put(chunk)
                except Exception:
                    import traceback

                    message_queue.put(("ERROR", traceback.format_exc()))
                finally:
                    message_queue.put(None)

            try:
                loop.run_until_complete(process_stream())
            finally:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        thread = threading.Thread(target=run_async_stream)
        thread.daemon = True
        thread.start()

        def result_generator():
            try:
                while not stop_event.is_set():
                    try:
                        result = message_queue.get(timeout=0.1)
                        if result is None:
                            break
                        if isinstance(result, tuple) and result[0] == "ERROR":
                            raise RuntimeError(f"Error in async thread: {result[1]}")
                        yield result
                    except queue.Empty:
                        if not thread.is_alive():
                            raise RuntimeError("Background thread died unexpectedly")
            finally:
                stop_event.set()
                if thread.is_alive():
                    thread.join(timeout=5.0)

        return result_generator()

    async def acall(self, question: str, messages: List[dict] = None):
        """Asynchronous version of agent."""

        self.REACT_PROMPT = self._merge_tool_explaination_and_react()

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
        response = call_model(
            self.model_obj,
            text_input,
            self.verbose,
        )

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
                print(
                    "WARNING. Cannot extract JSON format action from response, retry.\n\n"
                )
                text_input = format_sys_prompt(
                    self.REACT_PROMPT,
                    "Question: " + question + retri_messages + self.REMEMBER_PROMPT,
                    self.prompt_format_type,
                    self.model,
                )
                response = call_model(
                    self.model_obj,
                    text_input,
                    self.verbose,
                )
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
                try:
                    tool_name = cur_action["action"]
                    tool_input = cur_action["action_input"]

                    tool = self.tools[tool_name]
                    result = tool_runner(tool, tool_input)
                    if asyncio.iscoroutine(result):
                        firsthand_observation = await result  # Await async result
                    else:
                        firsthand_observation = result  # Sync result

                except Exception:
                    logging.warning("Cannot run the tool, retry.")
                    print("WARNING. Cannot run the tool, retry.\n\n")
                    text_input = format_sys_prompt(
                        self.REACT_PROMPT,
                        "Question: " + question + retri_messages + self.REMEMBER_PROMPT,
                        self.prompt_format_type,
                        self.model,
                    )
                    response = call_model(
                        self.model_obj,
                        text_input,
                        self.verbose,
                    )
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
                    observation = call_model(
                        self.model_obj,
                        text_input,
                        self.verbose,
                    )
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
            response = call_model(
                self.model_obj,
                text_input,
                self.verbose,
            )
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
        response = call_model(
            self.model_obj,
            text_input,
            self.verbose,
        )

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
                print(
                    "WARNING. Cannot extract JSON format action from response, retry.\n\n"
                )
                text_input = format_sys_prompt(
                    self.REACT_PROMPT,
                    "Question: " + question + retri_messages + self.REMEMBER_PROMPT,
                    self.prompt_format_type,
                    self.model,
                )
                response = call_model(
                    self.model_obj,
                    text_input,
                    self.verbose,
                )
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
            intermed_stream_text = "\n[THOUGHT]: " + str(thought) + "\n"
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
                try:
                    tool_name = cur_action["action"]
                    tool_input = cur_action["action_input"]

                    # yield action #
                    intermed_stream_text = (
                        "\n[ACTION]: "
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
                except Exception:
                    logging.warning("Cannot run the tool, retry.")
                    print("Cannot run the tool, retry.")
                    text_input = format_sys_prompt(
                        self.REACT_PROMPT,
                        "Question: " + question + retri_messages + self.REMEMBER_PROMPT,
                        self.prompt_format_type,
                        self.model,
                    )
                    response = call_model(
                        self.model_obj,
                        text_input,
                        self.verbose,
                    )
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
                    observation = call_model(
                        self.model_obj,
                        text_input,
                        self.verbose,
                    )
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
                    print("\n[OBSERVATION]: " + observation)
                # yield observation #
                intermed_stream_text = "\n[OBSERVATION]: " + str(observation) + "\n"
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
            response = call_model(
                self.model_obj,
                text_input,
                self.verbose,
            )
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

    def mcp_agent(self, connection_info: dict, prompt: str):
        """Call the agent with the given connection info and prompt.

        Args:
            connection_info (dict): _description_
            prompt (str): _description_

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_

        Returns:
            _type_: _description_

        Yields:
            _type_: _description_
        """
        self.question = prompt
        if not self.stream:
            return asyncio.run(self._call_agents_non_streaming(connection_info, prompt))

        # For streaming, use a thread approach with a queue
        import queue
        import threading

        # Create a queue for passing messages from async to sync
        message_queue = queue.Queue()
        stop_event = threading.Event()

        # Function to run in a background thread
        def run_async_stream():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def process_stream():
                try:
                    client = MultiServerMCPClient(connection_info)
                    tools = await client.get_tools()

                    if isinstance(tools, BaseTool):
                        tools = [tools]

                    self.tool_explaination = get_tool_explaination(tools)
                    for tool in tools:
                        if not isinstance(tool, BaseTool):
                            logging.warning("tools should be a list of BaseTool")
                            print("WARNING. tools should be a list of BaseTool\n\n")
                            continue
                        tool_name = tool.name
                        self.tools[tool_name] = tool

                    # Use the agent asynchronously
                    response = await self.acall(prompt)

                    # Process the streaming response in real-time
                    async for chunk in response:
                        # Put each chunk in the queue as it arrives
                        message_queue.put(chunk)
                except Exception:
                    import traceback

                    message_queue.put(("ERROR", traceback.format_exc()))
                finally:
                    # Signal that we're done
                    message_queue.put(None)

            # Run the async function and ensure proper cleanup
            try:
                loop.run_until_complete(process_stream())
            finally:
                # Clean up the event loop
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Give cancelled tasks a chance to clean up
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )

                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        # Start the background thread
        thread = threading.Thread(target=run_async_stream)
        thread.daemon = True  # The thread will exit when the main thread exits
        thread.start()

        # Return a generator that yields results as they arrive from the queue
        def result_generator():
            try:
                while not stop_event.is_set():
                    try:
                        # Timeout allows checking the stop_event occasionally
                        result = message_queue.get(timeout=0.1)

                        # None signals end of stream
                        if result is None:
                            break

                        # Check for error
                        if isinstance(result, tuple) and result[0] == "ERROR":
                            raise RuntimeError(f"Error in async thread: {result[1]}")

                        yield result

                    except queue.Empty:
                        # Just a timeout, check if the thread is still alive
                        if not thread.is_alive():
                            # Thread died unexpectedly
                            raise RuntimeError("Background thread died unexpectedly")
                        # Otherwise continue waiting
            finally:
                # Clean up
                stop_event.set()
                # Wait for thread to finish if it's still running
                if thread.is_alive():
                    thread.join(timeout=5.0)

        return result_generator()

    ## use MultiServerMCPClient to connect to multiple MCP servers and get the tools
    async def _call_agents_non_streaming(self, connection_info: dict, prompt: str):
        """Handle the non-streaming case where we want to return a complete string"""

        client = MultiServerMCPClient(connection_info)
        tools = await client.get_tools()

        if isinstance(tools, BaseTool):
            tools = [tools]

        self.tool_explaination = get_tool_explaination(tools)
        for tool in tools:
            if not isinstance(tool, BaseTool):
                logging.warning("tools should be a list of BaseTool")
                print("WARNING. tools should be a list of BaseTool\n\n")
                continue
            tool_name = tool.name
            self.tools[tool_name] = tool

        # Use the agent asynchronously
        response = await self.acall(prompt)

        # For non-streaming, collect the complete response
        if hasattr(response, "__aiter__"):
            # If it's an async generator, collect all chunks and join them
            result = ""
            async for chunk in response:
                result += chunk
            return result
        # If it's already a string or other non-generator response
        return response
