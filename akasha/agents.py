from langchain.tools import BaseTool
from langchain.agents import load_tools, initialize_agent, tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from typing import Callable, Union, List
import akasha
import akasha.helper as helper
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.utils import print_text
from typing import TYPE_CHECKING, Any, Dict, Optional
from langchain_core.agents import AgentAction, AgentFinish
import traceback, warnings, datetime, time
from langchain.llms.base import LLM

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _get_agent_type(agent_type: str) -> AgentType:

    try:
        agent_t = getattr(AgentType, agent_type)
    except Exception as e:
        print(f"Cannot find the agent type, use default instead\n\n")
        agent_t = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    return agent_t


class MyCallbackHandler(BaseCallbackHandler):
    """Callback Handler that prints to std out."""

    def __init__(self, color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        self.color = color
        self.log = {}

    def on_chain_start(self, serialized: Dict[str, Any],
                       inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we are entering a chain."""
        self.log['action'] = []
        self.log['observation'] = []
        self.log['llm-prefix'] = []
        # class_name = serialized.get("name",
        #                             serialized.get("id", ["<unknown>"])[-1])
        # print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m"
        #       )  # noqa: T201
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""

        # print("\n\033[1m> Finished chain.\033[0m")  # noqa: T201
        pass

    def on_agent_action(self,
                        action: AgentAction,
                        color: Optional[str] = None,
                        **kwargs: Any) -> Any:
        """Run on agent action."""
        # print_text("on_agent_action:\n" + action.log,
        #            color=color or self.color)
        self.log['action'].append(action.log)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            self.log['observation'].append(observation_prefix)
            self.log['observation'].append(output)
            # print_text(
            #     f"on_tool_end___observation_prefix:\n\n{observation_prefix}")
        #print_text(output, color=color or self.color)
        if llm_prefix is not None:
            # print_text(f"on_tool_end___llm_prefix:\n\n{llm_prefix}")

            self.log['llm-prefix'].append(llm_prefix)

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when agent ends."""
        #print_text("on_text: \n" + text, color=color or self.color, end=end)
        pass

    def on_agent_finish(self,
                        finish: AgentFinish,
                        color: Optional[str] = None,
                        **kwargs: Any) -> None:
        """Run on agent end."""
        # print_text("on_agent_finish: \n" + finish.log,
        #            color=color or self.color,
        #            end="\n")
        pass

    def get_log(self):
        return self.log


class agent:
    """basic class for akasha agent, implement _change_variables, _check_db, add_log and save_logs function."""

    def __init__(
        self,
        tools: Union[BaseTool, List],
        agent_type="STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION",
        model: str = "openai:gpt-3.5-turbo",
        verbose: bool = False,
        language: str = "ch",
        temperature: float = 0.0,
    ):
        """initials of agent class

        Args:
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
            **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
            **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
                max token size of llm input.\n
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n
        """

        self.verbose = verbose
        self.language = language
        self.temperature = temperature
        self.timestamp_list = []
        self.logs = {}
        self.agent_type = _get_agent_type(agent_type)
        self.model_obj = helper.handle_model(model, self.verbose,
                                             self.temperature)
        self.model = helper.handle_search_type(model)
        self.agent_obj = None
        self.new_agent_flag = True  # check if need to create a new agent
        self.log_callback = MyCallbackHandler()
        if not isinstance(tools, List):
            self.tools = [tools]
        else:
            self.tools = tools

    def _set_model(self, **kwargs):
        """change model, embeddings, search_type, temperature if user use **kwargs to change them."""
        ## check if we need to change db, model_obj or embeddings_obj ##
        if "model" in kwargs or "temperature" in kwargs:
            new_temp = self.temperature
            new_model = self.model
            if "temperature" in kwargs:
                new_temp = kwargs["temperature"]
            if "model" in kwargs:
                new_model = kwargs["model"]
            if new_model != self.model or new_temp != self.temperature:
                self.model_obj = helper.handle_model(new_model, self.verbose,
                                                     new_temp)
                self.new_agent_flag = True

    def _change_variables(self, **kwargs):
        """change other arguments if user use **kwargs to change them."""

        ### check input argument is valid or not ###
        for key, value in kwargs.items():
            if (key == "model") and key in self.__dict__:
                self.__dict__[key] = helper.handle_search_type(value)

            elif key in self.__dict__:  # check if variable exist
                if (getattr(self, key, None)
                        != value):  # check if variable value is different

                    if key == "tools":
                        self.new_agent_flag = True
                        if not isinstance(value, List):
                            self.tools = [value]
                        else:
                            self.tools = value

                    elif key == "agent_type":
                        self.new_agent_flag = True
                        self.agent_type = _get_agent_type(value)
                    else:
                        self.__dict__[key] = value

            else:
                print(f"argument {key} not exist")

        return

    def _add_basic_log(self, timestamp: str, fn_type: str):
        """add pre-process log to self.logs

        Args:
            timestamp (str): timestamp of this run
            fn_type (str): function type of this run
        """
        if timestamp not in self.logs:
            self.logs[timestamp] = {}
        self.logs[timestamp]["fn_type"] = fn_type
        self.logs[timestamp]["model"] = self.model
        self.logs[timestamp]["language"] = self.language
        self.logs[timestamp]["temperature"] = self.temperature
        return

    def _add_result_log(self, timestamp: str, time: float):
        """add post-process log to self.logs

        Args:
            timestamp (str): timestamp of this run
            time (float): spent time of this run
        """
        tool_list = []
        for tool in self.tools:
            tool_list.append(tool.name)

        agent_log = self.log_callback.get_log()

        self.logs[timestamp]["time"] = time
        self.logs[timestamp]["tools"] = tool_list
        self.logs[timestamp]["question"] = self.question
        self.logs[timestamp]["response"] = self.response
        self.logs[timestamp].update(agent_log)
        return

    def save_logs(self, file_name: str = "", file_type: str = "json"):
        """save logs into json or txt file

        Args:
            file_name (str, optional): file path and the file name. if not assign, use logs/{current time}. Defaults to "".
            file_type (str, optional): the extension of the file, can be txt or json. Defaults to "json".

        Returns:
            plain_text(str): string of the log
        """

        extension = ""
        ## set extension ##
        if file_name != "":
            tst_file = file_name.split(".")[-1]
            if file_type == "json":
                if tst_file != "json" and tst_file != "JSON":
                    extension = ".json"
            else:
                if tst_file != "txt" and tst_file != "TXT":
                    extension = ".txt"

        ## set filename if not given ##
        from pathlib import Path

        if file_name == "":
            file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            logs_path = Path("logs")
            if not logs_path.exists():
                logs_path.mkdir()
            file_path = Path("logs/" + file_name + extension)
        else:
            file_path = Path(file_name + extension)

        ## write file ##
        if file_type == "json":
            import json

            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(self.logs, fp, indent=4, ensure_ascii=False)

        else:
            with open(file_path, "w", encoding="utf-8") as fp:
                for key in self.logs:
                    text = key + ":\n"
                    fp.write(text)
                    for k in self.logs[key]:
                        if type(self.logs[key][k]) == list:
                            text = (k + ": " + "\n".join(
                                [str(w) for w in self.logs[key][k]]) + "\n\n")

                        else:
                            text = k + ": " + str(self.logs[key][k]) + "\n\n"

                        fp.write(text)

                    fp.write("\n\n\n\n")

        print("save logs to " + str(file_path))
        return

    def __call__(self, question: str, **kwargs):

        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.question = question
        start_time = time.time()

        if self.new_agent_flag:
            self.agent_obj = initialize_agent(
                self.tools,
                self.model_obj,
                agent=self.agent_type,
                handle_parsing_errors=True,
                verbose=self.verbose,
                callbacks=[self.log_callback],
            )
            self.new_agent_flag = False

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        self.timestamp_list.append(timestamp)
        self._add_basic_log(timestamp, "agent")

        # ask question #
        try:
            self.response = self.agent_obj(self.question)
            self.response = self.response['output']
        except Exception as e:
            self.response = traceback.format_exc(
            ) + "\n\n" + "Error: agent get response failed.\n" + e.__str__(
            ) + "\n\n"
            print(self.response)

        end_time = time.time()
        self._add_result_log(timestamp, end_time - start_time)

        return self.response


def create_tool(tool_name: str, tool_description: str,
                func: Callable) -> Union[BaseTool, None]:

    try:

        class custom_tool(BaseTool):
            name: str
            description: str

            def _run(self, *args, **kwargs):
                return func(*args, **kwargs)

    except Exception as e:
        print(f"Cannot create tool correctly, {e}\n\n")

        return None

    return custom_tool(name=tool_name, description=tool_description)


def get_wiki_tool(model: Union[str, Callable, LLM] = "openai:gpt-3.5-turbo",
                  verbose: bool = True,
                  temperature: float = 0.0):

    if isinstance(model, Callable) or isinstance(model, str):
        model = helper.handle_model(model,
                                    verbose=verbose,
                                    temperature=temperature)
    tools = load_tools(
        [
            #"llm-math",
            "wikipedia",
        ],
        llm=model)

    return tools[0]


def get_saveJSON_tool():
    return create_tool(
        tool_name="json_tool",
        tool_description=
        "This is the tool to save the question and answer into json file, the input contains file_path and content.",
        func=_jsonSaveTool)


def _jsonSaveTool(file_path: str = "default.json", content: str = None):

    if content:
        try:
            # change content from string to json
            try:
                content = akasha.helper.extract_json(content)
            except Exception as e:
                print(content)
            import json
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(content, f, indent=4, ensure_ascii=False)
            return f"Success create {file_path}"
        except Exception as e:
            print("content: ", content)
            return f"{e}, Cannot save file_path {file_path}, save file as default.json"
    else:
        return "Error: content is empty"
