from akasha.utils.atman import basic_llm
from akasha.utils.base import DEFAULT_MODEL, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_INPUT_TOKENS
from akasha.utils.prompts.gen_prompt import default_ask_prompt
from akasha.utils.db.load_docs import load_docs_from_info
from typing import Callable, Union, List, Tuple, Generator
from pathlib import Path
from langchain.schema import Document
import time, datetime


class ask(basic_llm):

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        language: str = "ch",
        record_exp: str = "",
        system_prompt: str = "",
        keep_logs: bool = False,
        verbose: bool = False,
        stream: bool = False,
        env_file: str = "",
    ):
        """_summary_

        Args:
            model (str, optional): _description_. Defaults to DEFAULT_MODEL.
            max_input_tokens (int, optional): _description_. Defaults to DEFAULT_MAX_INPUT_TOKENS.
            max_output_tokens (int, optional): _description_. Defaults to DEFAULT_MAX_OUTPUT_TOKENS.
            temperature (float, optional): _description_. Defaults to 0.0.
            language (str, optional): _description_. Defaults to "ch".
            record_exp (str, optional): _description_. Defaults to "".
            system_prompt (str, optional): _description_. Defaults to "".
            keep_logs (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to False.
            env_file (str, optional): _description_. Defaults to "".
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

        self.prompt = ""
        self.response = ""
        self.docs = []

        ## set default RAG prompt ##
        if self.system_prompt.replace(' ', '') == "":
            self.system_prompt = default_ask_prompt(self.language)

    def __call__(self,
                 prompt: str,
                 info: Union[str, list, Path, Document] = "",
                 history_messages: List[str] = [],
                 **kwargs) -> str:
        """_summary_

        Args:
            prompt (str): _description_
            info (Union[str, list], optional): _description_. Defaults to "".
            history_messages (list, optional): _description_. Defaults to [].

        Returns:
            str: _description_
        """

        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.prompt = prompt

        start_time = time.time()
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")

        self.docs = load_docs_from_info(info)
