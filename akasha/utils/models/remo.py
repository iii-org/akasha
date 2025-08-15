import logging
from typing import List, Any, Optional, Generator
from langchain.llms.base import LLM
from openai import OpenAI
import concurrent.futures

from pydantic import Field


class remote_model(LLM):
    max_token: int = 4096
    max_output_tokens: int = 1024
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    model_name: str = "remote_model"
    api_key: str = "123"
    tokenizer: Any = Field(default=None)
    model: OpenAI = Field(default=None)
    url: Any = Field(default=None)

    def __init__(
        self,
        base_url: str,
        temperature: float = 0.001,
        api_key: str = "123",
        model_name: str = "remote_model",
        **kwargs,
    ):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__()
        self.url = handle_url(base_url)
        self.temperature = temperature
        self.api_key = api_key
        self.model_name = model_name
        if "max_output_tokens" in kwargs:
            self.max_output_tokens = kwargs["max_output_tokens"]

        if self.temperature == 0.0:
            self.temperature = 0.01
        self.model = OpenAI(base_url=self.url, api_key=self.api_key)

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return "remote: api model"

    def stream(
        self, prompt: str, stop: Optional[List[str]] = None, verbose: bool = True
    ) -> Generator:
        """run llm and get the stream generator

        Args:
            prompt (str): _description_

        Yields:
            Generator: _description_
        """
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        yield from self.invoke_stream(prompt, stop, verbose)
        return

    def _call(self, prompt: str, stop: Optional[List[str]] = None, verbose=True) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """

        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        return self.invoke(prompt, stop, verbose)

    def _invoke_helper(self, args):
        messages, stop, verbose = args
        return self._call(messages, stop, verbose)

    def batch(
        self, prompt: List[str], stop: Optional[List[str]] = None, verbose: bool = False
    ) -> List[str]:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        # Number of threads should not exceed the number of prompts
        num_threads = min(
            len(prompt), concurrent.futures.thread.ThreadPoolExecutor()._max_workers
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(
                executor.map(
                    self._invoke_helper,
                    [(message, stop, verbose) for message in prompt],
                )
            )
        return results

    def invoke(
        self, messages: list, stop: Optional[List[str]] = None, verbose: bool = True
    ) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        stop_list = get_stop_list(stop)

        response = ""
        try:
            chat_completion = self.model.chat.completions.create(
                messages=messages,
                model=self.model_name,
                stream=True,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=0.95,
                stop=stop_list,
                frequency_penalty=1.2,
            )

            for message in chat_completion:
                content = message.choices[0].delta.content
                if isinstance(content, str):
                    if verbose:
                        print(message.choices[0].delta.content, end="")
                    response += message.choices[0].delta.content

        except Exception as e:
            logging.error("call remote model failed\n\n", e.__str__())
            raise e
        return response

    def invoke_stream(
        self, messages: list, stop: Optional[List[str]] = None, verbose: bool = True
    ) -> Generator:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        stop_list = get_stop_list(stop)
        url = self.url
        if url[-1] != "/":
            url += "/"

        if url[-3:] != "v1/":
            url = url + "v1/"

        try:
            chat_completion = self.model.chat.completions.create(
                messages=messages,
                model=self.model_name,
                stream=True,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=0.95,
                stop=stop_list,
                frequency_penalty=1.2,
            )

            for message in chat_completion:
                content = message.choices[0].delta.content
                if isinstance(content, str):
                    if verbose:
                        print(message.choices[0].delta.content, end="")
                    yield message.choices[0].delta.content

        except Exception as e:
            info = "call remote model failed\n\n"
            logging.error(info, e.__str__())
            yield info + e.__str__()

    def get_num_tokens(self, text: str) -> int:
        """get number of tokens in the text

        Args:
            **text (str)**: input text

        Returns:
            int: number of tokens
        """
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text))
        return num_tokens


def get_stop_list(stop: Optional[List[str]]) -> List[str]:
    """get stop list

    Args:
        stop (Optional[List[str]]): stop list

    Returns:
        List[str]: stop list
    """
    ret = ["<|eot_id|>", "<|end_header_id|>", "</s>"]
    if stop is not None:
        ret = stop
    return ret


def handle_url(url: str):
    if url[-1] != "/":
        url += "/"

    if url[-3:] != "v1/":
        url = url + "v1/"
    return url
