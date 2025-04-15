import google.generativeai as genai
from google.generativeai import GenerationConfig
from langchain.llms.base import LLM
from typing import Dict, List, Any, Optional, Generator, Union
from pydantic import Field
import concurrent.futures


class gemini_model(LLM):
    max_token: int = 4096
    max_output_tokens: int = 1024
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    model: genai.GenerativeModel = Field(default=None)
    model_name: str = "gemini-1.5-flash"

    def __init__(
        self, model_name: str, api_key: str, temperature: float = 0.0, **kwargs
    ):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__()
        genai.configure(api_key=api_key)
        self.temperature = temperature
        if "max_output_tokens" in kwargs:
            self.max_output_tokens = kwargs["max_output_tokens"]

        generation_config = GenerationConfig(
            max_output_tokens=self.max_output_tokens,
            temperature=temperature,
        )
        self.model = genai.GenerativeModel(
            model_name, generation_config=generation_config
        )
        self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return f"gemini:{self.model_name}"

    def stream(
        self, prompt: Union[str, List[Dict[str, Any]]], stop: Optional[List[str]] = None
    ) -> Generator:
        """run llm and get the stream generator

        Args:
            prompt (str): _description_

        Yields:
            Generator: _description_
        """
        if isinstance(prompt, list):
            prompt = check_format_prompt(prompt)

        generation_config = GenerationConfig(
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_sequences=stop,
        )
        streaming_response = self.model.generate_content(
            prompt, generation_config=generation_config, stream=True
        )

        for s in streaming_response:
            yield s.text
        return

    def _call(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        stop: Optional[List[str]] = None,
        verbose=True,
    ) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """

        if isinstance(prompt, list):
            prompt = check_format_prompt(prompt)

        generation_config = GenerationConfig(
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_sequences=stop,
        )

        ret = ""
        streaming_response = self.model.generate_content(
            prompt, generation_config=generation_config, stream=True
        )

        for s in streaming_response:
            print(s.text, end="", flush=True)
            ret += s.text

        return ret

    def _invoke_helper(self, args):
        messages, stop, verbose = args
        return self._call(messages, stop, verbose)

    def batch(self, prompt: List[str], stop: Optional[List[str]] = None) -> List[str]:
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
                    self._invoke_helper, [(message, stop, False) for message in prompt]
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
        return self._call(messages, stop, verbose)

    def invoke_stream(
        self, messages: list, stop: Optional[List[str]] = None
    ) -> Generator:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        return self.stream(messages, stop)


def check_format_prompt(prompts: list):
    """check and format the prompt to fit the correct gemini format"""
    for idx, prompt in enumerate(prompts):
        if prompt["role"] != "user":
            prompts[idx]["role"] = "model"
        if ("parts" not in prompt) and ("content" in prompt):
            prompts[idx]["parts"] = [prompts[idx]["content"]]
            prompts[idx].pop("content")

    return prompts


def calculate_token(
    prompt: str,
    model_name: str = "gemini-1.5-flash",
):
    num_tokens = genai.GenerativeModel(model_name).count_tokens(prompt)

    return num_tokens
