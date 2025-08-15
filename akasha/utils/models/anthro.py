from anthropic import Anthropic
from langchain.llms.base import LLM
from typing import Dict, List, Any, Optional, Generator, Union
from pydantic import Field
import concurrent.futures


class anthropic_model(LLM):
    max_token: int = 4096
    max_output_tokens: int = 1024
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    model: Anthropic = Field(default=None)
    model_name: str = "claude-3-5-sonnet-20241022"

    def __init__(
        self, model_name: str, api_key: str, temperature: float = 0.0, **kwargs
    ):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__()
        self.model = Anthropic(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        if "max_output_tokens" in kwargs:
            self.max_output_tokens = kwargs["max_output_tokens"]

        #### gcp vertex (need credentials) ####

        # from anthropic import AnthropicVertex

        # client = AnthropicVertex(region=os.environ['GCP_LOCATION'],
        #                         project_id=os.environ['GCP_PROJECT_ID'])

        # with client.messages.stream(
        #         max_tokens=1024,
        #         messages=[{
        #             "role": "user",
        #             "content": "Send me a recipe for banana bread.",
        #         }],
        #         model="claude-3-sonnet@20240229",
        # ) as stream:
        #     for text in stream.text_stream:
        #         print(text)

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return f"anthropic:{self.model_name}"

    def stream(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        stop: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Generator:
        """run llm and get the stream generator

        Args:
            prompt (str): _description_

        Yields:
            Generator: _description_
        """
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        with self.model.messages.stream(
            max_tokens=self.max_output_tokens,
            messages=prompt,
            model=self.model_name,
            stop_sequences=stop,
            temperature=self.temperature,
            top_p=self.top_p,
        ) as stream:
            for text in stream.text_stream:
                if verbose:
                    print(text, end="", flush=True)
                yield text

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

        ret = ""
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        with self.model.messages.stream(
            max_tokens=self.max_output_tokens,
            messages=prompt,
            model=self.model_name,
            stop_sequences=stop,
            temperature=self.temperature,
            top_p=self.top_p,
        ) as stream:
            for text in stream.text_stream:
                ret += text
                if verbose:
                    print(text, end="", flush=True)

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

    def call_image(
        self, prompt: list, stop: Optional[List[str]] = None, verbose: bool = False
    ) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """

        messages = self.model.messages.create(
            max_tokens=self.max_output_tokens,
            messages=prompt,
            model=self.model_name,
            stop_sequences=stop,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if verbose:
            print(messages.content[0].text, end="", flush=True)
        return messages.content[0].text

    def count_tokens(self, prompt: Union[list, str]) -> int:
        """caculate the token count

        Args:
            prompt (Union[list,str]): _description_

        Returns:
            int: _description_
        """
        if isinstance(prompt, str):
            input_text = [{"role": "user", "content": prompt}]

        token_count = self.model.beta.messages.count_tokens(
            model=self.model_name, messages=input_text
        )

        return token_count.input_tokens

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in the text using the model's tokenizer.

        Args:
            text (str): The text to be tokenized.

        Returns:
            int: The number of tokens in the text.
        """

        return self.count_tokens(text)
