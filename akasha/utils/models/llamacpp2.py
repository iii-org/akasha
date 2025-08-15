from typing import List, Optional
from langchain.llms.base import LLM

# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Generator, Union  # noqa: F811
from pydantic import Field
import atexit


class LlamaCPP(LLM):
    max_token: int = 4096
    model: LLM = Field(default=None)
    device: str = Field(default=None)
    model_id: str
    temperature: float = 0.01
    max_output_tokens: int = 1024
    verbose: bool = False

    def __init__(self, model_name: str, temperature: float, **kwargs):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__(model_id=model_name)
        from llama_cpp import Llama

        if temperature == 0.0:
            temperature = 0.01
        if "max_output_tokens" in kwargs:
            self.max_output_tokens = kwargs["max_output_tokens"]
        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]
        self.temperature = temperature

        try:
            self.model = Llama(
                self.model_id, n_ctx=8192, n_threads=16, n_batch=512, verbose=False
            )
        except Exception:
            try:
                repo_id = "/".join(self.model_id.split("/")[:-1])
                file_name = self.model_id.split("/")[-1]
                self.model = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=file_name,
                    n_ctx=8192,
                    n_threads=16,
                    n_batch=512,
                    verbose=self.verbose,
                )
            except Exception:
                print(f"model {model_name} not found.")
                raise Exception(f"model {model_name} not found.")

        # Register the cleanup function
        atexit.register(self.cleanup)

    def cleanup(self):
        """Cleanup function to be called on exit."""
        if hasattr(self, "model") and self.model is not None:
            try:
                del self.model
            except Exception as e:
                print(f"Error during cleanup: {e}")

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return "llama-cpp:" + self.model_id

    def stream(
        self,
        prompt: Union[list, str],
        stop: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Generator[str, None, None]:
        stop_list = get_stop_list(stop)
        input_text = ""
        if isinstance(prompt, list):
            for pp in prompt:
                input_text += pp["content"]
        else:
            input_text = prompt
        output = self.model(
            input_text,
            stream=True,
            stop=stop_list,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            presence_penalty=1,
            frequency_penalty=1,
        )

        for text in output:
            delta = text["choices"][0]["text"]
            if verbose:
                print(delta, end="", flush=True)
            yield delta

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None, verbose: bool = True
    ) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        stop_list = get_stop_list(stop)
        input_text = ""
        if isinstance(prompt, list):
            for pp in prompt:
                input_text += pp["content"]
        else:
            input_text = prompt

        output = self.model(
            input_text,
            stream=True,
            stop=stop_list,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            presence_penalty=1,
            frequency_penalty=1,
        )

        ret = ""
        for text in output:
            delta = text["choices"][0]["text"]
            if verbose:
                print(delta, end="", flush=True)
            ret += delta

        return ret


def get_stop_list(stop: Optional[List[str]]) -> List[str]:
    """get stop list

    Args:
        stop (Optional[List[str]]): stop list

    Returns:
        List[str]: stop list
    """
    ret = ["<|eot_id|>", "<|end_header_id|>", "</s>", "<|end_of_text|>"]
    if stop is not None:
        ret = stop
    return ret
