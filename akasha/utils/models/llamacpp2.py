from typing import Dict, List, Any, Optional
from langchain.llms.base import LLM
import torch, sys

# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Dict, List, Any, Optional, Callable, Generator, Union
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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if temperature == 0.0:
            temperature = 0.01
        if 'max_output_tokens' in kwargs:
            self.max_output_tokens = kwargs['max_output_tokens']
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        self.temperature = temperature
        if self.device == 'cuda':
            # chat_format="llama-2",
            self.model = Llama(model_path=model_name,
                               n_ctx=self.max_token,
                               n_gpu_layers=-1,
                               n_threads=16,
                               n_batch=512,
                               verbose=self.verbose)

        else:
            self.model = Llama(model_path=model_name,
                               n_ctx=self.max_token,
                               n_threads=16,
                               n_batch=512,
                               verbose=self.verbose)
        # Register the cleanup function
        atexit.register(self.cleanup)

    def cleanup(self):
        """Cleanup function to be called on exit."""
        if hasattr(self, 'model') and self.model is not None:
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

    def stream(self,
               prompt: Union[list, str],
               stop: Optional[List[str]] = None) -> Generator[str, None, None]:

        stop_list = get_stop_list(stop)
        if isinstance(prompt, str):
            prompt = [{
                "role": "system",
                "content": "you are a helpful assistant"
            }, {
                "role": "user",
                "content": prompt
            }]

        output = self.model.create_chat_completion(
            prompt,
            stream=True,
            stop=stop_list,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            presence_penalty=1,
            frequency_penalty=1)

        for text in output:
            delta = text['choices'][0]['delta']
            if 'role' in delta:
                yield delta['role'] + ': '
            elif 'content' in delta:
                yield delta['content']

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        stop_list = get_stop_list(stop)
        if isinstance(prompt, str):
            prompt = [{
                "role": "system",
                "content": "you are a helpful assistant"
            }, {
                "role": "user",
                "content": prompt
            }]

        output = self.model.create_chat_completion(
            prompt,
            stream=True,
            stop=stop_list,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            presence_penalty=1,
            frequency_penalty=1)

        ret = ""
        for text in output:
            delta = text['choices'][0]['delta']
            if 'role' in delta:
                sp = delta['role']
                print(sp, end=': ', flush=True)
                ret += (sp + ": ")

            elif 'content' in delta:
                sp = delta['content']
                print(sp, end='', flush=True)
                ret += sp

        return ret


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
