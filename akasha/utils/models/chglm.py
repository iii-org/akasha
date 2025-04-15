from typing import List, Any, Optional
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel

from pydantic import Field


class chatGLM(LLM):
    max_token: int = 4096
    max_output_tokens: int = 1024
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    tokenizer: Any = Field(default=None)
    model: Any = Field(default=None)
    model_name: str = ""

    def __init__(
        self, model_name: str, temperature: float = 0.01, max_output_tokens: int = 1024
    ):
        """define chatglm model and the tokenizer

        Args:
            **model_name (str)**: chatglm model name\n
        """
        if model_name == "":
            model_name = "THUDM/chatglm2-6b"

        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, device="cuda"
        )
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.01
        self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return f"chatglm:{self.model_name}  ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        self.model = self.model.eval()
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response
