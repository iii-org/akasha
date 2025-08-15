from typing import List, Any, Optional
from langchain.llms.base import LLM
from transformers import AutoTokenizer, TextStreamer

import torch
import sys
from pydantic import Field


class gptq(LLM):
    """define initials and _call function for gptq model

    Args:
        LLM (_type_): _description_

    Returns:
        _type_: _description_
    """

    max_token: int = 4096
    temperature: float = 0.01
    top_p: float = 0.95
    tokenizer: Any = Field(default=None)
    model: Any = Field(default=None)

    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.01,
        bit4: bool = True,
        max_token: int = 4096,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False, max_length=max_token, truncation=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token = max_token
        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.01
        if bit4 is False:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True,
            )
            self.model.eval()
        else:
            from auto_gptq import AutoGPTQForCausalLM

            self.model = AutoGPTQForCausalLM.from_quantized(
                model_name_or_path,
                low_cpu_mem_usage=True,
                device="cuda:0",
                use_triton=False,
                inject_fused_attention=False,
                inject_fused_mlp=False,
            )

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

    @property
    def _llm_type(self) -> str:
        return "gptq: model"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to("cuda")
        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": 1024,
            "do_sample": True,
            "top_k": 50,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": 1.2,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        generate_ids = self.model.generate(**generate_input)
        text = self.tokenizer.decode(generate_ids[0])
        return text

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


class peft_Llama2(LLM):
    """define initials and _call function for llama2 peft model

    Args:
        LLM (_type_): _description_

    Returns:
        _type_: _description_
    """

    max_token: int = 2048
    temperature: float = 0.01
    top_p: float = 0.95
    tokenizer: Any = Field(default=None)
    model: Any = Field(default=None)

    def __init__(
        self, model_name_or_path: str, max_token: int = 2048, temperature: float = 0.01
    ):
        super().__init__()
        from peft import AutoPeftModelForCausalLM

        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.01
        self.max_token = max_token
        device_map = {"": 0}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path + "/adapter_model",
            temperature=0.1,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

    @property
    def _llm_type(self) -> str:
        return "peft_Llama2:"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=self.temperature,
            do_sample=True,
        )

        result_message = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return result_message

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


class TaiwanLLaMaGPTQ(LLM):
    max_token: int = 300
    temperature: float = 0.01
    top_p: float = 0.95
    tokenizer: Any = Field(default=None)
    model: Any = Field(default=None)
    streamer: Any = Field(default=None)

    def __init__(self, model_name_or_path: str, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.01
        from auto_gptq import AutoGPTQForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            max_length=4096,
            truncation=True,
            add_eos_token=True,
        )
        self.model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            trust_remote_code=True,
            use_safetensors=True,
            device_map="auto",
            use_triton=False,
            strict=False,
        )

        self.streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    @property
    def _llm_type(self) -> str:
        return "Taiwan_LLaMa:"

    def _call(
        self, message: str, stop: Optional[List[str]] = None, verbose: bool = True
    ):
        prompt = message
        tokens = self.tokenizer(prompt, return_tensors="pt").input_ids
        generate_ids = self.model.generate(
            input_ids=tokens.cuda(),
            max_new_tokens=self.max_token,
            streamer=self.streamer,
            top_p=0.95,
            top_k=50,
            temperature=self.temperature,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output = self.tokenizer.decode(generate_ids[0, len(tokens[0]) : -1]).strip()
        if verbose:
            print(output, end="", flush=True)

        return output

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
