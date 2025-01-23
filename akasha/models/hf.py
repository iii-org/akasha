from typing import Dict, List, Any, Optional, Generator, Union
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

import torch
import requests
from threading import Thread

from PIL import Image
from pydantic import Field


class hf_model(LLM):

    max_token: int = 4096
    tokenizer: Any = Field(default=None)
    model: Any = Field(default=None)
    streamer: Any = Field(default=None)
    device: Any = Field(default=None)
    model_id: str
    processor: Any = None
    temperature: float = 0.01
    hf_token: Union[str, None] = None
    max_output_tokens: int = 1024
    model_name: str = ""

    def __init__(self, model_name: str, env_dict: dict, temperature: float,
                 **kwargs):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__(model_id=model_name)
        if "HF_TOKEN" in env_dict:
            self.hf_token = env_dict["HF_TOKEN"]
        elif "HUGGINGFACEHUB_API_TOKEN" in env_dict:
            self.hf_token = env_dict["HUGGINGFACEHUB_API_TOKEN"]
        else:
            self.hf_token = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if temperature == 0.0:
            temperature = 0.01
        if 'max_output_tokens' in kwargs:
            self.max_output_tokens = kwargs['max_output_tokens']

        self.temperature = temperature
        self.model_name = model_name
        if "vision" in model_name.lower():
            self.init_vision_model()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                           token=self.hf_token)
            self.streamer = TextIteratorStreamer(self.tokenizer,
                                                 skip_prompt=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=self.hf_token,
                temperature=temperature,
                repetition_penalty=1.2,
                top_p=0.95,
                torch_dtype=torch.float16,
                device_map="auto").to(self.device)

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return f"hf:{self.model_name} huggingface"

    def init_vision_model(self, **kwargs):
        """init vision model

        Args:
            model_name (str): model name
        """
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor

            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.streamer = TextIteratorStreamer(self.tokenizer,
                                                 skip_prompt=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.hf_token,
                temperature=self.temperature,
                repetition_penalty=1.2,
                top_p=0.95,
                torch_dtype=torch.float16,
                device_map="auto").to(self.device)

    def stream(self,
               prompt: str,
               stop: Optional[List[str]] = None) -> Generator[str, None, None]:

        stop_list = get_stop_list(stop)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        gerneration_kwargs = dict(
            inputs,
            streamer=self.streamer,
            max_new_tokens=self.max_output_tokens,
            do_sample=True,
            min_new_tokens=10,
            stop_strings=stop_list,
            tokenizer=self.tokenizer,
        )
        #self.model.generate(**inputs, streamer= self.streamer, max_new_tokens=1024, do_sample=True)

        thread = Thread(target=self.model.generate, kwargs=gerneration_kwargs)
        thread.start()
        # for text in self.streamer:
        #     yield text
        yield from self.streamer

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        stop_list = get_stop_list(stop)
        gerneration_kwargs = dict(
            inputs,
            streamer=self.streamer,
            max_new_tokens=self.max_output_tokens,
            do_sample=True,
            stop_strings=stop_list,
            tokenizer=self.tokenizer,
        )

        thread = Thread(target=self.model.generate, kwargs=gerneration_kwargs)
        thread.start()
        generated_text = ""
        for new_text in self.streamer:
            print(new_text, end='', flush=True)
            generated_text += new_text
        return generated_text

    def _generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        return self._call(prompt, stop)

    def batch(self,
              prompt: List[str],
              stop: Optional[List[str]] = None) -> List[str]:

        stop_list = get_stop_list(stop)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(prompt,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=1024).to(self.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_tokens,
            do_sample=True,
            stop_strings=stop_list)
        generated_texts = self.tokenizer.batch_decode(generated_ids,
                                                      skip_special_tokens=True)

        return generated_texts

    def call_image(self, prompt: list) -> str:

        def is_url(path):
            from urllib.parse import urlparse
            parsed_url = urlparse(path)
            return parsed_url.scheme in ('http', 'https', 'ftp')

        ### check if the image is from url or local path
        image_path = prompt[0]["content"][0]["type"]
        prompt[0]["content"][0]["type"] = "image"

        if is_url(image_path):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)

        ## apply chat template ##
        input_text = self.processor.apply_chat_template(
            prompt, add_generation_prompt=True)
        inputs = self.processor(image,
                                input_text,
                                add_special_tokens=False,
                                return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs,
                                     max_new_tokens=512,
                                     do_sample=True,
                                     temperature=self.temperature,
                                     top_p=0.95,
                                     length_penalty=1.0,
                                     repetition_penalty=1.0)
        return self.processor.decode(
            output[0][len(prompt[0]["content"][1]["text"]) + 8:])


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
