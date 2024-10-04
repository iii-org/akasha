from typing import Dict, List, Any, Optional, Callable, Generator, Union
from langchain.llms.base import LLM
from langchain.pydantic_v1 import BaseModel, Extra
from langchain.schema.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel, TextStreamer, AutoModelForCausalLM, TextIteratorStreamer
#from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
import torch
import numpy
import warnings, os, logging
import requests
import sys
from huggingface_hub import InferenceClient
from threading import Thread
from openai import OpenAI
import concurrent.futures
from PIL import Image


class chatGLM(LLM):
    max_token: int = 4096
    max_output_tokens: int = 1024
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    tokenizer: Any
    model: Any

    def __init__(self,
                 model_name: str,
                 temperature: float = 0.01,
                 max_output_tokens: int = 1024):
        """define chatglm model and the tokenizer

        Args:
            **model_name (str)**: chatglm model name\n
        """
        if model_name == "":
            model_name = "THUDM/chatglm2-6b"

        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name,
                                               trust_remote_code=True,
                                               device="cuda")
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.01

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return "ChatGLM"

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
    tokenizer: Any
    model: Any

    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.01,
        bit4: bool = True,
        max_token: int = 4096,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       use_fast=False,
                                                       max_length=max_token,
                                                       truncation=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token = max_token
        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.01
        if bit4 == False:
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
        return "gptq model"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        input_ids = self.tokenizer(
            prompt, return_tensors="pt",
            add_special_tokens=False).input_ids.to("cuda")
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


class custom_embed(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any  #: :meta private:
    model_name: str = "custom embedding model"
    """Model name to use."""
    # model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = {}
    """Keyword arguments to pass when calling the `encode` method of the model."""

    def __init__(self,
                 func: Any,
                 encode_kwargs: Dict[str, Any] = {},
                 **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        self.client = func
        self.encode_kwargs = encode_kwargs

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))

        embeddings = self.client(texts, **self.encode_kwargs)

        if isinstance(embeddings, numpy.ndarray):
            embeddings = embeddings.tolist()

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]


class custom_model(LLM):
    max_token: int = 4096
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    tokenizer: Any
    model: Any
    func: Any

    def __init__(self, func: Callable, temperature: float = 0.001):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__()
        self.func = func
        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.001

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return self.func.__name__

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """

        response = self.func(prompt)
        return response


class remote_model(LLM):
    max_token: int = 4096
    max_output_tokens: int = 1024
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    tokenizer: Any
    model: Any
    url: Any

    def __init__(self, base_url: str, temperature: float = 0.001, **kwargs):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__()
        self.url = base_url
        self.temperature = temperature
        if 'max_output_tokens' in kwargs:
            self.max_output_tokens = kwargs['max_output_tokens']

        if self.temperature == 0.0:
            self.temperature = 0.01

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return "remote api"

    def stream(self,
               prompt: str,
               stop: Optional[List[str]] = None) -> Generator:
        """run llm and get the stream generator

        Args:
            prompt (str): _description_

        Yields:
            Generator: _description_
        """
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        yield from self.invoke_stream(prompt, stop)
        return

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              verbose=True) -> str:
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

    def JSON_call(self,
                  prompt: str,
                  schema: dict = None,
                  stop: Optional[List[str]] = None) -> str:

        stop_list = get_stop_list(stop)
        try:
            client = InferenceClient(self.url)
            response = client.text_generation(
                prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_output_tokens,
                do_sample=True,
                top_k=10,
                top_p=0.95,
                grammar={
                    "type": "json",
                    "value": schema
                },
                repetition_penalty=1.2,
                stop_sequences=stop_list,
            )

        except Exception as e:
            logging.error("call remote model in JSON_call mode failed\n\n",
                          e.__str__())
            raise e
        return response  # response["generated_text"]

    def REGEX_call(self,
                   prompt: str,
                   regex: str = r"(yes|no)",
                   max_tokens: int = 1024,
                   stop: Optional[List[str]] = None) -> str:

        stop_list = get_stop_list(stop)
        try:
            client = InferenceClient(self.url)
            response = client.text_generation(
                prompt,
                temperature=self.temperature,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_k=10,
                top_p=0.95,
                grammar={
                    "type": "regex",
                    "value": regex
                },
                repetition_penalty=1.2,
                stop_sequences=stop_list,
            )

        except Exception as e:
            logging.error("call remote model in JSON_call mode failed\n\n",
                          e.__str__())
            raise e
        return response  # response["generated_text"]

    def _invoke_helper(self, args):
        messages, stop, verbose = args
        return self._call(messages, stop, verbose)

    def batch(self,
              prompt: List[str],
              stop: Optional[List[str]] = None) -> List[str]:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        # Number of threads should not exceed the number of prompts
        num_threads = min(
            len(prompt),
            concurrent.futures.thread.ThreadPoolExecutor()._max_workers)

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads) as executor:
            results = list(
                executor.map(self._invoke_helper,
                             [(message, stop, False) for message in prompt]))
        return results

    def invoke(self,
               messages: list,
               stop: Optional[List[str]] = None,
               verbose: bool = True) -> str:
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
        client = OpenAI(base_url=url, api_key="123")

        response = ""
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="remote_model",
                stream=True,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=0.95,
                stop=stop_list,
                frequency_penalty=1.2)

            for message in chat_completion:
                if verbose:
                    print(message.choices[0].delta.content, end="")
                response += message.choices[0].delta.content

        except Exception as e:
            logging.error("call remote model failed\n\n", e.__str__())
            raise e
        return response

    def invoke_stream(self,
                      messages: list,
                      stop: Optional[List[str]] = None) -> Generator:
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
        client = OpenAI(base_url=url, api_key="123")

        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="remote_model",
                stream=True,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=0.95,
                stop=stop_list,
                frequency_penalty=1.2)

            for message in chat_completion:
                yield message.choices[0].delta.content

        except Exception as e:
            info = "call remote model failed\n\n"
            logging.error(info, e.__str__())
            yield info + e.__str__()


class hf_model(LLM):

    max_token: int = 4096
    tokenizer: Any
    model: Any
    streamer: Any
    device: Any
    model_id: str
    processor: Any = None
    temperature: float = 0.01
    hf_token: Union[str, None] = None

    def __init__(self, model_name: str, temperature: float, **kwargs):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        self.hf_token = os.environ.get("HF_TOKEN")
        if self.hf_token is None:
            self.hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        super().__init__()
        self.model_id = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if temperature == 0.0:
            temperature = 0.01
        if 'max_output_tokens' in kwargs:
            self.max_output_tokens = kwargs['max_output_tokens']

        self.temperature = temperature
        if "vision" in model_name.lower():
            self.init_vision_model()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        return "huggingface text generation model hf"

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
            output[len(prompt[0]["content"][1]["text"]) + 8:])


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


def _url_requests(args):
    url, prompt, headers, parameters = args
    try:
        response = requests.post(url + "/generate",
                                 json={
                                     "inputs": prompt,
                                     "parameters": parameters
                                 },
                                 headers=headers).json()
        return response
    except Exception as e:
        logging.error("Call to remote model failed: ", e)
        return None
