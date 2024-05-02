from typing import Dict, List, Any, Optional, Callable, Generator
from langchain.llms.base import LLM
from langchain.pydantic_v1 import BaseModel, Extra
from langchain.schema.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel, TextStreamer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
import torch
import numpy
import warnings, os, logging
import requests
import sys
from huggingface_hub import InferenceClient


class chatGLM(LLM):
    max_token: int = 4096
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    tokenizer: Any
    model: Any

    def __init__(self, model_name: str, temperature: float = 0.01):
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
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    tokenizer: Any
    model: Any
    url: Any

    def __init__(self, base_url: str, temperature: float = 0.001):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__()
        self.url = base_url
        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.01

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return "remote api"

    def stream(self, prompt: str) -> Generator:
        """run llm and get the stream generator

        Args:
            prompt (str): _description_

        Yields:
            Generator: _description_
        """
        try:
            client = InferenceClient(self.url)

            yield from client.text_generation(
                prompt,
                max_new_tokens=1024,
                do_sample=True,
                top_k=10,
                top_p=0.95,
                stream=True,
                repetition_penalty=1.2,
                stop_sequences=["<|eot_id|>", "<|end_header_id|>"],
            )

        except Exception as e:
            info = "call remote model failed\n\n"
            logging.error(info, e.__str__())
            yield info + e.__str__()

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        try:
            client = InferenceClient(self.url)
            response = ""
            for token in client.text_generation(
                    prompt,
                    max_new_tokens=1024,
                    do_sample=True,
                    top_k=10,
                    top_p=0.95,
                    stream=True,
                    repetition_penalty=1.2,
                    stop_sequences=["<|eot_id|>", "<|end_header_id|>"],
            ):
                print(token, end='', flush=True)
                response += token
            # data = {
            #     "inputs": prompt,
            #     "parameters": {
            #         'temperature': self.temperature,
            #         'max_new_tokens': 1024,
            #         'do_sample': True,
            #         'top_k': 10,
            #         'top_p': 0.95,
            #     }
            # }
            # headers = {"Content-Type": "application/json"}

            # try:
            #     response = requests.post(self.url + "/generate",
            #                              json=data,
            #                              headers=headers).json()
        except Exception as e:
            logging.error("call remote model failed\n\n", e.__str__())
            raise e
        return response  # response["generated_text"]


class hf_model(LLM):

    max_token: int = 4096
    tokenizer: Any
    model: Any
    pipe_line: Any

    def __init__(self, pipe: pipeline):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__()
        self.pipe_line = HuggingFacePipeline(pipeline=pipe)

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return "huggingface pipeline"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        response = self.pipe_line._generate([prompt]).generations[0][0]
        return response.text

    def _generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.pipe_line._generate([prompt]).generations[0][0]
        return response.text


def get_hf_model(model_name, temperature: float = 0.0):
    """try different methods to define huggingface model, first use pipline and then use llama2.

    Args:
        model_name (str): huggingface model name\n

    Returns:
        _type_: llm model
    """
    """try different methods to define huggingface model, first use pipline and then use llama2.

    Args:
        model_name (str): huggingface model name\n

    Returns:
        _type_: llm model
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        streamer = TextStreamer(tokenizer)
        try:
            pipe = pipeline(
                "text-generation",
                model=model_name,
                token=hf_token,
                max_new_tokens=1024,
                tokenizer=tokenizer,
                streamer=streamer,
                model_kwargs={
                    "temperature": temperature,
                    "repetition_penalty": 1.2,
                },
                device_map="auto",
                batch_size=8,
                torch_dtype=torch.float16,
            )
            model = hf_model(pipe=pipe)

        except Exception as e:
            pipe = pipeline(
                "question-answering",
                model=model_name,
                token=hf_token,
                max_new_tokens=1024,
                tokenizer=tokenizer,
                streamer=streamer,
                model_kwargs={
                    "temperature": temperature,
                    "repetition_penalty": 1.2,
                },
                device_map="auto",
                batch_size=8,
                torch_dtype=torch.float16,
            )
            model = hf_model(pipe=pipe)

    return model
