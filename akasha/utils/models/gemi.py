from google import genai
from google.genai import types

from langchain.llms.base import LLM
from typing import Dict, List, Any, Optional, Generator, Union
from pydantic import BaseModel
import os
from langchain.schema.embeddings import Embeddings
from pathlib import Path

# from pydantic import Field
import concurrent.futures


class gemini_model(LLM):
    max_token: int = 4096
    max_output_tokens: int = 1024
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    model_name: str = "gemini-1.5-flash"
    client: genai.Client = None
    generation_config: types.GenerateContentConfig = None

    def __init__(
        self, model_name: str, api_key: str, temperature: float = 0.0, **kwargs
    ):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        super().__init__()
        self.client = genai.Client(api_key=api_key)
        self.temperature = temperature
        if "max_output_tokens" in kwargs:
            self.max_output_tokens = kwargs["max_output_tokens"]

        self.generation_config = types.GenerateContentConfig(
            max_output_tokens=self.max_output_tokens,
            temperature=temperature,
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
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        verbose: bool = True,
        stop: Optional[List[str]] = None,
        system_prompt: Union[str, None] = None,
    ) -> Generator:
        """run llm and get the stream generator

        Args:
            prompt (str): _description_

        Yields:
            Generator: _description_
        """
        if isinstance(prompt, list):
            prompt, system_prompt = check_format_prompt(prompt)

        generation_config = types.GenerateContentConfig(
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_sequences=stop,
            system_instruction=system_prompt,
        )
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name, contents=prompt, config=generation_config
        ):
            if chunk.text:
                if verbose:
                    print(chunk.text, end="", flush=True)
                yield chunk.text

        return

    def _call(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        stop: Optional[List[str]] = None,
        verbose=True,
        system_prompt: Union[str, None] = None,
        response_format: Union[dict, BaseModel, None] = None,
    ) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        if isinstance(prompt, list):
            prompt, system_prompt = check_format_prompt(prompt)
        config_param = {
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop_sequences": stop,
            "system_instruction": system_prompt,
        }
        if response_format is not None:
            config_param["response_schema"] = list[response_format]
            config_param["response_mime_type"] = "application/json"
        generation_config = types.GenerateContentConfig(**config_param)

        ret = ""

        for chunk in self.client.models.generate_content_stream(
            model=self.model_name, contents=prompt, config=generation_config
        ):
            if chunk.text:
                if verbose:
                    print(chunk.text, end="", flush=True)
                ret += chunk.text

        return ret

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
        self,
        messages: list,
        stop: Optional[List[str]] = None,
        verbose: bool = True,
        response_format: Union[dict, BaseModel, None] = None,
    ) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        return self._call(messages, stop, verbose, response_format=response_format)

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
        return self.stream(messages, stop, verbose=verbose)

    def generate(
        self,
        prompt: str,
        save_path: str = "./image.png",
        verbose: bool = True,
        **kwargs,
    ) -> Path:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )

        from PIL import Image
        from io import BytesIO

        path = Path(save_path)
        if path.is_dir():
            # If it's a directory, append default filename
            save_path = path / "image.png"

        for part in response.candidates[0].content.parts:
            if part.text is not None and verbose:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO((part.inline_data.data)))
                image.save(save_path.__str__())
                # image.show()
                if verbose:
                    print(f"Image saved to {save_path.__str__()}")

        return save_path

    def edit(
        self,
        prompt: str,
        images: list[str],
        save_path: str = "./image.png",
        verbose: bool = True,
        **kwargs,
    ):
        ## process the image list
        images_source = [prompt]
        for image in images:
            if isinstance(image, str):
                # use Path to check if the image is exist
                image_path = Path(image)
                if not image_path.exists():
                    raise ValueError(f"Image {image} does not exist.")
                images_source.append(self.client.files.upload(file=image))
            elif isinstance(image, Path):
                if not image.exists():
                    raise ValueError(f"Image {image} does not exist.")
                images_source.append(self.client.files.upload(file=image.__str__()))
            else:
                raise ValueError(f"Image {image} is not a valid path or file object.")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=images_source,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )

        from PIL import Image
        from io import BytesIO

        path = Path(save_path)
        if path.is_dir():
            # If it's a directory, append default filename
            save_path = path / "image.png"

        for part in response.candidates[0].content.parts:
            if part.text is not None and verbose:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO((part.inline_data.data)))
                image.save(save_path.__str__())
                # image.show()
                if verbose:
                    print(f"Image saved to {save_path.__str__()}")

        return save_path

    def get_num_tokens(self, text: str) -> int:
        try:
            num_tokens = calculate_token(text, model_name=self.model_name)
        except Exception:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(text))

        return num_tokens


def convert_vision_prompt(prompt: list):
    """convert the vision prompt to the correct format"""
    converted_prompts = []
    for idx, p in enumerate(prompt):
        if isinstance(p, str):
            converted_prompts.append(types.Part.from_text(text=p))
        elif isinstance(p, dict):
            if "text" in p:
                converted_prompts.append(types.Part.from_text(text=p["text"]))
            elif "image_url" in p:
                converted_prompts.append(
                    types.Part.from_bytes(data=p["image_url"], mime_type=p["mime_type"])
                )
            else:
                raise ValueError(f"Invalid prompt format: {p}")
        else:
            raise ValueError(f"Invalid prompt format: {p}")

    return converted_prompts


def check_format_prompt(prompts: list):
    """check and format the prompt to fit the correct gemini format"""
    converted_prompts = []
    system_prompt = None
    for idx, prompt in enumerate(prompts):
        if prompt["role"] == "user" or prompt["role"] == "human":
            if "parts" in prompt:
                converted_prompts.append(types.Part.from_text(text=prompt["parts"][0]))
            elif "content" in prompt:
                if isinstance(prompt["content"], list):
                    return convert_vision_prompt(prompt["content"]), system_prompt
                converted_prompts.append(types.Part.from_text(text=prompt["content"]))

        elif (
            prompt["role"] == "assistant"
            or prompt["role"] == "system"
            or prompt["role"] == "model"
        ):
            if "parts" in prompt:
                system_prompt = prompt["parts"][0]
            elif "content" in prompt:
                system_prompt = prompt["content"]

    return converted_prompts, system_prompt


def calculate_token(
    prompt: str,
    model_name: str = "gemini-1.5-flash",
):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    num_tokens = client.models.count_tokens(
        model=model_name, contents=prompt
    ).total_tokens

    return num_tokens


class gemini_embed(BaseModel, Embeddings):
    """gemini embedding models."""

    client: genai.Client = None
    model_name: str = "gemini-embedding-exp-03-07"
    """Model name to use."""
    # """Keyword arguments to pass to the model."""
    embedConfig: types.EmbedContentConfig = None
    """Keyword arguments to pass when calling the `encode` method of the model."""

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like genai.Client

    def __init__(self, model_name: str, api_key: str, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.embedConfig = types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))

        result = self.client.models.embed_content(
            model=self.model_name, contents=texts, config=self.embedConfig
        )
        embeddings = []
        # get list of embedding floats #
        for r in result.embeddings:
            embeddings.append(list(map(lambda x: float(x), r.values)))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        result = self.client.models.embed_content(
            model=self.model_name, contents=text, config=self.embedConfig
        )

        return result.embeddings[0].values
