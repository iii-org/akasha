from openai import OpenAI, AzureOpenAI
from langchain.llms.base import LLM
import base64
from pathlib import Path
import openai as ai
from typing import Optional, List, Union, Any, Generator
from pydantic import BaseModel
import asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI


async def _async_get_completion(
    client: Union[AsyncAzureOpenAI, AsyncOpenAI],
    model: str,
    prompt: dict,
    max_tokens: int,
    temperature: float,
    verbose: bool = False,
):
    response = await client.chat.completions.create(
        model=model,
        messages=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print(response.choices[0].message.content, end="", flush=True)

    return response.choices[0].message.content


class AzureOpenAIClient(LLM):
    model_name: str = "gpt-4o"
    max_output_tokens: int = 1024
    temperature: float = 0.0
    api_key: str = ""
    api_type: str = "openai"
    api_base: str = ""
    api_version: str = "2023-05-15"
    client: Union[OpenAI, AzureOpenAI, None] = None

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        api_type: str = "openai",
        api_base: str = "",
        api_version: str = "2023-05-15",
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.api_type = api_type
        self.api_key = api_key
        if api_type == "openai":
            self.client = OpenAI(
                api_key=api_key,
            )

        else:
            self.client = AzureOpenAI(
                azure_endpoint=api_base,
                azure_deployment=model_name,
                api_key=api_key,
                api_version=api_version,
            )
            self.api_base = api_base
            self.api_version = api_version

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return f"openai:{self.model_name}"

    def invoke(
        self,
        messages: Union[list, str],
        stop: Optional[List[str]] = None,
        verbose: bool = True,
        response_format: Union[dict, BaseModel, None] = None,
    ) -> str:
        """run llm and get the stream generator

        Args:
            prompt (str): _description_

        Yields:
            Generator: _description_
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }

        fully_text = ""
        if response_format is not None:
            params["response_format"] = response_format
            response = self.client.beta.chat.completions.parse(**params)
            message = response.choices[0].message

            return message.content
        else:
            params["stream"] = True
            response = self.client.chat.completions.create(**params)
        count = 0
        for chunk in response:
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.content:
                fully_text += chunk.choices[0].delta.content
                if verbose:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            count += 1
        return fully_text

    def _call(
        self,
        messages: list,
        stop: Optional[List[str]] = None,
        verbose: bool = True,
        resposne_format: Union[dict, BaseModel, None] = None,
    ) -> str:
        return self.invoke(messages, stop, verbose, resposne_format)

    def stream(
        self,
        messages: Union[list, str],
        stop: Optional[List[str]] = None,
        verbose: bool = True,
        resposne_format: Union[dict, BaseModel, None] = None,
    ) -> Generator[Any, Any, Any]:
        """run llm and get the stream generator

        Args:
            prompt (str): _description_

        Yields:
            Generator: _description_
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": True,
        }
        # fully_text = ""

        response = self.client.chat.completions.create(**params)

        for chunk in response:
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.content:
                if verbose:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                yield chunk.choices[0].delta.content
                # fully_text += chunk.choices[0].delta.content

        return

    def batch(
        self,
        inputs: list,
        stop: Optional[List[str]] = None,
        verbose: bool = False,
        resposne_format: Union[dict, BaseModel, None] = None,
    ) -> List[str]:
        messages_list = []
        for prompt in inputs:
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            messages_list.append(prompt)

        async def process_batch(prompts):
            # 根據 api_type 決定用哪個 async client
            if self.api_type == "openai":
                client = AsyncOpenAI(api_key=self.api_key)
            else:
                client = AsyncAzureOpenAI(
                    azure_endpoint=self.api_base,
                    azure_deployment=self.model_name,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            tasks = [
                _async_get_completion(
                    client,
                    self.model_name,
                    prompt,
                    self.max_output_tokens,
                    self.temperature,
                    verbose=verbose,
                )
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

        return asyncio.run(process_batch(messages_list))

    def generate(
        self,
        prompt: str,
        save_path: str = "./image.png",
        size: str = "auto",
        quality: str = "auto",
        moderation: str = "auto",
        background: str = "auto",
        verbose: bool = True,
    ) -> Path:
        """generate image based on the user prompt.\n
        Args:
            prompt (str): _description_
            size (str, optional): _description_. Defaults to "auto".
            quality (str, optional): _description_. Defaults to "auto".
            output_format (str, optional): _description_. Defaults to "png".
            moderation (str, optional): _description_. Defaults to "auto".
            background (str, optional): _description_. Defaults to "auto".
        """
        if "dall" in self.model_name.lower():
            if size == "auto":
                size = "1024x1024"
            if quality == "auto":
                quality = "standard"
        # Convert save_path to Path object
        path = Path(save_path)
        allowed_exts = {"png", "jpeg", "webp"}

        if path.is_dir():
            # If it's a directory, append default filename
            path = path / "image.png"
            output_format = "png"
        else:
            ext = path.suffix.lower().lstrip(".")
            if ext not in allowed_exts:
                # Invalid or missing extension → default to .png
                path = path.with_suffix(".png")
                output_format = "png"
            else:
                output_format = ext
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "output_format": output_format,
        }
        if "dall" in self.model_name.lower():
            # For DALL-E models, we need to set the response format to b64_json
            params["response_format"] = "b64_json"
        # Generate image
        result = self.client.images.generate(
            **params,
        )
        ## decode the image

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        ## save the image
        with open(path, "wb") as f:
            f.write(image_bytes)
        if verbose:
            print(f"Image saved to {path}")

        return path

    def edit(
        self,
        prompt: str,
        images: str,
        save_path: str = "./image.png",
        size: str = "auto",
        quality: str = "auto",
        moderation: str = "auto",
        background: str = "auto",
        verbose: bool = True,
    ) -> Path:
        """generate image based on the user prompt.\n
        Args:
            prompt (str): _description_
            size (str, optional): _description_. Defaults to "auto".
            quality (str, optional): _description_. Defaults to "auto".
            output_format (str, optional): _description_. Defaults to "png".
            moderation (str, optional): _description_. Defaults to "auto".
            background (str, optional): _description_. Defaults to "auto".
        """
        ai._azure._deployments_endpoints.add("/images/edits")  # hack fix openai bug
        ## process the image list
        images_source = []
        for image in images:
            if isinstance(image, str):
                # use Path to check if the image is exist
                image_path = Path(image)
                if not image_path.exists():
                    raise ValueError(f"Image {image} does not exist.")
                images_source.append(open(image, "rb"))
            elif isinstance(image, Path):
                if not image.exists():
                    raise ValueError(f"Image {image} does not exist.")
                images_source.append(open(image, "rb"))
            else:
                raise ValueError(f"Image {image} is not a valid path or file object.")

        # Convert save_path to Path object
        path = Path(save_path)
        allowed_exts = {"png", "jpeg", "webp"}

        if path.is_dir():
            # If it's a directory, append default filename
            path = path / "image.png"

        else:
            ext = path.suffix.lower().lstrip(".")
            if ext not in allowed_exts:
                # Invalid or missing extension → default to .png
                path = path.with_suffix(".png")

        # Generate image
        result = self.client.images.edit(
            image=images_source,
            model=self.model_name,
            prompt=prompt,
            size=size,
        )

        ## decode the image

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        ## save the image
        with open(path, "wb") as f:
            f.write(image_bytes)
        if verbose:
            print(f"Image saved to {path}")

        return path

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in the text using the model's tokenizer.

        Args:
            text (str): The text to be tokenized.

        Returns:
            int: The number of tokens in the text.
        """
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            tokens = encoding.encode(text)
            num_tokens = len(tokens)
            return num_tokens
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(text))
            return num_tokens
