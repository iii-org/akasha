import warnings
from pathlib import Path
from typing import Callable, Union, Tuple

# from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
import traceback
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from dotenv import dotenv_values
from akasha.utils.models.thinking import ThinkingBudget
from akasha.helper.base import separate_name, decide_embedding_type

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def _get_env_var(env_file: str = "") -> dict:
    """if env_file is not empty, get the environment variable from the file
        else get the environment variable from the os.environ

    Args:
        env_file (str, optional): the path of .env file. Defaults to "".

    Returns:
        dict: return the environment variable dictionary
    """

    require_env = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_BASE_URL",
        "SERPER_API_KEY",
        "GEMINI_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "ANTHROPIC_API_KEY",
        "REMOTE_API_KEY",
        "OLLAMA_API_BASE",
        "OLLAMA_API_KEY",
    ]
    if env_file == "" or not Path(env_file).exists():
        env_dict = {}
        os_env_dict = os.environ.copy()
        for req in require_env:
            if req in os_env_dict:
                env_dict[req] = os_env_dict[req]

    else:
        env_dict = dotenv_values(env_file)
    return env_dict


def _openai_endpoint(env_dict: dict, provider: str = "openai") -> Tuple[str, str]:
    """Return the endpoint and key for the explicitly selected provider."""
    if provider == "azure":
        api_key = env_dict.get("AZURE_OPENAI_API_KEY")
        base_url = env_dict.get("AZURE_OPENAI_BASE_URL")
        if not api_key or not base_url:
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_BASE_URL must be provided together."
            )
        return base_url, api_key

    api_key = env_dict.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("can not find the OPENAI_API_KEY in environment variable.\n\n")
    return env_dict.get("OPENAI_BASE_URL", ""), api_key


def handle_embeddings(
    embedding_name: str = "openai:text-embedding-ada-002",
    verbose: bool = False,
    env_file: str = "",
) -> Embeddings:
    """create model client used in document QA, default if openai "gpt-3.5-turbo"
        use openai:text-embedding-ada-002 as default.
    Args:
        **embedding_name (str)**: embeddings client you want to use.
            format is (type:name), which is the model type and model name.\n
            for example, "openai:text-embedding-ada-002", "huggingface:all-MiniLM-L6-v2".\n
        **logs (list)**: list that store logs\n
        **verbose (bool)**: print logs or not\n

    Returns:
        vars: embeddings client
    """
    if isinstance(embedding_name, Embeddings):
        return embedding_name

    if isinstance(embedding_name, Callable):
        from akasha.utils.models.custom import custom_embed

        embeddings = custom_embed(func=embedding_name)
        if verbose:
            print("selected custom embedding.")
        return embeddings

    embedding_type, embedding_name = separate_name(embedding_name)
    env_dict = _get_env_var(env_file)
    if embedding_type in ["text-embedding-ada-002", "openai", "openaiembeddings"]:
        from langchain_openai import OpenAIEmbeddings

        base_url, api_key = _openai_endpoint(env_dict, provider="openai")
        kwargs = {"model": embedding_name, "api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        embeddings = OpenAIEmbeddings(**kwargs)
        info = "selected openai embeddings.\n"

    elif embedding_type in [
        "huggingface",
        "huggingfaceembeddings",
        "transformers",
        "transformer",
        "hf",
    ]:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_name, model_kwargs={"trust_remote_code": True}
            )
            info = "selected hugging face embeddings.\n"
        except ImportError:
            raise ImportError(
                "Feature requiring 'torch/transformers' is not installed. Please install with: pip install akasha-terminal[full]"
            )

    elif embedding_type in [
        "tf",
        "tensorflow",
        "tensorflowhub",
        "tensorflowhubembeddings",
        "tensorflowembeddings",
    ]:
        try:
            from langchain_community.embeddings import TensorflowHubEmbeddings

            embeddings = TensorflowHubEmbeddings()
            info = "selected tensorflow embeddings.\n"
        except ImportError:
            raise ImportError(
                "Feature requiring 'tensorflow' is not installed. Please install with: pip install akasha-terminal[full]"
            )

    elif embedding_type in ["gemini", "gemi", "google"]:
        from akasha.utils.models.gemi import gemini_embed

        if not env_dict.get("GEMINI_API_KEY"):
            raise ValueError("can not find the GEMINI_API_KEY in environment variable.\n\n")
        embeddings = gemini_embed(
            model_name=embedding_name, api_key=env_dict["GEMINI_API_KEY"]
        )
        info = "selected gemini embeddings.\n"

    elif embedding_type in ["azure", "azure-openai", "azure_openai"]:
        from langchain_openai import OpenAIEmbeddings

        base_url, api_key = _openai_endpoint(env_dict, provider="azure")
        kwargs = {"model": embedding_name, "api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        embeddings = OpenAIEmbeddings(**kwargs)

        info = "selected Azure OpenAI-compatible embeddings.\n"

    else:
        from langchain_openai import OpenAIEmbeddings

        base_url, api_key = _openai_endpoint(env_dict, provider="openai")
        kwargs = {"model": embedding_name, "api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        embeddings = OpenAIEmbeddings(**kwargs)

        info = "can not find the embeddings, use openai as default.\n"
    if verbose:
        print(info)
    return embeddings


def handle_embeddings_and_name(
    embeddings: Union[str, Embeddings, Callable] = "openai:text-embedding-ada-002",
    verbose: bool = False,
    env_file: str = "",
) -> Tuple[Embeddings, str]:
    """get the embeddings object and embed name

    Args:
        embed (_type_, optional): _description_. Defaults to "openai:text-embedding-ada-002".
        verbose (bool, optional): _description_. Defaults to False.
        env_file (str, optional): _description_. Defaults to "".

    Returns:
        Tuple[Embeddings, str]: _description_
    """

    if callable(embeddings):
        model_name = embeddings.__name__
    elif isinstance(embeddings, str):
        model_name = embeddings
    else:
        return embeddings, decide_embedding_type(embeddings)

    model_obj = handle_embeddings(embeddings, verbose, env_file)

    return model_obj, model_name


def handle_model(
    model_name: Union[str, Callable] = "openai:gpt-3.5-turbo",
    verbose: bool = False,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    env_file: str = "",
    thinking: bool = False,
    thinking_budget: ThinkingBudget = None,
) -> BaseLanguageModel:
    """create model client used in document QA, default if openai "gpt-3.5-turbo"

    Args:
       ** model_name (str)**: open ai model name like "gpt-3.5-turbo","text-davinci-003", "text-davinci-002"\n
        **logs (list)**: list that store logs\n
        **verbose (bool)**: print logs or not\n

    Returns:
        vars: model client
    """
    if isinstance(model_name, BaseLanguageModel):
        if thinking:
            raise ValueError(
                "thinking settings must be configured on a directly supplied ChatModel."
            )
        return model_name

    if isinstance(model_name, Callable):
        from akasha.utils.models.custom import custom_model

        model = custom_model(func=model_name, temperature=temperature)
        if verbose:
            print("selected custom model.")
        return model

    model_type, model_name = separate_name(model_name)
    env_dict = _get_env_var(env_file)
    if model_type in ["remote", "server", "tgi", "text-generation-inference"]:
        from akasha.utils.models.remo import remote_model

        remote_api_key = "123"
        remote_model_name = "remote_model"
        base_url = model_name
        if "REMOTE_API_KEY" in env_dict:
            remote_api_key = env_dict["REMOTE_API_KEY"]
        if "@" in model_name:
            base_url, remote_model_name = model_name.split("@")

        info = "selected remote model. \n"
        model = remote_model(
            base_url,
            temperature,
            api_key=remote_api_key,
            model_name=remote_model_name,
            max_output_tokens=max_output_tokens,
        )

    elif model_type in ["ollama"]:
        from akasha.utils.models.chat import build_chat_model

        ollama_env = dict(env_dict)
        ollama_env.setdefault(
            "OLLAMA_API_BASE", env_dict.get("OLLAMA_API_BASE", "http://localhost:11434")
        )
        if "@" in model_name:
            ollama_api_base, ollama_model_name = model_name.split("@", 1)
            ollama_env["OLLAMA_API_BASE"] = (
                ollama_api_base.strip() or env_dict.get("OLLAMA_API_BASE", "http://localhost:11434")
            )
        else:
            ollama_model_name = model_name

        if not ollama_model_name.strip():
            raise ValueError(
                "ollama model name is required. Use 'ollama:<model>' or "
                "'ollama:<base_url>@<model>'."
            )

        info = "selected ollama ChatModel. \n"
        model = build_chat_model(
            "ollama", ollama_model_name, ollama_env, temperature, max_output_tokens,
            thinking, thinking_budget,
        )

    elif model_type in ["azure", "azure-openai", "azure_openai"]:
        from akasha.utils.models.chat import build_chat_model

        info = "selected Azure OpenAI-compatible ChatModel. \n"
        model = build_chat_model(
            "azure", model_name, env_dict, temperature, max_output_tokens,
            thinking, thinking_budget,
        )

    elif model_type in ["google", "gemini", "gemi"]:
        from akasha.utils.models.chat import build_chat_model

        info = "selected gemini ChatModel. \n"
        model = build_chat_model(
            "gemini", model_name, env_dict, temperature, max_output_tokens,
            thinking, thinking_budget,
        )

    elif model_type in ["anthropic", "anthropicai", "claude", "anthro"]:
        from akasha.utils.models.chat import build_chat_model

        info = "selected anthropic ChatModel. \n"
        model = build_chat_model(
            "anthropic", model_name, env_dict, temperature, max_output_tokens,
            thinking, thinking_budget,
        )

    elif (
        model_type in ["llama-cpu", "llama-gpu", "llama", "llama2", "llama-cpp"]
        and model_name != ""
    ):
        try:
            from akasha.utils.models.llamacpp2 import LlamaCPP

            model = LlamaCPP(
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            info = "selected llama-cpp model\n"
        except ImportError:
            raise ImportError(
                "Feature requiring 'llama-cpp-python' is not installed. Please install with: pip install akasha-terminal[full]"
            )
    elif model_type in [
        "huggingface",
        "huggingfacehub",
        "transformers",
        "transformer",
        "huggingface-hub",
        "hf",
    ]:
        try:
            from akasha.utils.models.hf import hf_model

            model = hf_model(
                model_name=model_name,
                env_dict=env_dict,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            info = f"selected huggingface model {model_name}.\n"
        except ImportError:
            raise ImportError(
                "Feature requiring 'torch/transformers' is not installed. Please install with: pip install akasha-terminal[full]"
            )

    elif model_type in ["chatglm", "chatglm2", "glm"]:
        try:
            from akasha.utils.models.chglm import chatGLM

            model = chatGLM(
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            info = f"selected chatglm model {model_name}.\n"
        except ImportError:
            raise ImportError(
                "Feature requiring 'torch/transformers' is not installed. Please install with: pip install akasha-terminal[full]"
            )

    elif model_type in ["lora", "peft"]:
        try:
            from akasha.utils.models.gtq import peft_Llama2

            model = peft_Llama2(model_name_or_path=model_name, temperature=temperature)
            info = f"selected peft model {model_name}.\n"
        except ImportError:
            raise ImportError(
                "Feature requiring 'torch/transformers/peft' is not installed. Please install with: pip install akasha-terminal[full]"
            )

    elif model_type in ["gptq"]:
        try:
            if model_name.lower().find("taiwan-llama") != -1:
                from akasha.utils.models.gtq import TaiwanLLaMaGPTQ

                model = TaiwanLLaMaGPTQ(
                    model_name_or_path=model_name, temperature=temperature
                )

            else:
                from akasha.utils.models.gtq import gptq

                model = gptq(
                    model_name_or_path=model_name,
                    temperature=temperature,
                    bit4=True,
                    max_token=4096,
                )
            info = f"selected gptq model {model_name}.\n"
        except ImportError:
            raise ImportError(
                "Feature requiring 'torch/transformers/auto-gptq' is not installed. Please install with: pip install akasha-terminal[full]"
            )
    else:
        if model_type not in ["openai", "gpt-3.5", "gpt"]:
            info = f"can not find the model {model_type}:{model_name}, use openai as default.\n"
            model_name = "gpt-3.5-turbo"
            print(info)
        from akasha.utils.models.chat import build_chat_model

        model = build_chat_model(
            "openai", model_name, env_dict, temperature, max_output_tokens,
            thinking, thinking_budget,
        )

        info = f"selected openai model {model_name}.\n"
    if verbose:
        print(info)

    return model


def handle_client(model: str, env_file: str = ""):
    client_type, model_name = separate_name(model)
    env_dict = _get_env_var(env_file)
    if client_type in ["openai", "azure", "azure-openai", "azure_openai"]:
        from akasha.utils.models.azure_openai import AzureOpenAIClient

        api_base, api_key = _openai_endpoint(
            env_dict,
            provider="azure" if client_type in ["azure", "azure-openai", "azure_openai"] else "openai",
        )
        client = AzureOpenAIClient(
            api_key=api_key,
            model_name=model_name,
            api_type="openai",
            api_base=api_base,
        )
    elif client_type == "gemini":
        from akasha.utils.models.gemi import gemini_model

        client = gemini_model(
            model_name=model_name,
            api_key=env_dict["GEMINI_API_KEY"],
        )
    else:
        raise ValueError(f"Unknown client type: {client_type}")

    return client


def handle_model_and_name(
    model: Union[str, Callable, BaseLanguageModel] = "openai:gpt-3.5-turbo",
    verbose: bool = False,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    env_file: str = "",
) -> Tuple[BaseLanguageModel, str]:
    """get the model object and model name

    Args:
        model (_type_, optional): _description_. Defaults to "openai:gpt-3.5-turbo".
        verbose (bool, optional): _description_. Defaults to False.
        temperature (float, optional): _description_. Defaults to 0.0.
        max_output_tokens (int, optional): _description_. Defaults to 1024.
        env_file (str, optional): _description_. Defaults to "".

    Returns:
        Tuple[BaseLanguageModel, str]: _description_
    """
    if isinstance(model, BaseLanguageModel):
        return model, model._llm_type

    if callable(model):
        model_name = model.__name__
    else:
        model_name = model

    model_obj = handle_model(model, verbose, temperature, max_output_tokens, env_file)

    return model_obj, model_name


def handle_model_type(
    search_type: Union[str, BaseLanguageModel, Embeddings], verbose: bool = False
) -> str:
    if isinstance(search_type, BaseLanguageModel):
        search_type_str = search_type._llm_type

    elif isinstance(search_type, Embeddings):
        search_type_str = decide_embedding_type(search_type)

    elif callable(search_type):
        search_type_str = search_type.__name__

    else:
        search_type_str = search_type

    # if verbose:
    #     print("search type is :", search_type_str)

    return search_type_str
