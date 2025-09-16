import warnings
from pathlib import Path
from typing import Callable, Union, Tuple

# from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import (
    OpenAIEmbeddings,
    AzureOpenAIEmbeddings,
)
import os
import traceback
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from dotenv import dotenv_values
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
        "OPENAI_API_BASE",
        "OPENAI_API_VERSION",
        "OPEANI_API_TYPE",
        "AZURE_API_KEY",
        "AZURE_API_BASE",
        "AZURE_API_VERSION",
        "AZURE_API_TYPE",
        "SERPER_API_KEY",
        "GEMINI_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "ANTHROPIC_API_KEY",
        "REMOTE_API_KEY",
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


def _handle_azure_env(env_dict: dict) -> Tuple[str, str, str]:
    """from environment variable dictionary env_dict get the api_base, api_key, api_version

    Returns:
        (str, str, str): api_base, api_key, api_version
    """
    check_env, ret, count = ["BASE", "KEY", "VERSION"], ["", "", ""], 0
    try:
        for check in check_env:
            if f"AZURE_API_{check}" in env_dict:
                ret[count] = env_dict[f"AZURE_API_{check}"]
                # if "OPENAI_API_BASE" in os.environ:
                # os.environ.pop("OPENAI_API_BASE", None)
            elif f"OPENAI_API_{check}" in os.environ:
                ret[count] = env_dict[f"OPENAI_API_{check}"]
                if check == "BASE":
                    env_dict["AZURE_API_BASE"] = os.environ["OPENAI_API_BASE"]
                    env_dict.pop("OPENAI_API_BASE", None)
            else:
                if check == "VERSION":
                    ret[count] = "2023-05-15"
                else:
                    raise Exception(
                        f"can not find the openai {check} in environment variable.\n\n"
                    )
            count += 1
    except Exception as err:
        traceback.print_exc()
        print(err)

    return ret[0], ret[1], ret[2]


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
        import openai

        if ("AZURE_API_TYPE" in env_dict and env_dict["AZURE_API_TYPE"] == "azure") or (
            "OPENAI_API_TYPE" in env_dict and env_dict["OPENAI_API_TYPE"] == "azure"
        ):
            embedding_name = embedding_name.replace(".", "")
            api_base, api_key, api_version = _handle_azure_env(env_dict)
            embeddings = AzureOpenAIEmbeddings(
                model=embedding_name,
                azure_endpoint=api_base,
                api_key=api_key,
                api_version=api_version,
                validate_base_url=False,
            )

        else:
            if "OPENAI_API_KEY" not in env_dict:
                raise Exception(
                    "can not find the OPENAI_API_KEY in environment variable.\n\n"
                )

            openai.api_type = "open_ai"
            embeddings = OpenAIEmbeddings(
                model=embedding_name,
                openai_api_base="https://api.openai.com/v1",
                api_key=env_dict["OPENAI_API_KEY"],
                openai_api_type="open_ai",
            )
        info = "selected openai embeddings.\n"

    elif embedding_type in ["gemini", "gemi", "google"]:
        from akasha.utils.models.gemi import gemini_embed

        embeddings = gemini_embed(
            model_name=embedding_name, api_key=os.environ["GEMINI_API_KEY"]
        )
        info = "selected gemini embeddings.\n"

    else:
        import openai

        if ("AZURE_API_TYPE" in env_dict and env_dict["AZURE_API_TYPE"] == "azure") or (
            "OPENAI_API_TYPE" in env_dict and env_dict["OPENAI_API_TYPE"] == "azure"
        ):
            embedding_name = embedding_name.replace(".", "")
            api_base, api_key, api_version = _handle_azure_env(env_dict)
            embeddings = AzureOpenAIEmbeddings(
                model=embedding_name,
                azure_endpoint=api_base,
                api_key=api_key,
                api_version=api_version,
                validate_base_url=False,
            )

        else:
            if "OPENAI_API_KEY" not in env_dict:
                raise Exception(
                    "can not find the OPENAI_API_KEY in environment variable.\n\n"
                )
            openai.api_type = "open_ai"

            embeddings = OpenAIEmbeddings(
                model=embedding_name,
                openai_api_base="https://api.openai.com/v1",
                api_key=env_dict["OPENAI_API_KEY"],
                openai_api_type="open_ai",
            )

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

    elif model_type in ["google", "gemini", "gemi"]:
        from akasha.utils.models.gemi import gemini_model

        if "GEMINI_API_KEY" not in env_dict:
            raise Exception(
                "can not find the GEMINI_API_KEY in environment variable.\n\n"
            )
        info = "selected gemini model. \n"
        model = gemini_model(
            model_name=model_name,
            api_key=env_dict["GEMINI_API_KEY"],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    elif model_type in ["anthropic", "anthropicai", "claude", "anthro"]:
        from akasha.utils.models.anthro import anthropic_model

        if "ANTHROPIC_API_KEY" not in env_dict:
            raise Exception(
                "can not find the ANTHROPIC_API_KEY in environment variable.\n\n"
            )
        info = "selected anthropic model. \n"
        model = anthropic_model(
            model_name=model_name,
            api_key=env_dict["ANTHROPIC_API_KEY"],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    else:
        if model_type not in ["openai", "gpt-3.5", "gpt"]:
            info = f"can not find the model {model_type}:{model_name}, use openai as default.\n"
            model_name = "gpt-3.5-turbo"
            print(info)
        import openai
        from akasha.utils.models.azure_openai import AzureOpenAIClient

        if ("AZURE_API_TYPE" in env_dict and env_dict["AZURE_API_TYPE"] == "azure") or (
            "OPENAI_API_TYPE" in env_dict and env_dict["OPENAI_API_TYPE"] == "azure"
        ):
            model_name = model_name.replace(".", "")
            api_base, api_key, api_version = _handle_azure_env(env_dict)
            model = AzureOpenAIClient(
                api_key=api_key,
                model_name=model_name,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                api_type="azure",
                api_base=api_base,
                api_version=api_version,
            )
        else:
            if "OPENAI_API_KEY" not in env_dict:
                raise Exception(
                    "can not find the OPENAI_API_KEY in environment variable.\n\n"
                )
            openai.api_type = "open_ai"
            model = AzureOpenAIClient(
                api_key=env_dict["OPENAI_API_KEY"],
                model_name=model_name,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                api_type="openai",
            )

        info = f"selected openai model {model_name}.\n"
    if verbose:
        print(info)

    return model


def handle_client(model: str, env_file: str = ""):
    client_type, model_name = separate_name(model)
    env_dict = _get_env_var(env_file)
    if client_type == "openai":
        from akasha.utils.models.azure_openai import AzureOpenAIClient

        if ("AZURE_API_TYPE" in env_dict and env_dict["AZURE_API_TYPE"] == "azure") or (
            "OPENAI_API_TYPE" in env_dict and env_dict["OPENAI_API_TYPE"] == "azure"
        ):
            model_name = model_name.replace(".", "")
            api_base, api_key, api_version = _handle_azure_env(env_dict)

            client = AzureOpenAIClient(
                api_key=api_key,
                model_name=model_name,
                api_type="azure",
                api_base=api_base,
                api_version=api_version,
            )

        else:
            if "OPENAI_API_KEY" not in env_dict:
                raise Exception(
                    "can not find the OPENAI_API_KEY in environment variable.\n\n"
                )
            client = AzureOpenAIClient(
                api_key=env_dict["OPENAI_API_KEY"],
                model_name=model_name,
                api_type="openai",
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
