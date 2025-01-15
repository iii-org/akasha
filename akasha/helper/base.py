from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.embeddings import TensorflowHubEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from akasha.utils.models.hf import custom_embed
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from typing import Union, Callable, Tuple


def separate_name(name: str):
    """separate type:name by ':'

    Args:
        **name (str)**: string with format "type:name" \n

    Returns:
        (str, str): res_type , res_name
    """
    sep = name.split(":")

    if len(sep) > 2:
        res_type = sep[0].lower()
        res_name = ':'.join(sep[1:])
    elif len(sep) < 2:
        ### if the format type not equal to type:name ###
        res_type = sep[0].lower()
        res_name = ""
    else:
        res_type = sep[0].lower()
        res_name = sep[1]

    return res_type, res_name


def decide_embedding_type(embeddings: Embeddings) -> str:

    if isinstance(embeddings, custom_embed):
        return embeddings.model_name

    elif isinstance(embeddings, GoogleGenerativeAIEmbeddings):
        return "gemini:" + embeddings.model

    elif isinstance(embeddings, OpenAIEmbeddings) or isinstance(
            embeddings, AzureOpenAIEmbeddings):
        return "openai:" + embeddings.model

    elif isinstance(embeddings, HuggingFaceEmbeddings):
        return "hf:" + embeddings.model_name

    elif isinstance(embeddings, TensorflowHubEmbeddings):
        return "tf:" + embeddings.model_url

    else:
        raise Exception("can not find the embeddings type.")


def get_embedding_type_and_name(
        embeddings: Union[Embeddings, str, Callable]) -> Tuple[str, str]:

    if isinstance(embeddings, Embeddings):
        embeddings_name = decide_embedding_type(embeddings)

    if callable(embeddings):
        embeddings_name = embeddings.__name__
    else:
        embeddings_name = embeddings

    embed_type, embed_name = separate_name(embeddings_name)

    return embed_type, embed_name
