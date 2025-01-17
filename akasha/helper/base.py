from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.embeddings import TensorflowHubEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from akasha.utils.models.hf import custom_embed
from akasha.utils.prompts.format import language_dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from typing import Union, Callable, Tuple, List
from langchain.schema import Document
import jieba

jieba.setLogLevel(
    jieba.logging.INFO)  ## ignore logging jieba model information
import opencc

cc = opencc.OpenCC("s2twp")


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

    if callable(embeddings):
        embeddings_name = embeddings.__name__
    elif isinstance(embeddings, str):
        embeddings_name = embeddings
    else:
        embeddings_name = decide_embedding_type(embeddings)

    embed_type, embed_name = separate_name(embeddings_name)

    return embed_type, embed_name


def sim_to_trad(text: str) -> str:
    """convert simplified chinese to traditional chinese

    Args:
        **text (str)**: simplified chinese\n

    Returns:
        str: traditional chinese
    """
    global cc
    return cc.convert(text)


def get_doc_length(language: str, text: str) -> int:
    """calculate the length of terms in a giving Document

    Args:
        **language (str)**: 'ch' for chinese and 'en' for others, default 'ch'\n
        **doc (Document)**: Document object\n

    Returns:
        doc_length: int Docuemtn length
    """

    if "chinese" in language_dict[language]:
        doc_length = len(list(jieba.cut(text)))
    else:
        doc_length = len(text.split())
    return doc_length


def get_docs_length(language: str, docs: List[Document]) -> int:
    """calculate the total length of terms in giving documents

    Args:
        language (str): 'ch' for chinese and 'en' for others, default 'ch'\n
        docs (list): list of Documents\n

    Returns:
        docs_length: int total Document length
    """
    docs_length = 0
    for doc in docs:
        docs_length += get_doc_length(language, doc.page_content)
    return docs_length
