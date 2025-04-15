from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from akasha.utils.prompts.format import language_dict
from langchain_core.embeddings import Embeddings
from typing import Union, Callable, Tuple, List
from langchain.schema import Document
import jieba
import json
import re
import traceback
import opencc

jieba.setLogLevel(jieba.logging.INFO)  ## ignore logging jieba model information


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
        res_name = ":".join(sep[1:])
    elif len(sep) < 2:
        ### if the format type not equal to type:name ###
        res_type = sep[0].lower()
        res_name = ""
    else:
        res_type = sep[0].lower()
        res_name = sep[1]

    return res_type, res_name


def decide_embedding_type(embeddings: Embeddings) -> str:
    """check the embedding type and return the type:name

    Args:
        embeddings (Embeddings): embedding class

    Raises:
        Exception: _description_

    Returns:
        str: type:name
    """
    if isinstance(embeddings, OpenAIEmbeddings) or isinstance(
        embeddings, AzureOpenAIEmbeddings
    ):
        return "openai:" + embeddings.model
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if isinstance(embeddings, GoogleGenerativeAIEmbeddings):
            return "gemini:" + embeddings.model

        from akasha.utils.models.custom import custom_embed

        if isinstance(embeddings, custom_embed):
            return embeddings.model_name

        else:
            from langchain_huggingface import HuggingFaceEmbeddings

            if isinstance(embeddings, HuggingFaceEmbeddings):
                return "hf:" + embeddings.model_name

            from langchain_community.embeddings import TensorflowHubEmbeddings

            if isinstance(embeddings, TensorflowHubEmbeddings):
                return "tf:" + embeddings.model_url

            else:
                raise Exception("can not find the embeddings type.")


def get_embedding_type_and_name(
    embeddings: Union[Embeddings, str, Callable],
) -> Tuple[str, str]:
    """get the type and name of the embeddings"""

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


def extract_json(s: str) -> Union[dict, None]:
    """parse the JSON part of the string

    Args:
        s (str): string that contains JSON part

    Returns:
        Union[dict, None]: return the JSON part of the string, if not found return None
    """

    match = re.search(r"\{(?:[^{}]*|{[^{}]*})*\}", s, re.DOTALL)

    if not match:
        return None

    json_part = match.group()

    try:
        # Attempt to parse JSON
        return json.loads(json_part)
    except json.JSONDecodeError:
        traceback.print_exc()
        print(f"Invalid JSON extracted: {json_part}")
        return None

    # # Use a regular expression to find the JSON part of the string
    # match = re.search(r'\{.*\}', s, re.DOTALL)
    # stack = []
    # start = 0
    # s = s.replace("\n", " ").replace("\r", " ").strip()
    # if match is None:
    #     return None

    # s = match.group()
    # for i, c in enumerate(s):
    #     if c == '{':
    #         stack.append(i)
    #     elif c == '}':
    #         if stack:
    #             start = stack.pop()
    #             if not stack:
    #                 try:
    #                     json_part = s[start:i + 1]
    #                     json_part = json_part.replace("\n", "")
    #                     json_part = json_part.replace("'", '"')
    #                     # Try to parse the JSON part of the string
    #                     return json.loads(json_part)
    #                 except json.JSONDecodeError:
    #                     traceback.print_exc()
    #                     print(s[start:i + 1])
    #                     print(
    #                         "The JSON part of the string is not well-formatted"
    #                     )
    #                     return None
    # return None
