from langchain.docstore.document import Document
from typing import Union, List


def handle_params(
    model: str,
    language: str,
    search_type: str,
    threshold: float = -1.0,
    embeddings: str = "",
    chunk_size: int = -1,
) -> dict:
    """save running parameters into dictionary in order to parse to aiido

    Args:
        **model (str)**: model name\n
        **embeddings (str)**: embedding name\n
        **chunk_size (int)**: chunk size of texts from documents\n
        **search_type (str)**: search type of finding relevant documents\n
        **threshold (float)**: only return documents that has similarity score larger than threshold\n
        **language (str)**: 'ch' for chinese and 'en' for other.\n

    Returns:
        dict: parameter dictionary
    """
    params = {}
    if model != "":
        params["model"] = model
    if embeddings != "":
        params["embeddings"] = embeddings
    if search_type != "":
        params["search_type"] = search_type
    if threshold != -1.0:
        params["threshold"] = threshold
    if language != "":
        params["language"] = language_dict[language]

    if chunk_size != -1:
        params["chunk_size"] = chunk_size
    return params


def handle_metrics(doc_length: int, time: float, tokens: int) -> dict:
    """save running metrics into dictionary in order to parse to aiido

    Args:
        **doc_length (int)**: length of texts from relevant documents  \n
        **time (float)**: total spent time\n
        **tokens (int)**: total tokens of texts from relevant documents\n

    Returns:
        dict: metric dictionary
    """
    metrics = {}

    metrics["doc_length"] = doc_length
    metrics["time"] = time
    metrics["tokens"] = tokens
    return metrics


def handle_table(prompt: str, docs: List[Union[Document, str]], response: str) -> dict:
    """save running results into dictionary in order to parse to aiido

    Args:
        **prompt (str)**: input query/question\n
        **docs (list)**: Document list and metadata\n
        **response (str)**: results from llm response\n

    Returns:
        dict: table dictionary
    """
    table = {}
    table["prompt"] = prompt
    table["response"] = response
    if len(docs) != 0:
        try:
            inputs = [doc.page_content for doc in docs]

            if "source" in docs[0].metadata and "page" in docs[0].metadata:
                # if the metadata has source and page, we can use it to show the source of the document
                metadata = [
                    doc.metadata["source"] + "    page: " + str(doc.metadata["page"])
                    for doc in docs
                ]
            elif "title" in docs[0].metadata and "url" in docs[0].metadata:
                # if the metadata has title and url, we can use it to show the source of the document
                metadata = [
                    doc.metadata["title"] + "    url: " + doc.metadata["url"]
                    for doc in docs
                ]
            else:
                raise Exception(
                    "metadata does not have source or page, or title and url"
                )
        except Exception:
            metadata = ["none" for _ in docs]
            inputs = [doc for doc in docs]
        table["inputs"] = inputs
        table["metadata"] = metadata

    else:
        table["inputs"] = ["none"]
        table["metadata"] = ["none"]

    return table


def handle_score_table(
    table: dict, bert: float, rouge: float, llm_score: float
) -> dict:
    """add each response's bert and rouge score into table dictionary

    Args:
        **table (dict)**: table dictionary that store texts data for a run of experiment.\n
        **bert (float)**: bert score\n
        **rouge (float)**: rouge score\n

    Returns:
        dict: table dictionary
    """

    table["bert"] = bert
    table["rouge"] = rouge
    table["llm_score"] = llm_score

    return table


def handle_language(language: str):
    if language not in language_dict:
        print("language not supported, use chinese as default\n\n")
        return "ch"
    else:
        return language


language_dict = {
    "en": "english",
    "ch": "traditional chinese",
    "jp": "japanese",
    "ja": "japanese",
    "zh": "traditional chinese",
    "cn": "simplified chinese",
    "de": "german",
    "el": "greek",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "ko": "korean",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "tr": "turkish",
    "vi": "vietnamese",
    "hi": "hindi",
    "ar": "arabic",
    "th": "thai",
    "id": "indonesian",
    "no": "norwegian",
    "sv": "swedish",
    "fi": "finnish",
    "da": "danish",
    "cs": "czech",
    "hu": "hungarian",
}

websearch_language_dict = {
    "en": {"brave": "en", "serper": "en", "wiki": "en"},
    "ch": {"brave": "zh-hant", "serper": "zh-TW", "wiki": "zh-TW"},
    "jp": {"brave": "ja", "serper": "ja", "wiki": "jp"},
    "ja": {"brave": "ja", "serper": "ja", "wiki": "jp"},
}

websearch_country_dict = {
    "en": {"brave": "us", "serper": "us", "wiki": "us"},
    "ch": {"brave": "tw", "serper": "tw", "wiki": "tw"},
    "jp": {"brave": "jp", "serper": "jp", "wiki": "jp"},
    "ja": {"brave": "jp", "serper": "jp", "wiki": "jp"},
}
