from typing import List, Tuple, Union, Callable
from akasha.helper.base import get_doc_length
from akasha.helper.token_counter import myTokenizer
from .auto_search import get_relevant_doc_auto
from langchain.schema import Document, BaseRetriever
from langchain_core.language_models.base import BaseLanguageModel


def _merge_docs(
    docs_list: List[List[Document]],
    topK: int,
    language: str,
    max_input_tokens: int = 3000,
    model: str = "openai:gpt-3.5-turbo",
) -> Tuple[list, int]:
    """merge different search types documents, if total len of documents too large,
        will not select all documents.
        use jieba to count length of chinese words, use split space otherwise.

    Args:
        **docs_list (list)**: list of all docs from selected search types\n
        **topK (int)**: for each search type, select topK documents\n
        **language (str)**: 'ch' for chinese, otherwise use split space to count words, default is chinese\n
        **verbose (bool)**: show log texts or not. Defaults to False.\n
        **max_token (int)**: max token size of llm input.\n

    Returns:
        list: merged list of Documents
    """
    res = []
    cur_count, cur_token = 0, 0
    page_contents = set()
    for i in range(topK):
        for docs in docs_list:
            if i >= len(docs):
                continue

            if docs[i].page_content in page_contents:
                continue

            words_len = get_doc_length(language, docs[i].page_content)
            # token_len = model.get_num_tokens(docs[i].page_content)
            token_len = myTokenizer.compute_tokens(docs[i].page_content, model)
            if cur_token + token_len > max_input_tokens:
                return res, cur_count, cur_token

            cur_count += words_len
            cur_token += token_len
            res.append(docs[i])
            page_contents.add(docs[i].page_content)

    return res, cur_count, cur_token


def search_docs(
    retriver_list: list,
    query: str,
    model: Union[str, BaseLanguageModel] = "openai:gpt-3.5-turbo",
    max_input_tokens: int = 3000,
    search_type: Union[str, Callable] = "auto",
    language: str = "ch",
) -> Tuple[list, int, int]:
    """search docs based on given search_type, default is merge, which contain 'mmr', 'svm', 'tfidf'
        and merge them together.

    Args:
        **retriver_list (list)**: list of retrievers that the search_type needed\n
        **query (str)**: the query str used to search similar documents\n
        **language (str)**: default to chinese 'ch', otherwise english, the language of documents and prompt,
            use to make sure docs won't exceed max token size of llm input.\n
        **search_type (str)**: search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf'.\n
        **model ()**: large language model name\n
        **max_input_tokens (int)**: max token size of llm input.\n

    Returns:
        list: selected list of similar documents.
    """

    topK = 10000

    final_docs = []
    if isinstance(model, BaseLanguageModel):
        try:
            model = model._llm_type
        except Exception:
            model = "openai:gpt-3.5-turbo"

    if not callable(search_type):
        search_type = search_type.lower()

        if search_type == "auto":
            docs = get_relevant_doc_auto(
                retriver_list,
                query,
            )
            docs, docs_len, tokens = _merge_docs(
                [docs], topK, language, max_input_tokens, model
            )
            return docs, docs_len, tokens
    for retri in retriver_list:
        docs = retri._get_relevant_documents(query)
        final_docs.append(docs)

    docs, docs_len, tokens = _merge_docs(
        final_docs, topK, language, max_input_tokens, model
    )

    return docs, docs_len, tokens


def retri_docs(
    retriver_list: List[BaseRetriever],
    query: str,
    search_type: Union[str, Callable],
    topK: int,
) -> list:
    """search docs based on given search_type, default is merge, which contain 'mmr', 'svm', 'tfidf'
        and merge them together.

    Args:
        **retriver_list (list)**: list of retrievers that the search_type needed\n
        **query (str)**: the query str used to search similar documents\n
        **topK (int)**: for each search type, return first topK documents\n
        **search_type (str)**: search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf'.\n

    Returns:
        list: selected list of similar documents.
    """

    final_docs = []

    def merge_docs(
        docs_list: List[Document],
        topK: int,
    ):
        res = []
        page_contents = set()

        for i, adocs in enumerate(docs_list):
            if len(page_contents) >= topK:
                break

            if adocs.page_content in page_contents:
                continue
            res.append(adocs)
            page_contents.add(adocs.page_content)
        return res

    if not callable(search_type):
        search_type = search_type.lower()

        if search_type == "auto":
            docs = get_relevant_doc_auto(
                retriver_list,
                query,
            )
            docs = merge_docs(docs, topK)
            return docs

    for retri in retriver_list:
        docs = retri._get_relevant_documents(query)
        # docs, scores = retri._gs(query)
        final_docs.extend(docs)

    docs = merge_docs(final_docs, topK)

    return docs
