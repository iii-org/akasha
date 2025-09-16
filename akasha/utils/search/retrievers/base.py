from .retri_bm25 import myBM25Retriever
from .retri_custom import customRetriever
from .retri_knn import myKNNRetriever
from .retri_mmr import myMMRRetriever
from .retri_svm import mySVMRetriever
from .retri_tfidf import myTFIDFRetriever
from .retri_faiss import myFAISSRetriever

from typing import List, Union, Callable
from akasha.utils.db.db_structure import dbs
from akasha.helper.handle_objects import handle_embeddings_and_name
from langchain_core.embeddings import Embeddings
from langchain.schema import BaseRetriever


def get_retrivers(
    db: Union[dbs, list],
    embeddings: Union[Embeddings, str],
    threshold: float = 0.0,
    search_type: Union[str, Callable] = "auto",
    env_file: str = "",
) -> List[BaseRetriever]:
    """get the retrivers based on given search_type, default is auto, which contain, 'svm', 'bm25'.
       'merge' method contain 'mmr','svm','tfidf' and merge them together.

    Args:
        **db (Chromadb)**: chroma db\n
        **embeddings (Union[Embeddings, str])**: embeddings used to store vector and search documents\n
        **query (str)**: the query str used to search similar documents\n
        **threshold (float)**: (deprecated) the similarity score threshold to select documents\n
        **search_type (str)**: search type to find similar documents from db, .
            includes 'auto', 'merge', 'mmr', 'svm', 'tfidf', 'bm25'.\n
        **use_rerank (bool)**: use rerank model to search docs. Defaults to False.\n

    Returns:
        List[BaseRetriever]: selected list of retrievers that the search_type needed .
    """

    topK = 10000

    retriver_list = []

    embeddings, embed_name = handle_embeddings_and_name(embeddings, False, env_file)
    if callable(search_type):
        custom_retriver = customRetriever.from_db(
            db, embeddings, search_type, topK, threshold
        )
        retriver_list.append(custom_retriver)

    else:
        search_type = search_type.lower()

        if search_type in ["merge", "tfidf", "auto", "auto_rerank", "bm25", "rerank"]:
            docs_list = db.get_Documents()

        if search_type in ["mmr", "merge"]:
            mmr_retriver = myMMRRetriever.from_db(db, embeddings, topK, threshold)
            retriver_list.append(mmr_retriver)

        if search_type in ["svm", "merge"]:
            svm_retriver = mySVMRetriever.from_db(db, embeddings, topK, threshold)
            retriver_list.append(svm_retriver)

        if search_type in ["tfidf", "merge"]:
            tfidf_retriver = myTFIDFRetriever.from_documents(docs_list, k=topK)
            retriver_list.append(tfidf_retriver)

        if search_type in ["faiss", "FAISS", "meta", "facebook"]:
            faiss_retriver = myFAISSRetriever.from_db(db, embeddings, topK, threshold)
            retriver_list.append(faiss_retriver)

        if search_type in ["knn", "auto", "auto_rerank"]:
            knn_retriver = myKNNRetriever.from_db(db, embeddings, topK, threshold)
            retriver_list.append(knn_retriver)

        if search_type in ["bm25", "auto", "auto_rerank"]:
            bm25_retriver = myBM25Retriever.from_documents(docs_list, topK, threshold)
            retriver_list.append(bm25_retriver)

    if len(retriver_list) == 0:
        raise ValueError(f"cannot find search type {search_type}, end process\n")

    return retriver_list
