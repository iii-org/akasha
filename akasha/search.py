from langchain.schema import Document
from langchain_community.retrievers import (
    TFIDFRetriever,
    SVMRetriever,
    KNNRetriever,
)
from rank_bm25 import BM25Okapi
from langchain_community.utils.math import cosine_similarity
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings
from typing import Any, List, Optional, Callable, Union, Tuple, Dict, Iterable
import numpy as np
import akasha.helper as helper
from akasha.db import dbs
import jieba


def _get_threshold_times(db: dbs):
    times = 1
    for embeds in db.get_embeds():

        if np.max(embeds) > times:
            times *= 10

    return times


def _get_relevant_doc_auto(
    retriver_list: list,
    docs_list: list,
    query: str,
    k: int,
    times: int,
    verbose: bool = False,
) -> list:
    """try every solution to get  to search relevant documents.

    Args:
        **db (Chromadb)**: chroma db\n
        **query (str)**: the query str used to search similar documents\n
        **k (int)**: for each search type, return first k documents\n
        **times (int)**: the magnification of max embeddings(?)\n

    Returns:
        list: list of selected relevant Documents
    """
    rate = 1.0
    if times != 1:
        rate = 0.3

    # mmrR = retriver_list[0]
    # docs_mmr, mmr_scores = mmrR._gs(query)
    # print("MMR: ", mmr_scores, len(mmr_scores), "\n\n")
    #print(docs_mmr[:4])

    ### svm ###
    svmR = retriver_list[0]
    docs_svm, svm_scores = svmR._gs(query)
    #print("SVM: ", svm_scores, docs_svm[0], "\n\n")

    # ### tfidf ###

    # tfretriever = retriver_list[1]
    # docs_tf, tf_scores = tfretriever._gs(query)
    #print("TFIDF", tf_scores, docs_tf[0], "\n\n")

    # ### knn ###
    # knnR = myKNNRetriever.from_db(db, embeddings, k, 0.0)
    # docs_knn, knn_scores = knnR._gs(query)
    # print("KNN: ", knn_scores, len(knn_scores), "\n\n")

    ### bm25 ###
    bm25R = retriver_list[1]
    docs_bm25, bm25_scores = bm25R._gs(query)
    #print("BM25: ", bm25_scores[:10], len(bm25_scores), "\n\n")

    ### decide which to use ###
    backup_docs = []
    final_docs = []  #docs_mmr[0]
    del bm25R
    ## backup_docs is all documents from docs_svm that svm_scores>0.2 ##
    low = 0
    for i in range(len(svm_scores)):
        if svm_scores[i] >= 0.2 * rate:
            backup_docs.append(docs_svm[i])
        else:
            low = i
            break

    if bm25_scores[0] >= 70:

        ## find out the idx that the sorted tf_scores is not 0
        idx = 0
        for i in range(len(bm25_scores)):
            if bm25_scores[i] < 70 or i >= 2:
                idx = i
                break
        final_docs.extend(docs_bm25[:idx])

    final_docs.extend(backup_docs)
    final_docs.extend(docs_svm[low:])
    return final_docs


def _get_relevant_doc_auto_rerank(
    retriver_list: list,
    docs_list: list,
    query: str,
    k: int,
    times: int,
    verbose: bool = False,
) -> list:
    """try every solution to get  to search relevant documents.

    Args:
        **db (Chromadb)**: chroma db\n
        **query (str)**: the query str used to search similar documents\n
        **k (int)**: for each search type, return first k documents\n
        **times (int)**: the magnification of max embeddings(?)\n

    Returns:
        list: list of selected relevant Documents
    """
    rate = 1.0
    if times != 1:
        rate = 0.3

    ### svm ###
    svmR = retriver_list[0]
    docs_svm, svm_scores = svmR._gs(query)
    #print("SVM: ", svm_scores, docs_svm[0], "\n\n")

    # ### tfidf ###

    # tfretriever = retriver_list[1]
    # docs_tf, tf_scores = tfretriever._gs(query)
    #print("TFIDF", tf_scores, docs_tf[0], "\n\n")

    ### bm25 ###
    bm25R = retriver_list[1]
    docs_bm25, bm25_scores = bm25R._gs(query)
    #print("BM25: ", bm25_scores[:10], len(bm25_scores), "\n\n")

    ### decide which to use ###
    backup_docs = []
    final_docs = []  #docs_mmr[0]
    del svmR, bm25R
    ## backup_docs is all documents from docs_svm that svm_scores>0.2 ##

    for i in range(len(svm_scores)):
        if svm_scores[i] >= 0.2 * rate:
            backup_docs.append(docs_svm[i])
        else:
            break

    if bm25_scores[0] >= 70:
        if verbose:
            print("<<search>>go to bm25\n\n")

        ## find out the idx that the sorted tf_scores is not 0
        idx = 0
        for i in range(len(bm25_scores)):
            if bm25_scores[i] < 70 or i >= 2:
                idx = i
                break
        final_docs.extend(docs_bm25[:idx])

    if svm_scores[0] >= 0.35 * rate:
        if verbose:
            print("<<search>>go to svm\n\n")

        final_docs.extend(backup_docs)

    elif svm_scores[0] >= 0.2 * rate:
        if verbose:
            print("<<search>>go to svm+rerank\n\n")
        final_docs.extend(rerank_reduce(query, backup_docs, k))

    else:
        if verbose:
            print("<<search>>go to rerank\n\n")
        final_docs.extend(rerank_reduce(query, docs_list, k))

    return final_docs


def _merge_docs(docs_list: list, topK: int, language: str, verbose: bool,
                max_doc_len: int, model) -> Tuple[list, int]:
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

            words_len = helper.get_doc_length(language, docs[i].page_content)
            token_len = model.get_num_tokens(docs[i].page_content)
            if cur_count + words_len > max_doc_len:
                if verbose:
                    print("\nwords length: ", cur_count, "tokens: ", cur_token)
                return res, cur_count, cur_token

            cur_count += words_len
            cur_token += token_len
            res.append(docs[i])
            page_contents.add(docs[i].page_content)

    if verbose:
        print("words length: ", cur_count, "tokens: ", cur_token)

    return res, cur_count, cur_token


def get_retrivers(
    db: Union[dbs, list],
    embeddings,
    use_rerank: bool,
    threshold: float,
    search_type: Union[str, Callable],
    log: dict = {},
) -> List[BaseRetriever]:
    """get the retrivers based on given search_type, default is merge, which contain 'mmr', 'svm', 'tfidf'
        and merge them together.

    Args:
        **db (Chromadb)**: chroma db\n
        **embeddings (Embeddings)**: embeddings used to store vector and search documents\n
        **query (str)**: the query str used to search similar documents\n
        **threshold (float)**: the similarity score threshold to select documents\n
        **search_type (str)**: search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf'.\n
        **verbose (bool)**: show log texts or not. Defaults to False.\n

    Returns:
        list: selected list of similar documents.
    """

    ### if use rerank to get more accurate similar documents, set topK to 200 ###
    if use_rerank:
        topK = 200
    else:
        topK = 199

    retriver_list = []
    if isinstance(embeddings, str):
        return retriver_list

    times = _get_threshold_times(db)
    if search_type == "auto":
        threshold = 0.0
    elif times != 1:
        threshold *= 0.3

    if callable(search_type):

        custom_retriver = customRetriever.from_db(db, embeddings, search_type,
                                                  topK, threshold, log)
        retriver_list.append(custom_retriver)

    else:
        search_type = search_type.lower()

        if search_type in ["merge", "tfidf", "auto", "auto_rerank", "bm25"]:
            docs_list = db.get_Documents()

        if search_type in ["mmr", "merge"]:
            mmr_retriver = myMMRRetriever.from_db(db, embeddings, topK,
                                                  threshold)
            retriver_list.append(mmr_retriver)

        if search_type in ["svm", "merge", "auto", "auto_rerank"]:
            svm_retriver = mySVMRetriever.from_db(db, embeddings, topK,
                                                  threshold)
            retriver_list.append(svm_retriver)

        if search_type in ["tfidf", "merge"]:
            tfidf_retriver = myTFIDFRetriever.from_documents(docs_list, k=topK)
            retriver_list.append(tfidf_retriver)

        if search_type in ["bm25", "auto", "auto_rerank"]:
            bm25_retriver = myBM25Retriever.from_documents(
                docs_list, topK, threshold)
            retriver_list.append(bm25_retriver)

        if search_type == "knn":
            knn_retriver = myKNNRetriever.from_db(db, embeddings, topK,
                                                  threshold)
            retriver_list.append(knn_retriver)

    if len(retriver_list) == 0:
        raise ValueError(
            f"cannot find search type {search_type}, end process\n")

    return retriver_list


def get_docs(
    db: Union[dbs, list],
    embeddings,
    retriver_list: list,
    query: str,
    use_rerank: bool,
    language: str,
    search_type: Union[str, Callable],
    verbose: bool,
    model,
    max_doc_len: int,
    compression: bool = False,
) -> Tuple[list, int, int]:
    """search docs based on given search_type, default is merge, which contain 'mmr', 'svm', 'tfidf'
        and merge them together.

    Args:
        **db (Chromadb)**: chroma db\n
        **embeddings (Embeddings)**: embeddings used to store vector and search documents\n
        **query (str)**: the query str used to search similar documents\n
        **topK (int)**: for each search type, return first topK documents\n
        **threshold (float)**: the similarity score threshold to select documents\n
        **language (str)**: default to chinese 'ch', otherwise english, the language of documents and prompt,
            use to make sure docs won't exceed max token size of llm input.\n
        **search_type (str)**: search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf'.\n
        **verbose (bool)**: show log texts or not. Defaults to False.\n
        **model ()**: large language model object\n

    Returns:
        list: selected list of similar documents.
    """

    ### if use rerank to get more accurate similar documents, set topK to 200 ###
    if use_rerank:
        topK = 200
    else:
        topK = 199

    final_docs = []

    if len(retriver_list) == 0:
        docs = rerank(query, db, 0.0, embeddings)
        docs, docs_len, tokens = _merge_docs([docs], topK, language, verbose,
                                             max_doc_len, model)
        return docs, docs_len, tokens

    if not callable(search_type):

        search_type = search_type.lower()

        if search_type == "auto":
            docs_list = db.get_Documents()
            times = _get_threshold_times(db)
            docs = _get_relevant_doc_auto(retriver_list, docs_list, query,
                                          topK, times, verbose)
            docs, docs_len, tokens = _merge_docs([docs], topK, language,
                                                 verbose, max_doc_len, model)
            return docs, docs_len, tokens
        elif search_type == "auto_rerank":
            docs_list = db.get_Documents()
            times = _get_threshold_times(db)
            docs = _get_relevant_doc_auto_rerank(retriver_list, docs_list,
                                                 query, topK, times, verbose)
            docs, docs_len, tokens = _merge_docs([docs], topK, language,
                                                 verbose, max_doc_len, model)
            return docs, docs_len, tokens

    for retri in retriver_list:
        if compression:
            compressor = LLMChainExtractor.from_llm(
                model, llm_chain_kwargs={"verbose": verbose})
            retri = ContextualCompressionRetriever(base_compressor=compressor,
                                                   base_retriever=retri)
        docs = retri._get_relevant_documents(query)
        # docs, scores = retri._gs(query)
        final_docs.append(docs)

    docs, docs_len, tokens = _merge_docs(final_docs, topK, language, verbose,
                                         max_doc_len, model)

    return docs, docs_len, tokens


def retri_docs(
    db: Union[dbs, list],
    embeddings,
    retriver_list: List[BaseRetriever],
    query: str,
    search_type: Union[str, Callable],
    topK: int,
    verbose: bool = True,
) -> list:
    """search docs based on given search_type, default is merge, which contain 'mmr', 'svm', 'tfidf'
        and merge them together.

    Args:
        **db (Chromadb)**: chroma db\n
        **embeddings (Embeddings)**: embeddings used to store vector and search documents\n
        **query (str)**: the query str used to search similar documents\n
        **topK (int)**: for each search type, return first topK documents\n
        **search_type (str)**: search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf'.\n
        **verbose (bool)**: show log texts or not. Defaults to False.\n

    Returns:
        list: selected list of similar documents.
    """

    ### if use rerank to get more accurate similar documents, set topK to 200 ###

    final_docs = []

    def merge_docs(
        docs_list: list,
        topK: int,
    ):
        res = []
        page_contents = set()
        for i in range(topK):
            for adocs in docs_list:
                if i >= len(docs_list):
                    continue

                if adocs.page_content in page_contents:
                    continue
                res.append(adocs)
                page_contents.add(adocs.page_content)
        return res

    if len(retriver_list) == 0:
        docs = rerank(query, db, 0.0, embeddings)

        return docs

    if not callable(search_type):

        search_type = search_type.lower()

        if search_type == "auto":
            docs_list = db.get_Documents()
            times = _get_threshold_times(db)
            docs = _get_relevant_doc_auto(retriver_list, docs_list, query,
                                          topK, times, verbose)
            docs = merge_docs(docs, topK)
            return docs
        elif search_type == "auto_rerank":
            docs_list = db.get_Documents()
            times = _get_threshold_times(db)
            docs = _get_relevant_doc_auto_rerank(retriver_list, docs_list,
                                                 query, topK, times, verbose)
            docs = merge_docs(docs, topK)
            return docs

    for retri in retriver_list:

        docs = retri._get_relevant_documents(query)
        # docs, scores = retri._gs(query)
        final_docs.extend(docs)

    docs = merge_docs(final_docs, topK)

    return docs


class myMMRRetriever(BaseRetriever):
    embeddings: Embeddings
    """Embeddings model to use."""
    index: Any
    """Index of embeddings."""
    texts: List[str]
    """List of texts to index."""
    metadata: List[dict]
    k: int = 3
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None
    log: dict = None
    lambda_mult: float = 0.5

    @classmethod
    def from_db(
        cls,
        db: dbs,
        embeddings: Embeddings,
        k: int = 3,
        relevancy_threshold: float = 0.2,
        log: dict = None,
        lambda_mult: float = 0.5,
    ):
        # db_data = _get_all_docs(db)
        index = np.array(db.get_embeds())
        texts = db.get_docs()
        metadata = db.get_metadatas()
        return cls(
            embeddings=embeddings,
            index=index,
            texts=texts,
            metadata=metadata,
            k=k,
            relevancy_threshold=relevancy_threshold,
            log=log,
            lambda_mult=lambda_mult,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """general function to retrieve relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]:  relevant documents
        """
        return self._gs(query)[0]

    def _get_relevant_documents(self, query: str) -> List[Document]:

        return self._gs(query)[0]

    def _gs(self, query: str) -> Tuple[List[Document], List[float]]:
        """implement using custom function to find relevant documents, the custom function func should
        have four input.
            1. a np.array of embedding vectors of query query_embeds np.array)
            2. a np.array of np.array contain embedding vectors of query and documents docs_embeds (np.array)
            3. the number of topK return documents k (int)
            4. relevant threshold from 0.0 ~ 1.0 threshold (float)
        And the function func should return a list of index which length is equals to k, represent
        the index of documents that are most relevant to the input query.

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """

        top_k_results = []
        query_embeds = np.array(self.embeddings.embed_query(query))
        docs_embeds = self.index

        if min(self.k, len(docs_embeds)) <= 0:
            return [], []
        if query_embeds.ndim == 1:
            query_embeds = np.expand_dims(query_embeds, axis=0)
        similarity_to_query = cosine_similarity(query_embeds, docs_embeds)[0]
        most_similar = int(np.argmax(similarity_to_query))
        mmr_scores = [similarity_to_query[most_similar]]
        relevant_docs_idx = [most_similar]
        selected = np.array([docs_embeds[most_similar]])
        while len(relevant_docs_idx) < min(self.k, len(docs_embeds)):
            best_score = -np.inf
            idx_to_add = -1
            similarity_to_selected = cosine_similarity(docs_embeds, selected)
            for i, query_score in enumerate(similarity_to_query):
                if i in relevant_docs_idx:
                    continue
                redundant_score = max(similarity_to_selected[i])
                equation_score = (self.lambda_mult * query_score -
                                  (1 - self.lambda_mult) * redundant_score)
                if equation_score > best_score:
                    best_score = equation_score
                    idx_to_add = i
            relevant_docs_idx.append(idx_to_add)
            mmr_scores.append(best_score)
            selected = np.append(selected, [docs_embeds[idx_to_add]], axis=0)

        ### from index rebuild the documents ###
        for idx in relevant_docs_idx[:self.k]:
            top_k_results.append(
                Document(page_content=self.texts[idx],
                         metadata=self.metadata[idx]))

        return top_k_results, mmr_scores[:self.k]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)[0]


class customRetriever(BaseRetriever):
    embeddings: Embeddings
    """Embeddings model to use."""
    index: Any
    """Index of embeddings."""
    texts: List[str]
    """List of texts to index."""
    metadata: List[dict]
    k: int = 3
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None
    func: Callable
    log: dict = None

    @classmethod
    def from_db(
        cls,
        db: dbs,
        embeddings: Embeddings,
        func: Callable,
        k: int = 3,
        relevancy_threshold: float = 0.2,
        log: dict = None,
    ):
        # db_data = _get_all_docs(db)
        index = np.array(db.get_embeds())
        texts = db.get_docs()
        metadata = db.get_metadatas()
        return cls(
            embeddings=embeddings,
            index=index,
            func=func,
            texts=texts,
            metadata=metadata,
            k=k,
            relevancy_threshold=relevancy_threshold,
            log=log,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """general function to retrieve relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]:  relevant documents
        """
        return self._gs(query)

    def _gs(self, query: str) -> List[Document]:
        """implement using custom function to find relevant documents, the custom function func should
        have four input.
            1. a np.array of embedding vectors of query query_embeds np.array)
            2. a np.array of np.array contain embedding vectors of query and documents docs_embeds (np.array)
            3. the number of topK return documents k (int)
            4. relevant threshold from 0.0 ~ 1.0 threshold (float)
        And the function func should return a list of index which length is equals to k, represent
        the index of documents that are most relevant to the input query.

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """

        top_k_results = []
        query_embeds = np.array(self.embeddings.embed_query(query))
        docs_embeds = self.index

        relevant_docs_idx = self.func(query_embeds, docs_embeds, self.k,
                                      self.relevancy_threshold, self.log)

        ### from index rebuild the documents ###
        for idx in relevant_docs_idx[:self.k]:
            top_k_results.append(
                Document(page_content=self.texts[idx],
                         metadata=self.metadata[idx]))

        return top_k_results

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)

    def _get_relevant_documents(self, query: str) -> List[Document]:

        return self._gs(query)


class myKNNRetriever(BaseRetriever):
    embeddings: Embeddings
    """Embeddings model to use."""
    index: Any
    """Index of embeddings."""
    texts: List[str]
    """List of texts to index."""
    metadata: List[dict]
    k: int = 3
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None

    @classmethod
    def from_db(
        cls,
        db: dbs,
        embeddings: Embeddings,
        k: int = 3,
        relevancy_threshold: float = 0.2,
        **kwargs: Any,
    ) -> KNNRetriever:
        # db_data = _get_all_docs(db)
        index = np.array(db.get_embeds())
        texts = db.get_docs()
        metadata = db.get_metadatas()
        return cls(
            embeddings=embeddings,
            index=index,
            texts=texts,
            metadata=metadata,
            k=k,
            relevancy_threshold=relevancy_threshold,
            **kwargs,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """general function to retrieve relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]:  relevant documents
        """
        return self._gs(query)[0]

    def _gs(self, query: str) -> Tuple[List[Document], List[float]]:
        """implement k-means search to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """
        query_embeds = np.array(self.embeddings.embed_query(query))
        # calc L2 norm
        index_embeds = self.index / np.sqrt(
            (self.index**2).sum(1, keepdims=True))
        query_embeds = query_embeds / np.sqrt((query_embeds**2).sum())

        similarities = index_embeds.dot(query_embeds)
        sorted_ix = np.argsort(-similarities)

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities -
                                   np.min(similarities)) / denominator
        # print([normalized_similarities[row]
        #        for row in sorted_ix[0:self.k]])  # stats

        top_k_scores = [
            normalized_similarities[row] for row in sorted_ix[0:self.k]
        ]
        top_k_results = [
            Document(page_content=self.texts[row], metadata=self.metadata[row])
            for row in sorted_ix[0:self.k]
            if (self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold)
        ]
        return top_k_results, top_k_scores

    def _aget_relevant_documents(self, query: str) -> List[Document]:
        """implement k-means search to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """
        return self._gs(query)[0]

    def _get_relevant_documents(self, query: str) -> List[Document]:

        return self._gs(query)[0]


class myTFIDFRetriever(TFIDFRetriever):

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        documents: Optional[List[Document]] = None,
        tfidf_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TFIDFRetriever:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError(
                "Could not import scikit-learn, please install with `pip install "
                "scikit-learn`.")

        tfidf_params = tfidf_params or {}
        vectorizer = TfidfVectorizer(**tfidf_params)
        tfidf_array = vectorizer.fit_transform(texts)
        return cls(vectorizer=vectorizer,
                   docs=documents,
                   tfidf_array=tfidf_array,
                   **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        *,
        tfidf_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TFIDFRetriever:
        texts, metadatas = zip(*((' '.join(list(jieba.cut(d.page_content))),
                                  d.metadata) for d in documents))
        return cls.from_texts(texts=texts,
                              tfidf_params=tfidf_params,
                              documents=documents,
                              **kwargs)

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        return self._gs(query)[0]

    def _gs(
        self,
        query: str,
    ) -> Tuple[List[Document], List[float]]:

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform(
            [query])  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
        results = cosine_similarity(self.tfidf_array, query_vec).reshape(
            (-1, ))  # Op -- (n_docs,1) -- Cosine Sim with each doc
        idxs = results.argsort()[-self.k:][::-1]
        return_docs = [self.docs[i] for i in idxs]
        return_values = [results[i] for i in idxs]

        return return_docs, return_values

    async def _aget_relevant_documents(self, query: str) -> List[Document]:

        return self._gs(query)[0]


class mySVMRetriever(BaseRetriever):
    embeddings: Embeddings
    """Embeddings model to use."""
    index: Any
    """Index of embeddings."""
    texts: List[str]
    """List of texts to index."""
    metadata: List[dict]
    k: int = 3
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None

    @classmethod
    def from_db(
        cls,
        db: dbs,
        embeddings: Embeddings,
        k: int = 3,
        relevancy_threshold: float = 0.2,
        **kwargs: Any,
    ) -> SVMRetriever:

        index = np.array(db.get_embeds())
        texts = db.get_docs()
        metadata = db.get_metadatas()
        return cls(
            embeddings=embeddings,
            index=index,
            texts=texts,
            metadata=metadata,
            k=k,
            relevancy_threshold=relevancy_threshold,
            **kwargs,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """general function to retrieve relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]:  relevant documents
        """
        return self._gs(query)[0]

    def _gs(self, query: str) -> Tuple[List[Document], List[float]]:
        """implement svm to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """
        try:
            from sklearn import svm
        except ImportError:
            raise ImportError(
                "Could not import scikit-learn, please install with `pip install "
                "scikit-learn`.")

        query_embeds = np.array(self.embeddings.embed_query(query))
        x = np.concatenate([query_embeds[None, ...], self.index])
        y = np.zeros(x.shape[0])
        y[0] = 1

        clf = svm.LinearSVC(class_weight="balanced",
                            verbose=False,
                            max_iter=10000,
                            tol=1e-5,
                            C=0.1)
        clf.fit(x, y)

        similarities = clf.decision_function(x)
        sorted_ix = np.argsort(-similarities)

        # svm.LinearSVC in scikit-learn is non-deterministic.
        # if a text is the same as a query, there is no guarantee
        # the query will be in the first index.
        # this performs a simple swap, this works because anything
        # left of the 0 should be equivalent.
        zero_index = np.where(sorted_ix == 0)[0][0]
        if zero_index != 0:
            sorted_ix[0], sorted_ix[zero_index] = sorted_ix[
                zero_index], sorted_ix[0]

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities -
                                   np.min(similarities)) / denominator

        top_k_results = []
        top_k_scores = []
        for row in sorted_ix[1:self.k + 1]:
            # print(normalized_similarities[row])  # stats
            if (self.relevancy_threshold is None or
                    normalized_similarities[row] >= self.relevancy_threshold):
                top_k_results.append(
                    Document(
                        page_content=self.texts[row - 1],
                        metadata=self.metadata[row - 1],
                    ))
                top_k_scores.append(normalized_similarities[row])
        return top_k_results, top_k_scores

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)[0]

    def _get_relevant_documents(self, query: str) -> List[Document]:

        return self._gs(query)[0]


class myBM25Retriever(BaseRetriever):
    bm25: BM25Okapi
    """BM25 class to use."""
    texts: List[str]
    """List of texts to index."""
    metadata: List[dict]
    docs: List[Document]
    k: int = 3
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None

    @classmethod
    def from_documents(
        cls,
        docs: List[Document],
        k: int = 3,
        relevancy_threshold: float = 0.2,
        **kwargs: Any,
    ) -> BaseRetriever:

        tokenize_corpus = [list(jieba.cut(doc.page_content)) for doc in docs]
        bm25 = BM25Okapi(tokenize_corpus)
        return cls(
            bm25=bm25,
            texts=[doc.page_content for doc in docs],
            metadata=[doc.metadata for doc in docs],
            k=k,
            docs=docs,
            relevancy_threshold=relevancy_threshold,
            **kwargs,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """general function to retrieve relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]:  relevant documents
        """
        return self._gs(query)[0]

    def _gs(self, query: str) -> Tuple[List[Document], List[float]]:
        """implement bm25 to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """

        tokenize_query = list(jieba.cut(query))
        docs_scores = self.bm25.get_scores(tokenize_query)
        top_k_idx = np.argsort(docs_scores)[::-1][:self.k]
        top_k_results = [self.docs[i] for i in top_k_idx]
        top_k_scores = [docs_scores[i] for i in top_k_idx]
        return top_k_results, top_k_scores

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)[0]

    def _get_relevant_documents(self, query: str) -> List[Document]:

        return self._gs(query)[0]


def rerank(query: str, docs: list, threshold: float, embed_name: str):
    import torch, gc
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = embed_name.split(":")[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
        device)
    model.eval()

    k, score_list = 0, []
    while k < len(docs):
        pairs = [[query, doc.page_content] for doc in docs[k:k + 10]]
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            scores = (model(**inputs,
                            return_dict=True).logits.view(-1, ).float())
        if k == 0:
            score_list = scores
        else:
            score_list = torch.cat([score_list, scores], dim=0)
        k += 10

    # Get the sorted indices in descending order
    sorted_indices = torch.argsort(score_list, descending=True)

    # Convert the indices to a Python list
    sorted_indices_list = sorted_indices.tolist()

    # Get the documents in the order of their scores, if lower than threshold, break
    documents = []
    for i in sorted_indices_list:
        if score_list[i] < threshold:
            break
        documents.append(docs[i])

    del model, inputs, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return documents


def rerank_reduce(query, docs, topK):
    import torch, gc
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "BAAI/bge-reranker-large"  # BAAI/bge-reranker-base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
        device)
    model.eval()
    #topK //= 2
    k, score_list = 0, []
    while k < len(docs):
        pairs = [[query, doc.page_content] for doc in docs[k:k + 10]]
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            scores = (model(**inputs,
                            return_dict=True).logits.view(-1, ).float())
        if k == 0:
            score_list = scores
        else:
            score_list = torch.cat([score_list, scores], dim=0)
        k += 10

    # Get the sorted indices in descending order
    sorted_indices = torch.argsort(score_list, descending=True)

    # Convert the indices to a Python list
    sorted_indices_list = sorted_indices.tolist()

    # Get the documents in the order of their scores, if lower than threshold, break
    documents = []
    for i in sorted_indices_list:
        if i > topK:
            break
        documents.append(docs[i])

    del model, inputs, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return documents
