from langchain.schema import Document
from langchain.retrievers import (
    TFIDFRetriever,
    ContextualCompressionRetriever,
    SVMRetriever,
    KNNRetriever,
)

from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.vectorstores import Chroma, chroma
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings
from typing import Any, List, Optional, Callable, Union
import numpy as np
import akasha.helper as helper
import akasha.prompts as prompts
from akasha.db import dbs


def _get_threshold_times(db: dbs):
    times = 1
    for embeds in db.get_embeds():

        if np.max(embeds) > times:
            times *= 10

    return times


def _get_relevant_doc_custom(
    db: dbs,
    embeddings,
    func: Callable,
    query: str,
    k: int,
    relevancy_threshold: float,
    log: dict,
    model,
    compression: bool = False,
    verbose: bool = False,
):
    customR = customRetriever.from_db(db, embeddings, func, k,
                                      relevancy_threshold, log)
    if compression:
        # compressor = LLMChainExtractor.from_llm(model)
        # customc = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=customR)
        # docs = customc.get_relevant_documents(query)
        docs = []
        pre_docs = customR.get_relevant_documents(query)
        for pre_doc in pre_docs:
            pre_doc.page_content = helper.call_model(
                model,
                prompts.format_compression_prompt(query, pre_doc.page_content))
            if pre_doc.page_content.replace(" ", "") != "":
                if verbose:
                    print(pre_doc.page_content)
                docs.append(pre_doc)

    else:
        docs = customR.get_relevant_documents(query)

    if k >= 100:
        docs = rerank_reduce(query, docs, k)

    return docs


def __get_relevant_doc_knn(
    db: dbs,
    embeddings,
    query: str,
    k: int,
    relevancy_threshold: float,
    model,
    compression: bool = False,
    verbose: bool = False,
):
    """use KNN to find relevant doc from query.

    Args:
        **db (Chromadb):** chroma db\n
        **embeddings (Embeddings)**: embeddings used to store vector and search documents\n
        **query (str)**: the query str used to search similar documents\n
        **k (int)**: for each search type, return first k documents\n
        **relevancy_threshold (float)**: the similarity score threshold to select documents\n

    Returns:
        list: list of Documents
    """

    knnR = myKNNRetriever.from_db(db, embeddings, k, relevancy_threshold)
    if compression:
        compressor = LLMChainExtractor.from_llm(
            model, llm_chain_kwargs={"verbose": verbose})
        knnc = ContextualCompressionRetriever(base_compressor=compressor,
                                              base_retriever=knnR)
        docs = knnc.get_relevant_documents(query)
    else:
        docs = knnR.get_relevant_documents(query)

    if k >= 100:
        docs = rerank_reduce(query, docs, k)

    return docs


def _get_relevant_doc_tfidf(
    docs_list: list,
    query: str,
    k: int,
    model,
    compression: bool = False,
    verbose: bool = False,
) -> list:
    """use Term Frequency-Inverse Document Frequency to find relevant doc from query.

    Args:
        **docs_list list of Documents_object\n
        **query (str)**: the query str used to search similar documents\n
        **k (int)**: for each search type, return first k documents\n

    Returns:
        list: list of Documents
    """

    retriever = TFIDFRetriever.from_documents(docs_list, k=k)
    if compression:
        compressor = LLMChainExtractor.from_llm(
            model, llm_chain_kwargs={"verbose": verbose})
        tfidfc = ContextualCompressionRetriever(base_compressor=compressor,
                                                base_retriever=retriever)
        docs = tfidfc.get_relevant_documents(query)
    else:
        docs = retriever.get_relevant_documents(query)

    if k >= 100:
        docs = rerank_reduce(query, docs[:k], k)

    return docs[:k]


def _get_relevant_doc_svm(
    db: dbs,
    embeddings,
    query: str,
    k: int,
    relevancy_threshold: float,
    model,
    compression: bool = False,
    verbose: bool = False,
) -> list:
    """use SVM to find relevant doc from query.

    Args:
        **db (Chromadb**): chroma db\n
        **embeddings (Embeddings)**: embeddings used to store vector and search documents\n
        **query (str)**: the query str used to search similar documents\n
        **k (int)**: for each search type, return first k documents\n
        **relevancy_threshold (float)**: the similarity score threshold to select documents\n

    Returns:
        list: list of Documents
    """

    svmR = mySVMRetriever.from_db(db, embeddings, k, relevancy_threshold)
    if compression:
        compressor = LLMChainExtractor.from_llm(
            model, llm_chain_kwargs={"verbose": verbose})
        svmc = ContextualCompressionRetriever(base_compressor=compressor,
                                              base_retriever=svmR)
        docs = svmc.get_relevant_documents(query)
    else:
        docs = svmR.get_relevant_documents(query)

    if k >= 100:
        docs = rerank_reduce(query, docs, k)
    return docs


def _get_relevant_doc_mmr(
    db: dbs,
    embeddings,
    query: str,
    k: int,
    relevancy_threshold: float,
    model,
    compression: bool = False,
    verbose: bool = False,
) -> list:
    """use Chroma.as_retriever().get_relevant_document() to search relevant documents.

    Args:
        **db (Chromadb)**: chroma db\n
        **query (str)**: the query str used to search similar documents\n
        **k (int)**: for each search type, return first k documents\n
        **relevancy_threshold (float)**: the similarity score threshold to select documents\n

    Returns:
        list: list of selected relevant Documents
    """

    # retriever = Chroma(embedding_function=embeddings).as_retriever(search_type="mmr",\
    #     search_kwargs={"k": k,'score_threshold': relevancy_threshold})
    # retriever.add_documents(docs_list)
    #import uuid

    embedding_list = db.get_embeds()
    text_list = db.get_docs()
    metadata_list = db.get_metadatas()
    id_list = db.get_ids()
    #id_list = [str(uuid.uuid1()) for _ in range(len(embedding_list))]
    retriever = Chroma(embedding_function=embeddings)
    retriever._collection.upsert(
        embeddings=embedding_list,
        documents=text_list,
        ids=id_list,
        metadatas=metadata_list,
    )
    retriever = retriever.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "score_threshold": relevancy_threshold
        },
    )

    if compression:
        compressor = LLMChainExtractor.from_llm(
            model, llm_chain_kwargs={"verbose": verbose})
        mmrc = ContextualCompressionRetriever(base_compressor=compressor,
                                              base_retriever=retriever)
        docs = mmrc.get_relevant_documents(query)
    else:
        docs = retriever.get_relevant_documents(query)

    del retriever

    if k >= 100:
        docs = rerank_reduce(query, docs, k)

    return docs


def _merge_docs(docs_list: list, topK: int, language: str, verbose: bool,
                max_doc_len: int, model) -> (list, int):
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


def get_docs(
    db: Union[dbs, list],
    embeddings,
    query: str,
    topK: int,
    threshold: float,
    language: str,
    search_type: Union[str, Callable],
    verbose: bool,
    model,
    max_token: int,
    log: dict,
    compression: bool = False,
) -> (list, int):
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

    if callable(search_type):

        times = _get_threshold_times(db)
        threshold *= times
        docs_cust = _get_relevant_doc_custom(
            db,
            embeddings,
            search_type,
            query,
            topK,
            threshold,
            log,
            model,
            compression,
            verbose,
        )
        docs, docs_len, tokens = _merge_docs([docs_cust], topK, language,
                                             verbose, max_token, model)

    elif isinstance(embeddings, str):
        docs = rerank(query, db, threshold, embeddings)
        docs, docs_len, tokens = _merge_docs([docs], topK, language, verbose,
                                             max_token, model)

    else:
        search_type = search_type.lower()
        times = _get_threshold_times(db)
        threshold *= times
        if search_type == "merge" or "tfidf":
            docs_list = db.get_Documents()

        if search_type == "merge":
            docs_mmr = _get_relevant_doc_mmr(db, embeddings, query, topK,
                                             threshold, model, compression,
                                             verbose)
            docs_svm = _get_relevant_doc_svm(db, embeddings, query, topK,
                                             threshold, model, compression,
                                             verbose)
            docs_tfidf = _get_relevant_doc_tfidf(docs_list, query, topK, model,
                                                 compression, verbose)

            docs, docs_len, tokens = _merge_docs(
                [docs_tfidf, docs_svm, docs_mmr],
                topK,
                language,
                verbose,
                max_token,
                model,
            )

        elif search_type == "mmr":
            docs_mmr = _get_relevant_doc_mmr(db, embeddings, query, topK,
                                             threshold, model, compression,
                                             verbose)
            docs, docs_len, tokens = _merge_docs([docs_mmr], topK, language,
                                                 verbose, max_token, model)

        elif search_type == "svm":
            docs_svm = _get_relevant_doc_svm(db, embeddings, query, topK,
                                             threshold, model, compression,
                                             verbose)
            docs, docs_len, tokens = _merge_docs([docs_svm], topK, language,
                                                 verbose, max_token, model)

        elif search_type == "tfidf":
            docs_tfidf = _get_relevant_doc_tfidf(docs_list, query, topK, model,
                                                 compression, verbose)
            docs, docs_len, tokens = _merge_docs([docs_tfidf], topK, language,
                                                 verbose, max_token, model)

        elif search_type == "knn":
            docs_knn = __get_relevant_doc_knn(db, embeddings, query, topK,
                                              threshold, model, compression,
                                              verbose)
            docs, docs_len, tokens = _merge_docs([docs_knn], topK, language,
                                                 verbose, max_token, model)

        else:
            info = f"cannot find search type {search_type}, end process\n"
            print(info)

            return None, None

    return docs, docs_len, tokens


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
        return self._ks(query)

    def _ks(self, query: str) -> List[Document]:
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

        top_k_results = [
            Document(page_content=self.texts[row], metadata=self.metadata[row])
            for row in sorted_ix[0:self.k]
            if (self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold)
        ]
        return top_k_results

    def _aget_relevant_documents(self, query: str) -> List[Document]:
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

        top_k_results = [
            Document(page_content=self.texts[row], metadata=self.metadata[row])
            for row in sorted_ix[0:self.k]
            if (self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold)
        ]
        return top_k_results


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
        return self._gs(query)

    def _gs(self, query: str) -> List[Document]:
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
                            tol=1e-6,
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
        for row in sorted_ix[1:self.k + 1]:
            if (self.relevancy_threshold is None or
                    normalized_similarities[row] >= self.relevancy_threshold):
                top_k_results.append(
                    Document(
                        page_content=self.texts[row - 1],
                        metadata=self.metadata[row - 1],
                    ))
        return top_k_results

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)


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
    topK //= 5
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
