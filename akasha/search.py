from langchain.schema import Document
from langchain.retrievers import TFIDFRetriever, ContextualCompressionRetriever, SVMRetriever, KNNRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings
from typing import Any, List, Optional
import numpy as np
import akasha.helper as helper


def __get_relevant_doc_knn(db, embeddings, query:str, k:int, relevancy_threshold:float, model, compression:bool):
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
        compressor = LLMChainExtractor.from_llm(model)
        knnc = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=knnR)
        docs = knnc.get_relevant_documents(query)
    else:
        docs = knnR.get_relevant_documents(query)
    
    return docs

def _get_relevant_doc_tfidf(db, query:str, k:int, model, compression:bool)->list:
    """use Term Frequency-Inverse Document Frequency to find relevant doc from query.

    Args:
        **db (Chromadb)**: chroma db\n
        **query (str)**: the query str used to search similar documents\n
        **k (int)**: for each search type, return first k documents\n

    Returns:
        list: list of Documents
    """
    all_docs = db.get(include=['documents','metadatas'])
    docs_list = []
    for i in range(len(all_docs['documents'])):
        docs_list.append(Document(page_content=all_docs['documents'][i],\
                                    metadata=all_docs['metadatas'][i]))
    retriever = TFIDFRetriever.from_documents(docs_list,search_kwargs={"k":k})

    if compression:
        compressor = LLMChainExtractor.from_llm(model)
        tfidfc = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        docs = tfidfc.get_relevant_documents(query)
    else:
        docs = retriever.get_relevant_documents(query)
    return docs[:k]


def _get_relevant_doc_svm(db, embeddings, query:str, k:int, relevancy_threshold:float, model, compression:bool)->list:
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
        compressor = LLMChainExtractor.from_llm(model)
        svmc = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=svmR)
        docs = svmc.get_relevant_documents(query)
    else:
        docs = svmR.get_relevant_documents(query)
    
    return docs
    

def _get_relevant_doc_mmr(db, query:str, k:int, relevancy_threshold:float, model, compression:bool)->list:
    """use Chroma.as_retriever().get_relevant_document() to search relevant documents.

    Args:
        **db (Chromadb)**: chroma db\n
        **query (str)**: the query str used to search similar documents\n
        **k (int)**: for each search type, return first k documents\n
        **relevancy_threshold (float)**: the similarity score threshold to select documents\n

    Returns:
        list: list of selected relevant Documents 
    """
    
    retriever = db.as_retriever(search_type="mmr",\
                search_kwargs={"k": k,'score_threshold': relevancy_threshold})
    
    if compression:
        compressor = LLMChainExtractor.from_llm(model)
        mmrc = ContextualCompressionRetriever(base_compressor = compressor, base_retriever = retriever)
        docs = mmrc.get_relevant_documents(query)
    else:

        docs = retriever.get_relevant_documents(query)
    return docs



def _merge_docs(docs_list:list, topK:int, language:str, verbose:bool, logs:list, max_token:int, model)->(list,int):
    """merge different search types documents, if total len of documents too large,
        will not select all documents.
        use jieba to count length of chinese words, use split space otherwise.

    Args:
        **docs_list (list)**: list of all docs from selected search types\n
        **topK (int)**: for each search type, select topK documents\n
        **language (str)**: 'ch' for chinese, otherwise use split space to count words, default is chinese\n
        **verbose (bool)**: show log texts or not. Defaults to False.\n
        **logs (list)**: list that store logs.\n
        **max_token (int)**: max token size of llm input.\n

    Returns:
        list: merged list of Documents
    """
    res = []
    cur_count = 0
    for i in range(topK):
        for docs in docs_list:
            if i >= len(docs):
                continue
           
            if docs[i] in res:
                continue
            
            words_len = model.get_num_tokens(docs[i].page_content)
            if cur_count + words_len > max_token:
                if verbose:
                    print("words length: ", cur_count)
                logs.append("words length: " + str(cur_count))
                return res, cur_count
            # if language=='ch':
            #     words_len = helper.get_doc_length(language, docs[i])
            #     if cur_count + words_len > 1500:
            #         if verbose:
            #             print("words length: ", cur_count)
            #         logs.append("words length: " + str(cur_count))
            #         return res
            # else:
            #     words_len = helper.get_doc_length(language, docs[i])
            #     if cur_count + words_len > 3000:
            #         if verbose:
            #             print("words length: ", cur_count)
            #         logs.append("words length: " + str(cur_count))
            #         return res
            
            cur_count += words_len
            res.append(docs[i])

    if verbose:
        print("words length: ", cur_count)
    logs.append("words length: " + str(cur_count))

    return res, cur_count



def get_docs(db, embeddings, query:str, topK:int, threshold:float, language:str, search_type:str,
             verbose:bool, logs:list, model, compression:bool, max_token:int)->(list,int):
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
        **logs (list)**: list that store logs.\n
        **model ()**: large language model object\n
        **compression (bool)**: if True, using llm to filter out contents in the relevant documents
            that are not related to the query.    \n

    Returns:
        list: selected list of similar documents.
    """
    search_type = search_type.lower()
    if search_type == 'merge':
        docs_mmr = _get_relevant_doc_mmr(db, query, topK, threshold, model, compression)
        docs_svm = _get_relevant_doc_svm(db, embeddings, query, topK, threshold, model, compression)
        docs_tfidf = _get_relevant_doc_tfidf(db, query, topK, model, compression)
      
        docs, tokens = _merge_docs([docs_tfidf, docs_svm, docs_mmr], topK, language, verbose, logs, max_token, model)
  
    elif search_type == 'mmr':
        docs_mmr = _get_relevant_doc_mmr(db, query, topK, threshold, model, compression)
        docs, tokens = _merge_docs([docs_mmr], topK, language, verbose, logs, max_token, model)
        
    
    elif search_type == 'svm':
        docs_svm = _get_relevant_doc_svm(db, embeddings, query, topK, threshold, model, compression)
        docs, tokens = _merge_docs([docs_svm], topK, language, verbose, logs, max_token, model)
        
    
    elif search_type == 'tfidf':
        docs_tfidf = _get_relevant_doc_tfidf(db, query, topK, model, compression)
        docs, tokens = _merge_docs([docs_tfidf], topK, language, verbose, logs, max_token, model)

    elif search_type == 'knn':
        docs_knn = __get_relevant_doc_knn(db,embeddings, query, topK, threshold, model, compression) 
        docs, tokens = _merge_docs([docs_knn], topK, language, verbose, logs, max_token, model)
    
    else:
        info = f"cannot find search type {search_type}, end process\n"
        print(info)
        logs.append(info)
        return None

    return docs, tokens



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
        cls, db, embeddings: Embeddings, k:int = 3, relevancy_threshold:float=0.2,**kwargs: Any
    ) -> KNNRetriever:
        db_data = db.get(include=['embeddings', 'documents', 'metadatas'])
        index = np.array(db_data['embeddings'])
        texts = db_data['documents']
        metadata = db_data['metadatas']
        return cls(embeddings=embeddings, index=index, texts=texts, metadata=metadata\
                   ,k=k, relevancy_threshold=relevancy_threshold, **kwargs)
    
    def get_relevant_documents(self, query:str) ->List[Document]:
        """general function to retrieve relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]:  relevant documents
        """
        return self._ks(query)
    

    def _ks(self, query:str)-> List[Document]:
        """implement k-means search to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """
        query_embeds = np.array(self.embeddings.embed_query(query))
        # calc L2 norm
        index_embeds = self.index / np.sqrt((self.index**2).sum(1, keepdims=True))
        query_embeds = query_embeds / np.sqrt((query_embeds**2).sum())

        similarities = index_embeds.dot(query_embeds)
        sorted_ix = np.argsort(-similarities)

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities - np.min(similarities)) / denominator

        top_k_results = [
            Document(page_content=self.texts[row], metadata=self.metadata[row])
            for row in sorted_ix[0 : self.k]
            if (
                self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold
            )
        ]
        return top_k_results
    def _aget_relevant_documents(self, query:str)-> List[Document]:
        """implement k-means search to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """
        query_embeds = np.array(self.embeddings.embed_query(query))
        # calc L2 norm
        index_embeds = self.index / np.sqrt((self.index**2).sum(1, keepdims=True))
        query_embeds = query_embeds / np.sqrt((query_embeds**2).sum())

        similarities = index_embeds.dot(query_embeds)
        sorted_ix = np.argsort(-similarities)

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities - np.min(similarities)) / denominator

        top_k_results = [
            Document(page_content=self.texts[row], metadata=self.metadata[row])
            for row in sorted_ix[0 : self.k]
            if (
                self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold
            )
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
        cls , db, embeddings: Embeddings, k:int = 3, relevancy_threshold:float=0.2 ,**kwargs: Any
    ) -> SVMRetriever:
        #index = create_index(texts, embeddings)
        db_data = db.get(include=['embeddings','documents','metadatas'])
        index = db_data['embeddings']
        texts = db_data['documents']
        metadata = db_data['metadatas']
        return cls(embeddings=embeddings, index=index, texts=texts, metadata=metadata\
                   ,k=k, relevancy_threshold=relevancy_threshold, **kwargs)
    
    
    
    def get_relevant_documents(
        self, query: str
    ) -> List[Document]:
        """general function to retrieve relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]:  relevant documents
        """
        return self._gs(query)
    

    
    def _gs(
        self, query: str
    ) -> List[Document]:
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
                "scikit-learn`."
            )
        
        query_embeds = np.array(self.embeddings.embed_query(query))
        x = np.concatenate([query_embeds[None, ...], self.index])
        y = np.zeros(x.shape[0])
        y[0] = 1

        clf = svm.LinearSVC(
            class_weight="balanced", verbose=False, max_iter=10000, tol=1e-6, C=0.1
        )
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
            sorted_ix[0], sorted_ix[zero_index] = sorted_ix[zero_index], sorted_ix[0]

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities - np.min(similarities)) / denominator

        top_k_results = []
        for row in sorted_ix[1 : self.k + 1]:
            if (
                self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold
            ):
                top_k_results.append(Document(page_content=self.texts[row - 1],metadata=self.metadata[row-1]))
        return top_k_results
    
    
    
    async def _aget_relevant_documents(
        self, query: str
    ) -> List[Document]:
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
                "scikit-learn`."
            )

        query_embeds = np.array(self.embeddings.embed_query(query))
        x = np.concatenate([query_embeds[None, ...], self.index])
        y = np.zeros(x.shape[0])
        y[0] = 1

        clf = svm.LinearSVC(
            class_weight="balanced", verbose=False, max_iter=10000, tol=1e-6, C=0.1
        )
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
            sorted_ix[0], sorted_ix[zero_index] = sorted_ix[zero_index], sorted_ix[0]

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities - np.min(similarities)) / denominator

        top_k_results = []
        for row in sorted_ix[1 : self.k + 1]:
            if (
                self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold
            ):
                top_k_results.append(Document(page_content=self.texts[row - 1],metadata=self.metadata[row-1]))
        return top_k_results