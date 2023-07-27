from langchain.schema import Document
from langchain.retrievers import TFIDFRetriever

import numpy as np
from sklearn import svm
import jieba


def _get_relevant_doc_tfidf(db, query:str, k:int):
    """use Term Frequency-Inverse Document Frequency to find relevant doc from query.

    Args:
        db (_type_): chromadb
        query (_type_): query str
        k (int, optional): return top k docs
        relevany_threshold (float, optional): similarity threshold to select the doc
        not used in tfidf method. Defaults to 0.2.

    Returns:
        list: list of Documents
    """
    all_docs = db.get(include=['documents','metadatas'])
    docs_list = []
    for i in range(len(all_docs['documents'])):
        docs_list.append(Document(page_content=all_docs['documents'][i],\
                                    metadata=all_docs['metadatas'][i]))
    retriever = TFIDFRetriever.from_documents(docs_list,search_kwargs={"k":k})
    result = retriever.get_relevant_documents(query)
    return result[:k]


def _get_relevant_doc_svm(db, embeddings, query:str, k:int, relevancy_threshold:float):
    """use SVM to find relevant doc from query.

    Args:
        db (_type_): chromadb
        query (_type_): _description_
        embeddings (_type_): _description_
        k (int, optional): _description_. Defaults to 2.
        relevancy_threshold (float, optional): _description_. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    indx =db.get(include=['embeddings','documents','metadatas'])
    #print(indx['documents'])

    query_embeds = np.array(embeddings.embed_query(query))
    x = np.concatenate([query_embeds[None, ...], indx['embeddings']])
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
    for row in sorted_ix[1 : k + 1]:
        if (
            relevancy_threshold is None
            or normalized_similarities[row] >= relevancy_threshold
        ):
            top_k_results.append(Document(page_content=indx['documents'][row-1],\
                                    metadata=indx['metadatas'][row-1]))
    return top_k_results


def _get_relevant_doc_mmr(db, query:str, k:int, relevancy_threshold:float):
    retriever = db.as_retriever(search_type="similarity_score_threshold",\
                search_kwargs={"k": k,'score_threshold': relevancy_threshold})
    

    docs = retriever.get_relevant_documents(query)
    return docs



def _merge_docs(docs_list:list, topK:int, language:str, verbose:bool, logs:list):
    res = []
    cur_count = 0
    for i in range(topK):
        for docs in docs_list:
            if i >= len(docs):
                continue
           
            if docs[i] in res:
                continue
            
            
            if language=='ch':
                words_len = len(list(jieba.cut(docs[i].page_content)))
                if cur_count + words_len > 1800:
                    if verbose:
                        print("words length: ", cur_count)
                    logs.append("words length: " + str(cur_count))
                    return res
            else:
                words_len = len(docs[i].page_content.split())
                if cur_count + words_len > 3000:
                    if verbose:
                        print("words length: ", cur_count)
                    logs.append("words length: " + str(cur_count))
                    return res
            
            cur_count += words_len
            res.append(docs[i])

    if verbose:
        print("words length: ", cur_count)
    logs.append("words length: " + str(cur_count))

    return res



def get_docs(db, embeddings, query:str, topK:int, threshold:float, language:str, search_type:str,
             verbose:bool, logs:list):
    """search docs based on given search_type, default is merge, which contain 'mmr', 'svm', 'tfidf'
        and merge them together. 

    Args:
        db (_type_): _description_
        embeddings (_type_): _description_
        query (str): _description_
        topK (int): _description_
        threshold (float): _description_
        language (str): _description_
        search_type (str): _description_
        verbose (bool): _description_
        logs (list): _description_

    Returns:
        _type_: _description_
    """

    if search_type == 'merge':
        docs_mmr = _get_relevant_doc_mmr(db, query, topK, threshold)
        docs_svm = _get_relevant_doc_svm(db, embeddings, query, topK, threshold)
        docs_tfidf = _get_relevant_doc_tfidf(db, query, topK)
      
        docs = _merge_docs([docs_mmr, docs_svm, docs_tfidf], topK, language, verbose, logs)
  
    elif search_type == 'mmr':
        docs_mmr = _get_relevant_doc_mmr(db, query, topK, threshold)
        docs = _merge_docs([docs_mmr], topK, language, verbose, logs)
        
    
    elif search_type == 'svm':
        docs_svm = _get_relevant_doc_svm(db, embeddings, query, topK, threshold)
        docs = _merge_docs([docs_svm], topK, language, verbose, logs)
        
    
    elif search_type == 'tfidf':
        docs_tfidf = _get_relevant_doc_tfidf(db, query, topK)
        docs = _merge_docs([docs_tfidf], topK, language, verbose, logs)
        
    
    else:
        info = f"cannot find search type {search_type}, end process\n"
        print(info)
        logs.append(info)
        return None

    return docs