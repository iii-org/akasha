from .retrievers.retri_rerank import rerank_reduce


def get_relevant_doc_auto(
    retriver_list: list,
    query: str,
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

    # mmrR = retriver_list[0]
    # docs_mmr, mmr_scores = mmrR._gs(query)
    # print("MMR: ", mmr_scores, len(mmr_scores), "\n\n")
    # print(docs_mmr[:4])

    ### svm ###
    # svmR = retriver_list[0]
    # docs_svm, svm_scores = svmR._gs(query)
    # print("SVM: ", svm_scores, docs_svm[0], "\n\n")

    # ### tfidf ###

    # tfretriever = retriver_list[1]
    # docs_tf, tf_scores = tfretriever._gs(query)
    # print("TFIDF", tf_scores, docs_tf[0], "\n\n")

    # ### knn ###
    knnR = retriver_list[0]
    docs_knn, knn_scores = knnR._gs(query)
    # print("KNN: ", knn_scores, len(knn_scores), "\n\n")

    ### bm25 ###
    bm25R = retriver_list[1]
    docs_bm25, bm25_scores = bm25R._gs(query)
    # print("BM25: ", bm25_scores[:10], len(bm25_scores), "\n\n")

    ### decide which to use ###
    backup_docs = []
    final_docs = []  # docs_mmr[0]
    del bm25R
    ## backup_docs is all documents from docs_svm that svm_scores>0.2 ##
    low = 0
    for i in range(len(knn_scores)):
        if knn_scores[i] >= 0.95:
            backup_docs.append(docs_knn[i])
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
    final_docs.extend(docs_knn[low:])
    return final_docs


def get_relevant_doc_auto_rerank(
    retriver_list: list,
    query: str,
    k: int,
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

    ### knn ###
    knnR = retriver_list[0]
    docs_knn, knn_scores = knnR._gs(query)
    pr = max(int(0.1 * len(docs_knn)), 10)
    # print("KNN: ", knn_scores, docs_knn[0], "\n\n")

    ### bm25 ###
    bm25R = retriver_list[1]
    docs_bm25, bm25_scores = bm25R._gs(query)
    # print("BM25: ", bm25_scores[:10], len(bm25_scores), "\n\n")

    ### decide which to use ###
    backup_docs = []
    final_docs = []  # docs_mmr[0]
    del knnR, bm25R
    ## backup_docs is all documents from docs_svm that svm_scores>0.2 ##

    for i in range(len(knn_scores)):
        if knn_scores[i] >= 0.9:
            backup_docs.append(docs_knn[i])
        else:
            break

    if bm25_scores[0] >= 70:
        # if verbose:
        #     print("<<search>>go to bm25\n\n")

        ## find out the idx that the sorted tf_scores is not 0
        idx = 0
        for i in range(len(bm25_scores)):
            if bm25_scores[i] < 70 or i >= 2:
                idx = i
                break
        final_docs.extend(docs_bm25[:idx])

    if len(backup_docs) < pr:
        # if verbose:
        #     print("<<search>>go to knn\n\n")

        final_docs.extend(backup_docs)

    else:
        # if verbose:
        #     print("<<search>>go to knn+rerank\n\n")
        final_docs.extend(rerank_reduce(query, backup_docs, k))

    return final_docs
