


def handle_params(model:str, embeddings:str, chunk_size:int, search_type:str, topK:int, threshold:float, language:str, compression:bool)->dict:
    """save running parameters into dictionary in order to parse to aiido

    Args:
        **model (str)**: model name\n
        **embeddings (str)**: embedding name\n
        **chunk_size (int)**: chunk size of texts from documents\n
        **search_type (str)**: search type of finding relevant documents\n
        **topK (int)**: return top k documents\n
        **threshold (float)**: only return documents that has similarity score larger than threshold\n
        **language (str)**: 'ch' for chinese and 'en' for other.\n
        **compression (bool)**: compress the relevant documents or not.\n

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
    if topK != -1:
        params["topK"] = topK
    if threshold != -1.0:
        params["threshold"] = threshold
    if language != "":
        params["language"] = language
    
    params["compression"] = compression
    if chunk_size != -1:
        params["chunk_size"] = chunk_size
    return params

def handle_metrics(doc_length:int, time:float, tokens:int)->dict:
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
    metrics['tokens'] = tokens
    return metrics

def handle_table(prompt:str, docs:list, response:str)->dict:
    """save running results into dictionary in order to parse to aiido

    Args:
        **prompt (str)**: input query/question\n
        **docs (list)**: Document list and metadata\n
        **response (str)**: results from llm response\n

    Returns:
        dict: table dictionary
    """
    table = {}
    table["prompt"] = [prompt]
    table["response"] = [response]
    if len(docs) != 0 :
        
        
        try:
            inputs = '\n\n'.join([doc.page_content for doc in docs])
            metadata = '\n\n'.join([doc.metadata['source'] + "    page: " + str(doc.metadata['page']) for doc in docs])
        except:
            metadata = "none"
            inputs = '\n\n'.join([doc for doc in docs])
        table["inputs"] = [inputs]
        table["metadata"] = [metadata]
        
    else:
        table["inputs"] = ["none"]
        table["metadata"] = ["none"]
    
    return table


def handle_score_table(table:dict, bert:float, rouge:float, llm_score:float)->dict:
    """add each response's bert and rouge score into table dictionary

    Args:
        **table (dict)**: table dictionary that store texts data for a run of experiment.\n
        **bert (float)**: bert score\n
        **rouge (float)**: rouge score\n

    Returns:
        dict: table dictionary
    """
    
    table["bert"] = [bert]
    table["rouge"] = [rouge]
    table["llm_score"] = [llm_score]
    
    return table