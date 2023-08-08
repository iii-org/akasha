


def handle_params(model:str, embeddings:str, chunk_size:int, search_type:str, topK:int, threshold:float, language:str, compression:bool)->dict:
    """save running parameters into dictionary in order to parse to aiido

    Args:
        model (str): model name
        embeddings (str): embedding name
        chunk_size (int): chunk size of texts from documents
        search_type (str): search type of finding relevant documents
        topK (int): return top k documents
        threshold (float): only return documents that has similarity score larger than threshold
        language (str): 'ch' for chinese and 'en' for other.
        compression (bool): compress the relevant documents or not.

    Returns:
        dict: parameter dictionary
    """
    params = {}

    params["model"] = model
    params["embeddings"] = embeddings
    params["search_type"] = search_type
    params["topK"] = topK
    params["threshold"] = threshold
    params["language"] = language
    params["compression"] = compression
    params["chunk_size"] = chunk_size
    return params

def handle_metrics(doc_length:int, time:float)->dict:
    """save running metrics into dictionary in order to parse to aiido

    Args:
        doc_length (int): length of texts from relevant documents  
        time (float): total spent time

    Returns:
        dict: metric dictionary
    """
    metrics = {}

    metrics["doc_length"] = doc_length
    metrics["time"] = time

    return metrics

def handle_table(prompt:str, docs:list, response:str)->dict:
    """save running results into dictionary in order to parse to aiido

    Args:
        prompt (str): input query/question
        docs (list): Document list and metadata
        response (str): results from llm response

    Returns:
        dict: table dictionary
    """
    table = {}
    
    inputs = '\n\n'.join([doc.page_content for doc in docs])
    try:
        metadata = '\n\n'.join([doc.metadata['source'] for doc in docs])
    except:
        metadata = "none"
    table["prompt"] = prompt
    table["inputs"] = inputs
    table["response"] = response
    table["metadata"] = metadata
    return table