


def handle_params(model, embeddings, chunk_size, search_type, topK, threshold, language, compression)->dict:

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
    metrics = {}

    metrics["doc_length"] = doc_length
    metrics["time"] = time

    return metrics

def handle_table(prompt:str, docs:list, response:str)->dict:

    table = {}

    inputs = '\n\n'.join([doc.page_content for doc in docs])
    metadata = '\n\n'.join([doc.metadata['source'] for doc in docs])

    table["prompt"] = prompt
    table["inputs"] = inputs
    table["response"] = response
    table["metadata"] = metadata
    return table