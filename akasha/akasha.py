import os
from pathlib import Path
import time
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import akasha.helper as helper
import akasha.search as search
import akasha.format as format
import datetime
from dotenv import load_dotenv
load_dotenv() 

def get_response(doc_path:str, prompt:str = "", embeddings:str = "openai:text-embedding-ada-002"\
                 , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
                 language:str = 'ch' , search_type:str = 'merge', compression:bool = False, record_exp:bool = False \
                      )->str:
    """input the documents directory path and question, will first store the documents
        into vectors db (chromadb), then search similar documents based on the prompt question.
        llm model will use these documents to generate the response of the question.

    Args:
        doc_path (str): documents directory path
        prompt (str, optional): question you want to ask. Defaults to "".
        embeddings (str, optional): the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002"\.
        model (str, optional): llm model to use. Defaults to "gpt-3.5-turbo".
        verbose (bool, optional): show log texts or not. Defaults to False.
        topK (int, optional): search top k number of similar documents. Defaults to 2.
        threshold (float, optional): the similarity threshold of searching. Defaults to 0.2.
        language (str, optional): the language of documents and prompt, use to make sure docs won't exceed
            max token size of llm input.
        search_type (str, optional): search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf'.

    Returns:
        str: llm output str
    """
    start_time = time.time()
    logs = []
    params = format.handle_params(model, embeddings, search_type, topK, threshold, language, compression)
    embeddings_name = embeddings
    embeddings = helper.handle_embeddings(embeddings, logs, verbose)
    model = helper.handle_model(model, logs, verbose)
    logs.append(datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S"))
    

    print("building chroma db...\n")
    db = helper.create_chromadb(doc_path, logs, verbose, embeddings, embeddings_name, model)

    if db is None:
        info = "document path not exist\n"
        print(info)
        logs.append(info)
        return ""


    
    docs = search.get_docs(db, embeddings, prompt, topK, threshold, language, search_type, verbose,\
                     logs, model, compression)
    if docs is None:
        return ""
    
    doc_length = helper.get_docs_length(language, docs)
    
    chain = load_qa_chain(llm=model, chain_type="stuff",verbose=False)
    if verbose:
        print(docs)
    logs.append("\n\ndocuments: \n\n" + ''.join([doc.page_content for doc in docs]))
    
    res = chain.run(input_documents=docs, question=prompt)
    response = res.split("Finished chain.")
    
    
    if verbose:
        print(response)
    logs.append("\n\nresponse:\n\n"+ response[-1])
    
    end_time = time.time()
    if record_exp:    
        metrics = format.handle_metrics(doc_length, end_time - start_time)
        table = format.handle_table(prompt, docs, response)
        aiido_upload("exp_akasha_gr", params, metrics, table)
    helper.save_logs(logs)
    return response[-1]









def chain_of_thought(doc_path:str, prompt:list, embeddings:str = "openai:text-embedding-ada-002"\
                 , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
                 language:str = 'ch' , search_type:str = 'merge',  compression:bool = False, record_exp:bool = False )->str:
    """input the documents directory path and question, will first store the documents
        into vectors db (chromadb), then search similar documents based on the prompt question.
        llm model will use these documents to generate the response of the question.

    Args:
        doc_path (str): documents directory path
        prompt (str, optional): question you want to ask. Defaults to "".
        embeddings (str, optional): the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002"\.
        model (str, optional): llm model to use. Defaults to "gpt-3.5-turbo".
        verbose (bool, optional): show log texts or not. Defaults to False.
        topK (int, optional): search top k number of similar documents. Defaults to 2.
        threshold (float, optional): the similarity threshold of searching. Defaults to 0.2.
        language (str, optional): the language of documents and prompt, use to make sure docs won't exceed
            max token size of llm input.
        search_type (str, optional): search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf'.

    Returns:
        str: llm output str
    """
    start_time = time.time()
    logs = []
    params = format.handle_params(model, embeddings, search_type, topK, threshold, language, compression)
    embeddings_name = embeddings
    embeddings = helper.handle_embeddings(embeddings, logs, verbose)
    model = helper.handle_model(model, logs, verbose)
    logs.append(datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S"))
    

    print("building chroma db...\n")
    db = helper.create_chromadb(doc_path, logs, verbose, embeddings, embeddings_name, model)

    if db is None:
        info = "document path not exist\n"
        print(info)
        logs.append(info)
        return ""



    docs = search.get_docs(db, embeddings, prompt[0], topK, threshold, language, search_type,\
                            verbose, logs, model, compression)
    if docs is None:
        return ""
    
    doc_length = helper.get_docs_length(language, docs)
    chain = load_qa_chain(llm=model, chain_type="stuff",verbose=False)
    if verbose:
        print(docs)
    logs.append("\n\ndocuments: \n\n" + ''.join([doc.page_content for doc in docs]))
   
    
    
    for i in range(len(prompt)):

        res = chain.run(input_documents=docs, question=prompt[i])
        response = res.split("Finished chain.")
        print(response)
        logs.append("\n\nresponse:\n\n"+ response[-1])
        docs = [Document(page_content=''.join(response))]
        print(docs)
    

    end_time = time.time()    
    if record_exp:    
        metrics = format.handle_metrics(doc_length, end_time - start_time)
        table = format.handle_table(prompt, docs, response)
        aiido_upload("exp_akasha_cot", params, metrics, table)
    helper.save_logs(logs)
    return response[-1]




def aiido_upload(exp_name, params:dict={}, metrics:dict={}, table:dict={}):
    import aiido
    mod = params["model"].split(':')[0]
    emb = params["embeddings"].split(':')[0]
    sea = params["search_type"]
    aiido.init(experiment=exp_name, run = mod+'-'+emb+'-'+sea)
    aiido.log_params_and_metrics(params=params, metrics=metrics)


    if len(table) > 0:
        import mlflow
        mlflow.log_table(table,"table.json")
    aiido.end_run()
    return