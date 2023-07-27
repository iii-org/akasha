import os
from pathlib import Path
from langchain.chains.question_answering import load_qa_chain
import helper
import search



def get_response(doc_path:str, prompt:str = "", embeddings:str = "text-embedding-ada-002"\
                 , model:str = "gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
                 language:str = 'ch' , search_type:str = 'merge' )->str:
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
    logs = []
    embeddings = helper.handle_embeddings(embeddings, logs, verbose)
    model = helper.handle_model(model, logs, verbose)

    print("building chroma db...\n")
    db = helper.create_chromadb(doc_path, logs, verbose, embeddings)

    if db is None:
        info = "document path not exist\n"
        print(info)
        logs.append(info)
        return ""



    docs = search.get_docs(db, embeddings, prompt, topK, threshold, language, search_type, verbose, logs)
    if docs is None:
        return ""
    

    chain = load_qa_chain(llm=model, chain_type="stuff",verbose=False)
    
    if verbose:
        print(docs)
    logs.append("\n\ndocuments: \n\n" + ''.join([doc.page_content for doc in docs]))
    
    res = chain.run(input_documents=docs, question=prompt)
    response = res.split("Finished chain.")

    if verbose:
        print(response)
    logs.append("\n\nresponse:\n\n"+ response[-1])

    helper.save_logs(logs)
    return response[-1]

