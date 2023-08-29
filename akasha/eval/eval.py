import datetime
import time
from tqdm import tqdm
import akasha
import os
import numpy as np
import torch
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain


def auto_create_questionset(doc_path:str, question_num:int = 10, embeddings:str = "openai:text-embedding-ada-002", chunk_size:int=1000\
                 , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
                 language:str = 'ch' , search_type:str = 'merge', record_exp:str = "", \
                 system_prompt:str = "" ):
    """auto create question set by llm model, each time it will randomly select a range of documents from the documents directory, 
    then use llm model to generate a question and answer pair, and save it into a txt file.

    Args:
        doc_path (str): documents directory path
        question_num (int, optional): number of questions you want to create. Defaults to 10.
        embeddings (str, optional): the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002".
        chunk_size (int, optional): chunk size of texts from documents. Defaults to 1000.
        model (str, optional): llm model to use. Defaults to "gpt-3.5-turbo".
        verbose (bool, optional): show log texts or not. Defaults to False.
        topK (int, optional): search top k number of similar documents. Defaults to 2.
        threshold (float, optional): the similarity threshold of searching. Defaults to 0.2.
        language (str, optional): the language of documents and prompt, use to make sure docs won't exceed
            max token size of llm input.
        search_type (str, optional): search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf'.
        record_exp (str, optional): use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
            record_exp as experiment name.  default "".
        system_prompt (str, optional): the system prompt that you assign special instruction to llm model, so will not be used
        in searching relevant documents. Defaults to "".
    Returns:
        (list, list): question list and answer list
    """
    
    
    ### define variables ###
    doc_range = (1999+chunk_size)//chunk_size   # doc_range is determine by the chunk size, so the select documents won't be too short to having trouble genereating a question
    start_time = time.time()
    logs = ["\n\n-----------------auto_create_questionset----------------------\n"]
    params = akasha.format.handle_params(model, embeddings, chunk_size, search_type, topK, threshold, language, False)
    embeddings_name = embeddings
    embeddings = akasha.helper.handle_embeddings(embeddings, logs, verbose)
    model = akasha.helper.handle_model(model, logs, verbose)
    logs.append(datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S"))
    table = {}
    doc_length = 0
    tokens = 0
    question = []
    answer = []
    
    
    ### load docuemtns from db ###
    db = akasha.helper.create_chromadb(doc_path, logs, verbose, embeddings, embeddings_name, chunk_size)
    if db is None:
        info = "document path not exist\n"
        print(info)
        logs.append(info)
        return ""
    
    
    db_data = db.get(include=['documents','metadatas'])
    texts = db_data['documents']
    metadata =  db_data['metadatas']
    
    
    
    
    ### random select a range of documents from the documents , and use llm model to generate a question and answer pair ###
    for i in range(question_num):
        random_index = np.random.randint(len(texts) - doc_range)
        doc_text = '\n'.join(texts[random_index:random_index + doc_range])
        docs = [Document(page_content=texts[k], metadata=metadata[k]) for k in range(random_index,random_index+doc_range)]
        doc_length += akasha.helper.get_docs_length(language, docs)
        tokens += model.get_num_tokens(doc_text)
        
        q_prompt = akasha.prompts.format_create_question_prompt(doc_text)
        
        
        try:    ### try call openai llm model 
            response = model.predict(q_prompt)
            
        except:
            response = model._call(q_prompt)

        response = akasha.helper.sim_to_trad(response) #transform simplified chinese to traditional chinese
        print(response)
        logs.append(doc_text)
        logs.append("\n\nresponse:\n\n"+ response)
        process = ''.join(response.split("問題：")).split("答案：")
        question.append("問題："+process[0])
        answer.append("答案："+process[1])
        
        new_table = akasha.format.handle_table(question, docs, answer)
        for key in new_table:
            if key not in table:
                table[key] = []
            table[key].append(new_table[key])
        
        
    
    end_time = time.time()

    ### record logs ###
    if record_exp != "":    
        metrics = akasha.format.handle_metrics(doc_length, end_time - start_time)
        params['doc_range'] = doc_range
        metrics['tokens'] = tokens
        akasha.aiido_upload(record_exp, params, metrics, table)
    akasha.helper.save_logs(logs)
    
    
    ### write question and answer into txt file, but first check if "questionset" directory exist or not, it not, first create it.
    ### for filename, count the files in the questionset directory that has doc_path in the file name, and use it as the file name.
    if not os.path.exists("questionset"):
        os.makedirs("questionset")
    count = 0
    suf_path = doc_path.split('/')[-2]
    for filename in os.listdir("questionset"):
        if suf_path in filename:
            count += 1
    file_name = "questionset/"+suf_path+"_"+str(count)+".txt"
    with open(file_name, "w", encoding="utf-8") as f:
        
        for w in range(len(question)):
            f.write(question[w]  + answer[w] + "\n\n")
    
    print("question set saved in ", file_name,"\n\n")
    return question, answer
    
    


def auto_evaluation(questionset_path:str, doc_path:str, embeddings:str = "openai:text-embedding-ada-002", chunk_size:int=1000\
                 , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
                 language:str = 'ch' , search_type:str = 'merge', record_exp:str = "")->(float, float):
    """parse the question set txt file generated from "auto_create_questionset" function, and use llm model to generate response, 
    evaluate the performance of the given paramters based on similarity between responses and the default answers, use bert_score 
    and rouge_l to evaluate the response.

    Args:
        questionset_path (str): the path of question set txt file
        doc_path (str): documents directory path
        embeddings (str, optional): the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002".
        chunk_size (int, optional): chunk size of texts from documents. Defaults to 1000.
        model (str, optional): llm model to use. Defaults to "gpt-3.5-turbo".
        verbose (bool, optional): show log texts or not. Defaults to False.
        topK (int, optional): search top k number of similar documents. Defaults to 2.
        threshold (float, optional): the similarity threshold of searching. Defaults to 0.2.
        language (str, optional): the language of documents and prompt, use to make sure docs won't exceed
            max token size of llm input.
        search_type (str, optional): search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf'.
        record_exp (str, optional): use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
            record_exp as experiment name.  default "".
        
        
    Returns:
        (float, float): average bert_score and average rouge_l score of all questions
    """
    #read the whole txt file and separate it into question and answer list by "\n\n"
    with open(questionset_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.split("\n\n")
    question = []
    answer = []
    bert = []
    rouge = []

    ### parse the questions and answers into question list and answer list ###
    for i in range(len(content)):
        if content[i] == "":
            continue
        process = ''.join(content[i].split("問題：")).split("答案：")
        
        question.append(process[0])
        answer.append(process[1])
    
    
    
    ### if language is "ch", use chinese system prompt ###
    if language=='ch':
        system_prompt = "[INST] <<SYS>> 用中文回答 <</SYS>> [/INST]" 
    else:
        system_prompt = ""
        
    
    ### define variables ###
    total_question = len(question)
    doc_length = 0
    tokens = 0
    start_time = time.time()
    logs = ["\n\n---------------auto_evaluation------------------------\n"]
    table = {}
    params = akasha.format.handle_params(model, embeddings, chunk_size, search_type, topK, threshold, language, False)
    embeddings_name = embeddings
    embeddings = akasha.helper.handle_embeddings(embeddings, logs, verbose)
    model = akasha.helper.handle_model(model, logs, verbose)
    progress = tqdm(total = total_question, desc="Run Auto Evaluation")
    logs.append(datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S"))
    

    
    ### load documents from db ###
    db = akasha.helper.create_chromadb(doc_path, logs, verbose, embeddings, embeddings_name, chunk_size)

    if db is None:
        info = "document path not exist\n"
        print(info)
        logs.append(info)
        return -1.0, -1.0

    
    ### for each question and answer, use llm model to generate response, and evaluate the response by bert_score and rouge_l ###
    for i in range(total_question):
        
        
        progress.update(1)
        docs = akasha.search.get_docs(db, embeddings, question[i], topK, threshold, language, search_type, verbose,\
                     logs, model, False)
        doc_length += akasha.helper.get_docs_length(language, docs)
        tokens += model.get_num_tokens(''.join([doc.page_content for doc in docs]))
        
        

        try:
            chain = load_qa_chain(llm=model, chain_type="stuff",verbose=False)
            response = chain.run(input_documents = docs, question =  system_prompt + question[i])
            response = akasha.helper.sim_to_trad(response)
            response = response.split("Finished chain.")
        except:
            print("running model error\n")
            response = ["running model error"]
            torch.cuda.empty_cache()

        
        bert.append(akasha.eval.scores.get_bert_score(response[-1],answer[i],language))
        rouge.append(akasha.eval.scores.get_rouge_score(response[-1],answer[i],language))
        
        logs.append("\n\ndocuments: \n\n" + ''.join([doc.page_content for doc in docs]))
        logs.append("\n\nresponse:\n\n"+ response[-1])
        
        new_table = akasha.format.handle_table(question[i], docs, response)
        new_table = akasha.format.handle_score_table(new_table, bert[-1], rouge[-1])
        for key in new_table:
            if key not in table:
                table[key] = []
            table[key].append(new_table[key])
                
        
    progress.close()
    end_time = time.time()

    if record_exp != "":    
        metrics = akasha.format.handle_metrics(doc_length, end_time - start_time)
        metrics['avg_bert'] = sum(bert)/len(bert)
        metrics['avg_rouge'] = sum(rouge)/len(rouge)
        metrics['tokens'] = tokens
        akasha.aiido_upload(record_exp, params, metrics, table)
    akasha.helper.save_logs(logs)
    
    
    return metrics['avg_bert'], metrics['avg_rouge']
  

