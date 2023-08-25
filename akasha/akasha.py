import os,sys
import numpy as np
import time
from tqdm import tqdm
import torch
from langchain.chains.question_answering import load_qa_chain, LLMChain
from langchain import PromptTemplate
from langchain.schema import Document
import akasha.helper as helper
import akasha.search as search
import akasha.format as format
import akasha.prompts as prompts
import akasha.eval as eval
import datetime
from dotenv import load_dotenv
load_dotenv(sys.path[1]) 
print(sys.path[1])
def get_response(doc_path:str, prompt:str = "", embeddings:str = "openai:text-embedding-ada-002", chunk_size:int=1000\
                 , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
                 language:str = 'ch' , search_type:str = 'merge', compression:bool = False, record_exp:str = "", \
                 system_prompt:str = ""  )->str:
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
        compression (bool): compress the relevant documents or not.
        record_exp (str, optional): use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
            record_exp as experiment name.  default ''.

    Returns:
        str: llm output str
    """
    start_time = time.time()
    logs = ["\n\n-----------------get_response----------------------\n"]
    params = format.handle_params(model, embeddings, chunk_size, search_type, topK, threshold, language, compression)
    embeddings_name = embeddings
    embeddings = helper.handle_embeddings(embeddings, logs, verbose)
    model = helper.handle_model(model, logs, verbose)
    logs.append(datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S"))
    

    db = helper.create_chromadb(doc_path, logs, verbose, embeddings, embeddings_name, chunk_size)

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
    
    res = chain.run(input_documents=docs, question=system_prompt + prompt)
    res =  helper.sim_to_trad(res)
    response = res.split("Finished chain.")
    
    
    if verbose:
        print(response)
    logs.append("\n\nresponse:\n\n"+ response[-1])
    
    end_time = time.time()
    if record_exp != "":    
        metrics = format.handle_metrics(doc_length, end_time - start_time)
        table = format.handle_table(prompt, docs, response)
        model.get_num_tokens(''.join([doc.page_content for doc in docs]))
        aiido_upload(record_exp, params, metrics, table)
    helper.save_logs(logs)
    return response[-1]









def chain_of_thought(doc_path:str, prompt:list, embeddings:str = "openai:text-embedding-ada-002", chunk_size:int=1000\
                 , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
                 language:str = 'ch' , search_type:str = 'merge',  compression:bool = False, record_exp:str = "", system_prompt:str="" )->str:
    """input the documents directory path and question, will first store the documents
        into vectors db (chromadb), then search similar documents based on the prompt question.
        llm model will use these documents to generate the response of the question.

        In chain_of_thought function, you can separate your question into multiple small steps so that llm can have better response.
        chain_of_thought function will use all responses from the previous prompts, and combine the documents search from current prompt to generate
        response.
        

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
        compression (bool): compress the relevant documents or not.
        record_exp (str, optional): use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
            record_exp as experiment name.  default ''.

    Returns:
        str: llm output str
    """
    start_time = time.time()
    logs = ["\n\n---------------chain_of_thought------------------------\n"]
    params = format.handle_params(model, embeddings, chunk_size, search_type, topK, threshold, language, compression)
    embeddings_name = embeddings
    embeddings = helper.handle_embeddings(embeddings, logs, verbose)
    model = helper.handle_model(model, logs, verbose)
    logs.append(datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S"))
    

    db = helper.create_chromadb(doc_path, logs, verbose, embeddings, embeddings_name, chunk_size)

    if db is None:
        info = "document path not exist\n"
        print(info)
        logs.append(info)
        return ""

    chain = load_qa_chain(llm=model, chain_type="stuff",verbose=False)

    
    ori_docs = []
    doc_length = 0
    pre_result = []
    for i in range(len(prompt)):

        docs = search.get_docs(db, embeddings, prompt[i], topK, threshold, language, search_type,\
                            verbose, logs, model, compression)
        
        doc_length += helper.get_docs_length(language, docs)
        ori_docs.extend(docs)
        if verbose:
            print(docs)
        logs.append("\n\ndocuments: \n\n" + ''.join([doc.page_content for doc in docs]))



        res = chain.run(input_documents=docs + pre_result, question=system_prompt + prompt[i])
        res = helper.sim_to_trad(res)
        response = res.split("Finished chain.")
        print(response)

        logs.append("\n\nresponse:\n\n"+ response[-1])
        pre_result.append(Document(page_content=''.join(response)))
        
    

    end_time = time.time()    
    if record_exp != "":    
        metrics = format.handle_metrics(doc_length, end_time - start_time)
        table = format.handle_table('\n\n'.join([p for p in prompt]), ori_docs, response)
        aiido_upload(record_exp, params, metrics, table)
    helper.save_logs(logs)
    return response[-1]




def aiido_upload(exp_name, params:dict={}, metrics:dict={}, table:dict={}):
    """upload params_metrics, table to mlflow server for tracking.

    Args:
        exp_name (str): experiment name on the tracking server, if not found, will create one .
        params (dict, optional): parameters dictionary. Defaults to {}.
        metrics (dict, optional): metrics dictionary. Defaults to {}.
        table (dict, optional): table dictionary, used to compare text context between different runs in the experiment. Defaults to {}.
    """
    import aiido
    mod = params["model"].split(':')
    emb = params["embeddings"].split(':')[0]
    sea = params["search_type"]
    aiido.init(experiment=exp_name, run = emb + '-' + sea + '-' + '-'.join(mod))
    aiido.log_params_and_metrics(params=params, metrics=metrics)


    if len(table) > 0:
        import mlflow
        mlflow.log_table(table,"table.json")
    aiido.end_run()
    return




def test_performance(q_file:str, doc_path:str, embeddings:str = "openai:text-embedding-ada-002", chunk_size:int=1000\
                 , model:str = "openai:gpt-3.5-turbo", topK:int = 2, threshold:float = 0.2,\
                 language:str = 'ch' , search_type:str = 'merge', compression:bool = False, record_exp:str = "" ):
    """input a txt file includes list of single choice questions and the answer, will test all of the questions and return the
    correct rate of this parameters(model, embeddings, search_type, chunk_size)
    the format of q_file(.txt) should be one line one question, and the possibles answers and questions are separate by space,
    the last one is which possisble answers is the correct answer, for example, the file should look like: 
        "What is the capital of Taiwan?" Taipei  Kaohsiung  Taichung  Tainan     1
        何者是台灣的首都?  台北 高雄 台中 台南   1
    
    Args:
        q_file (str): the file path of the question file
        doc_path (str): documents directory path
        embeddings (str, optional): the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002"\.
        model (str, optional): llm model to use. Defaults to "gpt-3.5-turbo".
        topK (int, optional): search top k number of similar documents. Defaults to 2.
        threshold (float, optional): the similarity threshold of searching. Defaults to 0.2.
        language (str, optional): the language of documents and prompt, use to make sure docs won't exceed
            max token size of llm input.
        search_type (str, optional): search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf', 'knn'.
        compression (bool): compress the relevant documents or not.
        record_exp (str, optional): use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
            record_exp as experiment name.  default ''.

    Returns:
        float: the correct rate of all questions
    """
    query_list = helper.get_question_from_file(q_file)
    correct_count = 0
    total_question = len(query_list)
    doc_length = 0
    tokens = 0
    verbose = False
    start_time = time.time()
    logs = ["\n\n---------------test_performance------------------------\n"]
    table = {}
    params = format.handle_params(model, embeddings, chunk_size, search_type, topK, threshold, language, compression)
    embeddings_name = embeddings
    embeddings = helper.handle_embeddings(embeddings, logs, verbose)
    model = helper.handle_model(model, logs, verbose)
    logs.append(datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S"))
    

    db = helper.create_chromadb(doc_path, logs, verbose, embeddings, embeddings_name, chunk_size)

    if db is None:
        info = "document path not exist\n"
        logs.append(info)
        return ""


    for question in tqdm(query_list,total=total_question,desc="Run Question Set"):
        
        query, ans = prompts.format_question_query(question)

        docs = search.get_docs(db, embeddings, query, topK, threshold, language, search_type, verbose,\
                     logs, model, compression)
        doc_length += helper.get_docs_length(language, docs)
        tokens += model.get_num_tokens(''.join([doc.page_content for doc in docs]))
        query_with_prompt = prompts.format_llama_json(query)
        

        try:
            chain = load_qa_chain(llm=model, chain_type="stuff",verbose=False)
            response = chain.run(input_documents = docs, question = query_with_prompt)
            response = helper.sim_to_trad(response)
            response = response.split("Finished chain.")
        except:
            print("running model error\n")
            response = ["running model error"]
            torch.cuda.empty_cache()

        logs.append("\n\ndocuments: \n\n" + ''.join([doc.page_content for doc in docs]))
        logs.append("\n\nresponse:\n\n"+ response[-1])
        
        new_table = format.handle_table(query, docs, response)
        for key in new_table:
            if key not in table:
                table[key] = []
            table[key].append(new_table[key])
                

        result = helper.extract_result(response)

        if str(result) == str(ans):
            correct_count += 1
    end_time = time.time()

    if record_exp != "":    
        metrics = format.handle_metrics(doc_length, end_time - start_time)
        metrics['correct_rate'] = correct_count/total_question
        metrics['tokens'] = tokens
        aiido_upload(record_exp, params, metrics, table)
    helper.save_logs(logs)

    return correct_count/total_question , tokens



def detect_exploitation(texts:str, model:str = "openai:gpt-3.5-turbo", verbose:bool = False, record_exp:str = ""):
    """ check the given texts have harmful or sensitive information

    Args:
        texts (str): texts that we want llm to check.
        model (str, optional): llm model name. Defaults to "openai:gpt-3.5-turbo".
        verbose (bool, optional): show log texts or not. Defaults to False.
        record_exp (str, optional): use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
            record_exp as experiment name.  default ''.

    Returns:
        str: response from llm
    """

    logs = []
    model = helper.handle_model(model, logs, verbose)
    sys_b, sys_e = "<<SYS>>\n", "\n<</SYS>>\n\n"
    system_prompt = "[INST]" + sys_b +\
    "check if below texts have any of Ethical Concerns, discrimination, hate speech, "+\
    "illegal information, harmful content, Offensive Language, or encourages users to share or access copyrighted materials"+\
    " And return true or false. Texts are: " + sys_e  + "[/INST]"
       
    template = system_prompt+""" 
    
    Texts: {texts}
    Answer: """
    prompt = PromptTemplate(template=template, input_variables={"texts"})
    response = LLMChain(prompt=prompt,llm=model).run(texts)
    print(response)
    return response





def optimum_combination(q_file:str, doc_path:str, embeddings_list:list = ["openai:text-embedding-ada-002"], chunk_size_list:list=[500]\
                 , model_list:list = ["openai:gpt-3.5-turbo"], topK_list:list = [2], threshold:float = 0.2,\
                 language:str = 'ch' , search_type_list:list = ['merge','svm','tfidf','mmr'], compression:bool = False, record_exp:str = "" ):
    """test all combinations of giving lists, and run test_performance to find parameters of the best result.

    Args:
        q_file (str): the file path of the question file
        doc_path (str): documents directory path
        embeddings_list (_type_, optional): list of embeddings models. Defaults to ["openai:text-embedding-ada-002"].
        chunk_size_list (list, optional): list of chunk sizes. Defaults to [500]\.
        model_list (_type_, optional): list of models. Defaults to ["openai:gpt-3.5-turbo"].
        topK_list (list, optional): list of topK. Defaults to [2].
        threshold (float, optional): the similarity threshold of searching. Defaults to 0.2.
        search_type_list (list, optional): list of search types, currently have "merge", "svm", "knn", "tfidf", "mmr". Defaults to ['merge','svm','tfidf','mmr'].
        compression (bool): compress the relevant documents or not.
        record_exp (str, optional): use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
            record_exp as experiment name.  default ''.
    Returns:
        (list,list): return best score combination and best cost-effective combination
    """
    logs = ["\n\n----------------optimum_combination-----------------------\n"]
    start_time = time.time()
    combinations = helper.get_all_combine(embeddings_list, chunk_size_list, model_list, topK_list, search_type_list)
    progress = tqdm(len(combinations),total = len(combinations), desc="RUN LLM")
    print("\n\ntotal combinations: ", len(combinations))
    result_list = []
    bcr = 0.0
    for embed, chk, mod, tK, st in combinations:
        progress.update(1)
        cur_correct_rate, tokens = test_performance(q_file, doc_path, embeddings=embed, chunk_size=chk, model=mod, topK=tK, threshold=threshold,\
                            language=language, search_type=st, compression=compression, record_exp=record_exp) 
        bcr = max(bcr,cur_correct_rate)
        cur_tup = (cur_correct_rate, cur_correct_rate/tokens, embed, chk, mod, tK, st)
        result_list.append(cur_tup)
        
    progress.close()


    ### record logs ###
    print("Best correct rate: ", "{:.3f}".format(bcr))
    score_comb = "Best score combination: \n"
    print(score_comb)
    logs.append(score_comb)
    bs_combination = helper.get_best_combination(result_list, 0,logs)


    print("\n\n")
    cost_comb = "Best cost-effective: \n"
    print(cost_comb)
    logs.append(cost_comb)
    bc_combination = helper.get_best_combination(result_list, 1,logs)



   
    end_time = time.time()
    format_time = "time spend: " + "{:.3f}".format(end_time-start_time)
    print( format_time )
    logs.append( format_time )
    helper.save_logs(logs)
    return bs_combination, bc_combination





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
    params = format.handle_params(model, embeddings, chunk_size, search_type, topK, threshold, language, False)
    embeddings_name = embeddings
    embeddings = helper.handle_embeddings(embeddings, logs, verbose)
    model = helper.handle_model(model, logs, verbose)
    logs.append(datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S"))
    table = {}
    doc_length = 0
    tokens = 0
    question = []
    answer = []
    
    
    ### load docuemtns from db ###
    db = helper.create_chromadb(doc_path, logs, verbose, embeddings, embeddings_name, chunk_size)

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
        doc_length += helper.get_docs_length(language, docs)
        tokens += model.get_num_tokens(doc_text)
        
        q_prompt = prompts.format_create_question_prompt(doc_text)
        
        
        try:    ### try call openai llm model 
            response = model.predict(q_prompt)
            
        except:
            response = model._call(q_prompt)

        response = helper.sim_to_trad(response) #transform simplified chinese to traditional chinese
        print(response)
        logs.append(doc_text)
        logs.append("\n\nresponse:\n\n"+ response)
        process = ''.join(response.split("問題：")).split("答案：")
        question.append("問題："+process[0])
        answer.append("答案："+process[1])
        
        new_table = format.handle_table(question, docs, answer)
        for key in new_table:
            if key not in table:
                table[key] = []
            table[key].append(new_table[key])
        
        
    
    end_time = time.time()

    ### record logs ###
    if record_exp != "":    
        metrics = format.handle_metrics(doc_length, end_time - start_time)
        params['doc_range'] = doc_range
        metrics['tokens'] = tokens
        aiido_upload(record_exp, params, metrics, table)
    helper.save_logs(logs)
    
    
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
    params = format.handle_params(model, embeddings, chunk_size, search_type, topK, threshold, language, False)
    embeddings_name = embeddings
    embeddings = helper.handle_embeddings(embeddings, logs, verbose)
    model = helper.handle_model(model, logs, verbose)
    progress = tqdm(total = total_question, desc="Run Auto Evaluation")
    logs.append(datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S"))
    

    
    ### load documents from db ###
    db = helper.create_chromadb(doc_path, logs, verbose, embeddings, embeddings_name, chunk_size)

    if db is None:
        info = "document path not exist\n"
        print(info)
        logs.append(info)
        return -1.0, -1.0

    
    ### for each question and answer, use llm model to generate response, and evaluate the response by bert_score and rouge_l ###
    for i in range(total_question):
        
        
        progress.update(1)
        docs = search.get_docs(db, embeddings, question[i], topK, threshold, language, search_type, verbose,\
                     logs, model, False)
        doc_length += helper.get_docs_length(language, docs)
        tokens += model.get_num_tokens(''.join([doc.page_content for doc in docs]))
        
        

        try:
            chain = load_qa_chain(llm=model, chain_type="stuff",verbose=False)
            response = chain.run(input_documents = docs, question =  system_prompt + question[i])
            response = helper.sim_to_trad(response)
            response = response.split("Finished chain.")
        except:
            print("running model error\n")
            response = ["running model error"]
            torch.cuda.empty_cache()

        
        bert.append(eval.get_bert_score(response[-1],answer[i],language))
        rouge.append(eval.get_rouge_score(response[-1],answer[i],language))
        
        logs.append("\n\ndocuments: \n\n" + ''.join([doc.page_content for doc in docs]))
        logs.append("\n\nresponse:\n\n"+ response[-1])
        
        new_table = format.handle_table(question[i], docs, response)
        new_table = format.handle_score_table(new_table, bert[-1], rouge[-1])
        for key in new_table:
            if key not in table:
                table[key] = []
            table[key].append(new_table[key])
                
        
    progress.close()
    end_time = time.time()

    if record_exp != "":    
        metrics = format.handle_metrics(doc_length, end_time - start_time)
        metrics['avg_bert'] = sum(bert)/len(bert)
        metrics['avg_rouge'] = sum(rouge)/len(rouge)
        metrics['tokens'] = tokens
        aiido_upload(record_exp, params, metrics, table)
    helper.save_logs(logs)
    
    
    return metrics['avg_bert'], metrics['avg_rouge']
  

