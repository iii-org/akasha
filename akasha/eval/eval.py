import datetime
import time
from tqdm import tqdm
import akasha
import akasha.eval as eval
import os
import numpy as np
import torch
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain




def  _generate_single_choice_question(doc_text:str, question:str, cor_ans:str, model, system_prompt:str, choice_num:int)->str:
    """Based on gernerated question and answer, generate wrong answers for single choice question

    Args:
        **doc_text (str)**: the document text that used to generate question and answer\n
        **question (str)**: question generated from previous step\n
        **cor_ans (str)**: correct answer genereated from previous step\n
        **model (var)**: llm model\n
        **system_prompt (str)**: the system prompt that you assign special instruction to llm model, currently not be used in the function\n
        **choice_num (int)**: the number of options for each single choice question\n

    Raises:
        Exception: if the format of the response is not correct, raise exception

    Returns:
        str: choice_num of wrong answers and a correct answer, and the index of correct answer, separated by "\t"
    """
    res = ""
    count = 0
    random_index = np.random.randint(choice_num)
    q_prompt = akasha.prompts.format_wrong_answer(choice_num-1, doc_text, question, cor_ans)
    response = akasha.helper.call_model(model, system_prompt + q_prompt)
    response = akasha.helper.sim_to_trad(response) #transform simplified chinese to traditional chinese
    
    ### separate the response into wrong answers ###
    try:
        process = response.split("錯誤答案：")
        process = process[1:]
        if len(process)!= choice_num-1:
            raise Exception("Answer Format Error")
    except:
        process = response.split("：")[1:]
    
    
    ### combine the wrong answers and correct answer into a single choice question ###
    for wrong_ans in process:
        if wrong_ans == "":
            continue
        elif count == random_index:
            res += '\t' + cor_ans.replace('\n','')
            count += 1
        
        wrong_ans = wrong_ans.replace('\n','').replace('錯誤答案','')
        res += '\t' + wrong_ans
        count += 1
    
    if count < choice_num :
        res += '\t' + cor_ans.replace('\n','')
    
    res += '\t' + str(random_index+1)
    
    return res


def auto_create_questionset(doc_path:str, question_num:int = 10, embeddings:str = "openai:text-embedding-ada-002", chunk_size:int=1000\
                 , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
                 language:str = 'ch' , search_type:str = 'merge', record_exp:str = "", \
                 system_prompt:str = "", question_type:str="essay",choice_num:int = 4 ):
    """auto create question set by llm model, each time it will randomly select a range of documents from the documents directory, 
    then use llm model to generate a question and answer pair, and save it into a txt file.

    Args:
        **doc_path (str)**: documents directory path\n
        **question_num (int, optional)**: number of questions you want to create. Defaults to 10.\n
        **embeddings (str, optional)**: the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002".\n
        **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.\n
        **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
        **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
        **topK (int, optional)**: search top k number of similar documents. Defaults to 2.\n
        **threshold (float, optional)**: the similarity threshold of searching. Defaults to 0.2.\n
        **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
            max token size of llm input.\n
        **search_type (str, optional)**: search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf'.\n
        **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
            record_exp as experiment name.  default "".\n
        **system_prompt (str, optional)**: the system prompt that you assign special instruction to llm model, so will not be used
        in searching relevant documents. Defaults to "".\n
        **question_type (str, optional)**: the type of question you want to generate, "essay" or "single_choice". Defaults to "essay".\n
        **choice_num (int, optional)**: the number of choices for each single choice question, only use it if question_type is "single_choice".
        Defaults to 4.\n
    Returns:
        (list, list): question list and answer list
    """
    
    
    ### define variables ###
    doc_range = (1999+chunk_size)//chunk_size   # doc_range is determine by the chunk size, so the select documents won't be too short to having trouble genereating a question
    vis_doc_range = set()
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
    
    
    progress = tqdm(total = question_num, desc=f"Create Q({question_type})")
    regenerate_limit = question_num
    ### random select a range of documents from the documents , and use llm model to generate a question and answer pair ###
    for i in range(question_num):
        progress.update(1)
        
        random_index = akasha.helper.get_non_repeat_rand_int(vis_doc_range, len(texts) - doc_range)
        
        doc_text = '\n'.join(texts[random_index:random_index + doc_range])
        docs = [Document(page_content=texts[k], metadata=metadata[k]) for k in range(random_index,random_index+doc_range)]
        doc_length += akasha.helper.get_docs_length(language, docs)
        tokens += model.get_num_tokens(doc_text)
        
        try:
            q_prompt = akasha.prompts.format_create_question_prompt(doc_text, question_type)
            
            response = akasha.helper.call_model(model, q_prompt)       
            response = akasha.helper.sim_to_trad(response) #transform simplified chinese to traditional chinese
            
            process = ''.join(response.split("問題：")).split("答案：")
            if len(process) < 2:
                raise Exception("Question Format Error")
        except:
            if regenerate_limit > 0:
                regenerate_limit -=1
                i-=1
                progress.update(-1)
                print("Question Format Error while generating questions. Regenerate\n")
                continue
            else:
                print("Question Format Error while generating questions. Stop\n")
                break 
        
        question.append("問題："+process[0])
        
        
        if question_type == "essay":
            answer.append("答案："+process[1])
            
        else:
            anss = _generate_single_choice_question(doc_text, process[0], process[1], model, system_prompt, choice_num)
            answer.append(anss)
            response = process[0] + "\n"  + "選項:\n" + anss + "\n\n"
        
        
        if verbose:
            print(response)
        logs.append(doc_text)
        logs.append("\n\nresponse:\n\n"+ response)
        new_table = akasha.format.handle_table(question[-1], docs, answer[-1])
        for key in new_table:
            if key not in table:
                table[key] = []
            table[key].append(new_table[key])
        
    progress.close() #end running llm progress bar
    
    end_time = time.time()

    ### record logs ###
    if record_exp != "":    
        metrics = akasha.format.handle_metrics(doc_length, end_time - start_time, tokens)
        params['doc_range'] = doc_range
        akasha.aiido_upload(record_exp, params, metrics, table)
    akasha.helper.save_logs(logs)
    
    
    ### write question and answer into txt file, but first check if "questionset" directory exist or not, it not, first create it.
    ### for filename, count the files in the questionset directory that has doc_path in the file name, and use it as the file name.
    if not os.path.exists("questionset"):
        os.makedirs("questionset")
    count = 1
    suf_path = doc_path.split('/')[-2]
    for filename in os.listdir("questionset"):
        if suf_path in filename:
            count += 1
    file_name = "questionset/"+suf_path+"_"+str(count)+".txt"
    with open(file_name, "w", encoding="utf-8") as f:
        
        for w in range(len(question)):
            if question_type == "essay":
                f.write(question[w]  + answer[w] + "\n\n")
            else:    
                if w == len(question)-1:
                    f.write(question[w].replace('\n','') + answer[w].replace('\n',''))
                else:
                    f.write(question[w].replace('\n','') + answer[w].replace('\n','') + "\n")
    
    print("question set saved in ", file_name,"\n\n")
    return question, answer
    
    


def auto_evaluation(questionset_path:str, doc_path:str, question_type:str="essay", embeddings:str = "openai:text-embedding-ada-002", chunk_size:int=1000\
                 , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
                 language:str = 'ch' , search_type:str = 'merge', record_exp:str = "", eval_model:str = "openai:gpt-3.5-turbo"\
                , max_token:int=3000):
    """parse the question set txt file generated from "auto_create_questionset" function if you use essay type to generate questionset, 
    and then use llm model to generate response, 
    evaluate the performance of the given paramters based on similarity between responses and the default answers, use bert_score 
    and rouge_l to evaluate the response.

    Args:
        **questionset_path (str)**: the path of question set txt file\n
        **question_type (str, optional)**: the type of question you want to generate, "essay" or "single_choice". Defaults to "essay".\n
        **doc_path (str)**: documents directory path\n
        **embeddings (str, optional)**: the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002".\n
        **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.\n
        **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
        **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
        **topK (int, optional)**: search top k number of similar documents. Defaults to 2.\n
        **threshold (float, optional)**: the similarity threshold of searching. Defaults to 0.2.\n
        **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
            max token size of llm input.\n
        **search_type (str, optional)**: search type to find similar documents from db, default 'merge'.
            includes 'merge', 'mmr', 'svm', 'tfidf'.\n
        **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
            record_exp as experiment name.  default "".\n
        **eval_model (str, optional)**: llm model use to score the response. Defaults to "gpt-3.5-turbo".\n
        
    Returns:
        (float, float, float): average bert_score, average rouge_l score  and avg llm_score of all questions
    """
    #read the whole txt file and separate it into question and answer list by "\n\n"

    
    question, answer = akasha.helper.get_question_from_file(questionset_path, question_type)
    if question_type == "essay":
        bert = []
        rouge = []
        llm_score = []    
    else:
        correct_count = 0

    
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
    logs = [f"\n\n---------------auto_evaluation({question_type})------------------------\n"]
    table = {}
    params = akasha.format.handle_params(model, embeddings, chunk_size, search_type, topK, threshold, language, False)
    embeddings_name = embeddings
    embeddings = akasha.helper.handle_embeddings(embeddings, logs, verbose)
    model = akasha.helper.handle_model(model, logs, verbose)
    progress = tqdm(total = total_question, desc=f"Run Eval({question_type})")
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
        if question_type.lower() == "essay":
            query = question[i]
            query_with_prompt = system_prompt + question[i]
        else:
            query, ans = akasha.prompts.format_question_query(question[i])
            query_with_prompt = akasha.prompts.format_llama_json(query)
            
        docs, docs_token = akasha.search.get_docs(db, embeddings, query, topK, threshold, language, search_type, verbose,\
                     logs, model, False, max_token)
        doc_length += akasha.helper.get_docs_length(language, docs)
        tokens += docs_token
        
        

        try:
            chain = load_qa_chain(llm=model, chain_type="stuff",verbose=False)
            response = chain.run(input_documents = docs, question =  query_with_prompt)
            response = akasha.helper.sim_to_trad(response)
            response = response.split("Finished chain.")
        except:
            print("running model error\n")
            response = ["running model error"]
            torch.cuda.empty_cache()

        if verbose:
            print(question[i],"\n\n")
            print(response[-1],"\n\n")
        
        logs.append("\n\ndocuments: \n\n" + ''.join([doc.page_content for doc in docs]))
        logs.append("\n\nresponse:\n\n"+ response[-1])
        
        if question_type.lower() == "essay":
            bert.append(eval.scores.get_bert_score(response[-1], answer[i], language))
            rouge.append(eval.scores.get_rouge_score(response[-1], answer[i], language))
            llm_score.append(eval.scores.get_llm_score(response[-1], answer[i], eval_model))
        
            new_table = akasha.format.handle_table(question[i], docs, response[-1])
            new_table = akasha.format.handle_score_table(new_table, bert[-1], rouge[-1],llm_score[-1])
        else:
            new_table = akasha.format.handle_table(query, docs, response)
            result = akasha.helper.extract_result(response)
            if str(result) == str(ans):
                correct_count += 1
        
        for key in new_table:
            if key not in table:
                table[key] = []
            table[key].append(new_table[key])
                
        
    progress.close() #end running llm progress bar
    
    ### record logs ###
    end_time = time.time()
    akasha.helper.save_logs(logs)
    if question_type.lower() == "essay":
        avg_bert = round(sum(bert)/len(bert),3)
        avg_rouge = round(sum(rouge)/len(rouge),3)
        avg_llm_score = round(sum(llm_score)/len(llm_score),3)
        if record_exp != "":    
            metrics = akasha.format.handle_metrics(doc_length, end_time - start_time, tokens)
            metrics['avg_bert'] = avg_bert
            metrics['avg_rouge'] = avg_rouge
            metrics['avg_llm_score'] = avg_llm_score
            akasha.aiido_upload(record_exp, params, metrics, table)
        return avg_bert, avg_rouge, avg_llm_score, tokens
    
    else:
        if record_exp != "":    
            metrics = akasha.format.handle_metrics(doc_length, end_time - start_time, tokens)
            metrics['correct_rate'] = correct_count/total_question
            akasha.aiido_upload(record_exp, params, metrics, table)
        return correct_count/total_question , tokens
  

