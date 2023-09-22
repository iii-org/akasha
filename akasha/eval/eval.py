import datetime
import time
from tqdm import tqdm
import akasha
import os
import numpy as np
import torch,gc
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from typing import Callable, Union



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
                 language:str = 'ch' , search_type:Union[str,Callable] = 'merge', record_exp:str = "", \
                 system_prompt:str = "", question_type:str="essay",choice_num:int = 4 ):
    """auto create question set by llm model, each time it will randomly select a range of documents from the documents directory, 
    then use llm model to generate a question and answer pair, and save it into a txt file.
    1.The format of "single_choice" questionset should be one line one question, and the possibles answers and questions are separate by tab(\t),
    the last one is which options is the correct answer, for example, the file should look like: \n
        "What is the capital of Taiwan?" Taipei  Kaohsiung  Taichung  Tainan     1
        何者是台灣的首都?   台北    高雄    台中    台南    1
    2. The format of "essay" questionset should be one line one question, and the reference answer is next line, every questions are separate by 
    two newline(\n\n). For example, the file should look like: \n
        問題：根據文件中的訊息，智慧製造的複雜性已超越系統整合商的負荷程度，未來產業鏈中的角色將傾向朝共和共榮共創智慧製造商機，而非過往的單打獨鬥模式發展。請問為什麼供應商、電信商、軟體開發商、平台商、雲端服務供應商、系統整合商等角色會傾向朝共和共榮共創智慧製造商機的方向發展？
        答案：因為智慧製造的複雜性已超越系統整合商的負荷程度，單一角色難以完成整個智慧製造的需求，而共和共榮共創的模式可以整合各方的優勢，共同創造智慧製造的商機。

        問題：根據文件中提到的資訊技術商（IT）和營運技術商（OT），請列舉至少兩個邊緣運算產品或解決方案。
        答案：根據文件中的資訊，NVIDIA的邊緣運算產品包括Jetson系列和EGX系列，而IBM的邊緣運算產品包括IBM Edge Application Manager和IBM Watson Anywhere。

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
    logs = [f"\n\n-----------------auto_create_questionset({question_type})----------------------\n"]
    if callable(search_type):
        search_type_str = search_type.__name__
    else:
        search_type_str = search_type
    params = akasha.format.handle_params(model, embeddings, chunk_size, search_type_str, topK, threshold, language, False)
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
    
    del model,embeddings,db,docs
    gc.collect()
    torch.cuda.empty_cache()
    return question, answer
    
    


def auto_evaluation(questionset_path:str, doc_path:str, question_type:str="essay", embeddings:str = "openai:text-embedding-ada-002", chunk_size:int=1000\
                 , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
                 language:str = 'ch' , search_type:Union[str,Callable]= 'merge', record_exp:str = "", eval_model:str = "openai:gpt-3.5-turbo"\
                , max_token:int=3000, system_prompt:str=""):
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
        **max_token (int, optional)**: the max token size of llm input. Defaults to 3000.\n
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
        system_prompt =  system_prompt + " 用中文回答 " 
    if system_prompt != "":
        system_prompt = "[INST] <<SYS>> " + system_prompt + " <</SYS>> [/INST]"
    
    ### define variables ###
    total_question = len(question)
    doc_length = 0
    tokens = 0
    start_time = time.time()
    logs = [f"\n\n---------------auto_evaluation({question_type})------------------------\n"]
    table = {}
    if callable(search_type):
        search_type_str = search_type.__name__
    else:
        search_type_str = search_type
    params = akasha.format.handle_params(model, embeddings, chunk_size, search_type_str, topK, threshold, language, False)
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
            print("Question: ", question[i],"\n\n")
            if question_type.lower() == "essay":
                print("Reference Answer: ", answer[i],"\n\n")
            else:    
                print("Reference Answer: ", ans,"\n\n")
            print("Generated Response: ", response[-1],"\n\n")
        
        logs.append("\n\ndocuments: \n\n" + ''.join([doc.page_content for doc in docs]))
        logs.append("\n\nresponse:\n\n"+ response[-1])
        
        if question_type.lower() == "essay":
            bert.append(eval.scores.get_bert_score(response[-1], answer[i], language))
            rouge.append(eval.scores.get_rouge_score(response[-1], answer[i], language))
            llm_score.append(eval.scores.get_llm_score(response[-1], answer[i], eval_model))
        
            new_table = akasha.format.handle_table(question[i] + "\nAnswer:  "+ answer[i], docs, response[-1])
            new_table = akasha.format.handle_score_table(new_table, bert[-1], rouge[-1],llm_score[-1])
        else:
            new_table = akasha.format.handle_table(query + "\nAnswer:  " + ans, docs, response)
            result = akasha.helper.extract_result(response)
            if str(result) == str(ans):
                correct_count += 1
        
        for key in new_table:
            if key not in table:
                table[key] = []
            table[key].append(new_table[key])
                
        
    progress.close() #end running llm progress bar
    
    del model,chain,embeddings,db
    gc.collect()
    torch.cuda.empty_cache()
    
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
  







def optimum_combination(q_file:str, doc_path:str, question_type:str="essay", embeddings_list:list = ["openai:text-embedding-ada-002"], chunk_size_list:list=[500]\
                 , model_list:list = ["openai:gpt-3.5-turbo"], topK_list:list = [2], threshold:float = 0.2,\
                 language:str = 'ch' , search_type_list:list = ['svm','tfidf','mmr'], record_exp:str = ""\
                    , max_token:int=3000, system_prompt:str=""):
    """test all combinations of giving lists, and run test_performance to find parameters of the best result.

    Args:
        **q_file (str)**: the file path of the question file\n
        **doc_path (str)**: documents directory path\n
        **question_type (str, optional)**: the type of question you want to generate, "essay" or "single_choice". Defaults to "essay".\n
        **embeddings_list (_type_, optional)**: list of embeddings models. Defaults to ["openai:text-embedding-ada-002"].\n
        **chunk_size_list (list, optional)**: list of chunk sizes. Defaults to [500].\n
        **model_list (_type_, optional)**: list of models. Defaults to ["openai:gpt-3.5-turbo"].\n
        **topK_list (list, optional)**: list of topK. Defaults to [2].\n
        **threshold (float, optional)**: the similarity threshold of searching. Defaults to 0.2.\n
        **search_type_list (list, optional)**: list of search types, currently have "merge", "svm", "knn", "tfidf", "mmr". Defaults to ['svm','tfidf','mmr'].
        **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
            record_exp as experiment name.  default ''.\n
        **max_token (int, optional)**: max token size of llm input. Defaults to 3000.\n
        **system_prompt (str, optional)**: the system prompt that you assign special instruction to llm model, currently only be used in the essay
        question type\n
    Returns:
        (list,list): return best score combination and best cost-effective combination
    """
    logs = ["\n\n----------------optimum_combination-----------------------\n"]
    start_time = time.time()
    combinations = akasha.helper.get_all_combine(embeddings_list, chunk_size_list, model_list, topK_list, search_type_list)
    progress = tqdm(len(combinations),total = len(combinations), desc="RUN LLM COMBINATION")
    print("\n\ntotal combinations: ", len(combinations))
    result_list = []
    if question_type.lower() == "essay":
        bcb = 0.0
        bcr = 0.0
        bcl = 0.0
    else:
        bcr = 0.0
    
    for embed, chk, mod, tK, st in combinations:
        progress.update(1)
        
        if question_type.lower() == "essay":
            cur_bert, cur_rouge, cur_llm, tokens = auto_evaluation(q_file, doc_path, question_type,embeddings=embed, chunk_size=chk, model=mod, topK=tK, threshold=threshold,\
                                language=language, search_type=st, record_exp=record_exp, max_token=max_token, system_prompt=system_prompt) 
            bcb = max(bcb,cur_bert)
            bcr = max(bcr,cur_rouge)
            bcl = max(bcl,cur_llm)
            cur_tup = (cur_bert, cur_rouge, cur_llm, embed, chk, mod, tK, st)
        else:
            cur_correct_rate, tokens = auto_evaluation(q_file, doc_path, question_type,embeddings=embed, chunk_size=chk, model=mod, topK=tK, threshold=threshold,\
                                language=language, search_type=st, record_exp=record_exp, max_token=max_token) 
            bcr = max(bcr,cur_correct_rate)
            cur_tup = (cur_correct_rate, cur_correct_rate/tokens, embed, chk, mod, tK, st)
        result_list.append(cur_tup)
        
    progress.close()


    if question_type.lower() == "essay":
        ### record bert score logs ###
        print("Best Bert Score: ", "{:.3f}".format(bcb))
        
        bs_combination = akasha.helper.get_best_combination(result_list, 0, logs)
        print("\n\n")
        
        ### record rouge score logs ###
        print("Best Rouge Score: ", "{:.3f}".format(bcr))
        
        rs_combination = akasha.helper.get_best_combination(result_list, 1, logs)
        print("\n\n")
        
        ### record llm_score logs ###
        print("Best llm score: ", "{:.3f}".format(bcl))
        # score_comb = "Best score combination: \n"
        # print(score_comb)
        # logs.append(score_comb)
        ls_combination = akasha.helper.get_best_combination(result_list, 2, logs)
        print("\n\n")
        
        
    else:
        ### record logs ###
        print("Best correct rate: ", "{:.3f}".format(bcr))
        score_comb = "Best score combination: \n"
        print(score_comb)
        logs.append(score_comb)
        bs_combination = akasha.helper.get_best_combination(result_list, 0,logs)


        print("\n\n")
        cost_comb = "Best cost-effective: \n"
        print(cost_comb)
        logs.append(cost_comb)
        bc_combination = akasha.helper.get_best_combination(result_list, 1,logs)



   
    end_time = time.time()
    format_time = "time spend: " + "{:.3f}".format(end_time-start_time)
    print( format_time )
    logs.append( format_time )
    akasha.helper.save_logs(logs)
    
    if question_type.lower() == "essay":
        return bs_combination, rs_combination, ls_combination
    return bs_combination, bc_combination




class Model_Eval(akasha.atman):
    def __init__(self, embeddings:str = "openai:text-embedding-ada-002", chunk_size:int=1000\
        , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
        language:str = 'ch' , search_type:Union[str,Callable] = 'svm', record_exp:str = "", \
        system_prompt:str = "", max_token:int=3000, temperature:float=0.0):
        
        super().__init__(chunk_size, model, verbose, topK, threshold,\
        language , search_type, record_exp, system_prompt, max_token, temperature)
        ### set argruments ###
        self.doc_path = ""
        self.question_type = ""
        self.question_num = 0
        self.embeddings = embeddings

        

        ### set variables ###
        self.logs = {}
        self.model_obj = akasha.helper.handle_model(model, self.verbose, self.temperature)
        self.embeddings_obj = akasha.helper.handle_embeddings(embeddings, self.verbose)
        self.search_type = search_type
        self.db = None
        self.docs = []
        self.doc_tokens = 0
        self.doc_length = 0
        self.response = ""
        self.prompt = ""