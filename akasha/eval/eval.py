import datetime
import time
from tqdm import tqdm
import akasha
import akasha.eval as eval
import os
import numpy as np
import torch,gc
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from typing import Callable, Union, Tuple, List



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





class Model_Eval(akasha.atman):
    """class for implement evaluation of llm model, include auto_create_questionset and auto_evaluation.

    """
    def __init__(self, embeddings:str = "openai:text-embedding-ada-002", chunk_size:int=1000\
        , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
        language:str = 'ch' , search_type:Union[str,Callable] = 'svm', record_exp:str = "", \
        system_prompt:str = "", max_token:int=3000, temperature:float=0.0, question_type:str="essay"):
        """initials of Model_Eval class

        Args:
            **embeddings (str, optional)**: the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002".\n
            **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.\n
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
            **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
            **topK (int, optional)**: search top k number of similar documents. Defaults to 2.\n
            **threshold (float, optional)**: the similarity threshold of searching. Defaults to 0.2.\n
            **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
                max token size of llm input.\n
            **search_type (str, optional)**: search type to find similar documents from db, default 'merge'.
                includes 'merge', 'mmr', 'svm', 'tfidf', also, you can custom your own search_type function, as long as your
                function input is (query_embeds:np.array, docs_embeds:list[np.array], k:int, relevancy_threshold:float, log:dict) 
                and output is a list [index of selected documents].\n
            **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
                record_exp as experiment name.  default "".\n
            **system_prompt (str, optional)**: the system prompt that you assign special instruction to llm model, so will not be used
                in searching relevant documents. Defaults to "".\n
            **max_token (int, optional)**: max token size of llm input. Defaults to 3000.\n
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n
            **question_type (str, optional)**: the type of question you want to generate, "essay" or "single_choice". Defaults to "essay".\n
        """
        
        super().__init__(chunk_size, model, verbose, topK, threshold,\
        language , search_type, record_exp, system_prompt, max_token, temperature)
        ### set argruments ###
        self.doc_path = ""
        self.question_type = question_type
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
        self.question = []
        self.answer = []
        self.response = []
        self.score = {}
        
        
        
    def auto_create_questionset(self, doc_path:Union[List[str],str], question_num:int = 10, choice_num:int = 4, **kwargs)->(list,list):
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
            **choice_num (int, optional)**: the number of choices for each single choice question, only use it if question_type is "single_choice".
        Defaults to 4.\n
            **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
            embeddings, chunk_size, model, verbose, topK, threshold, language , search_type, record_exp, 
            system_prompt, max_token, temperature.
        Raises:
            Exception: _description_

        Returns:
            (question_list:list, answer_list:list): the question and answer list that generated by llm model
        """
        ## set class variables ##
        self._set_model( **kwargs)
        self._change_variables(**kwargs)
        self.doc_path = doc_path
        self.question_num = question_num

        ## check db ##
        
        self.db = akasha.helper.processMultiDB(self.doc_path, self.verbose, "eval_get_doc", self.embeddings, self.chunk_size)
        if not self._check_db():
            return ""
        
        ## set local variables ##
        timestamp = datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S")
        self.timestamp_list.append(timestamp)
        start_time = time.time()
        doc_range = (1999+self.chunk_size)//self.chunk_size   # doc_range is determine by the chunk size, so the select documents won't be too short to having trouble genereating a question
        vis_doc_range = set()  
        self.doc_tokens, self.doc_length = 0, 0
        self.question, self.answer, self.docs = [], [], []
        table = {}
        ## add logs ##
        
        self._add_basic_log(timestamp, "auto_create_questionset")
        self.logs[timestamp]["doc_range"] = doc_range
        self.logs[timestamp]["question_num"] = question_num
        self.logs[timestamp]["question_type"] = self.question_type
        self.logs[timestamp]["choice_num"] = choice_num
        
        
        # db_data = self.db.get(include=['documents','metadatas'])
        # texts = db_data['documents']
        # metadata =  db_data['metadatas']
        texts = [doc.page_content for doc in self.db]
        metadata = [doc.metadata for doc in self.db]
        
        progress = tqdm(total = question_num, desc=f"Create Q({self.question_type})")
        regenerate_limit = question_num
        ### random select a range of documents from the documents , and use llm model to generate a question and answer pair ###
        for i in range(question_num):
            progress.update(1)
            
            random_index = akasha.helper.get_non_repeat_rand_int(vis_doc_range, len(texts) - doc_range)
            
            doc_text = '\n'.join(texts[random_index:random_index + doc_range])
            docs = [Document(page_content=texts[k], metadata=metadata[k]) for k in range(random_index,random_index+doc_range)]
            self.doc_length += akasha.helper.get_docs_length(self.language, docs)
            self.doc_tokens += self.model_obj.get_num_tokens(doc_text)
            self.docs.extend(docs)
            try:
                q_prompt = akasha.prompts.format_create_question_prompt(doc_text, self.question_type)
                
                response = akasha.helper.call_model(self.model_obj, q_prompt)       
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
            
            self.question.append("問題："+process[0])
            
            
            if self.question_type == "essay":
                self.answer.append("答案："+process[1])
                
            else:
                anss = _generate_single_choice_question(doc_text, process[0], process[1], self.model_obj, self.system_prompt, choice_num)
                self.answer.append(anss)
                response = process[0] + "\n"  + "選項:\n" + anss + "\n\n"
            
            
            if self.verbose:
                print(response)
           
            new_table = akasha.format.handle_table(self.question[-1], docs, self.answer[-1])
            for key in new_table:
                if key not in table:
                    table[key] = []
                table[key].append(new_table[key])
            
        progress.close() #end running llm progress bar
        
        end_time = time.time()

        ### record logs ###
        if self.record_exp != "":    
            params =  akasha.format.handle_params(self.model, self.embeddings, self.chunk_size, self.search_type_str,\
                self.topK, self.threshold, self.language)   
            metrics = akasha.format.handle_metrics(self.doc_length, end_time - start_time, self.doc_tokens)
            params['doc_range'] = doc_range
            akasha.aiido_upload(self.record_exp, params, metrics, table)

        
        
        ### write question and answer into txt file, but first check if "questionset" directory exist or not, it not, first create it.
        ### for filename, count the files in the questionset directory that has doc_path in the file name, and use it as the file name.
        if not os.path.exists("questionset"):
            os.makedirs("questionset")
        count = 1
        if isinstance(doc_path, list):
            suf_path = doc_path[0].split('/')[-2]
        else:
            suf_path = doc_path.split('/')[-2]
        for filename in os.listdir("questionset"):
            if suf_path in filename:
                count += 1
        file_name = "questionset/"+suf_path+"_"+str(count)+".txt"
        with open(file_name, "w", encoding="utf-8") as f:
            
            for w in range(len(self.question)):
                if self.question_type == "essay":
                    f.write(self.question[w]  + self.answer[w] + "\n\n")
                else:    
                    if w == len(self.question)-1:
                        f.write(self.question[w].replace('\n','') + self.answer[w].replace('\n',''))
                    else:
                        f.write(self.question[w].replace('\n','') + self.answer[w].replace('\n','') + "\n")
        
        print("question set saved in ", file_name,"\n\n")
        
        self._add_result_log(timestamp, end_time-start_time)
        self.logs[timestamp]["question"] = self.question
        self.logs[timestamp]["answer"] = self.answer
        self.logs[timestamp]["questionset_path"] = file_name
        
            
        del self.db
        return self.question, self.answer
    
    
    
    
    def auto_evaluation(self, questionset_file:str, doc_path:Union[List[str],str], eval_model:str="openai:gpt-3.5-turbo", **kwargs)\
    ->Union[Tuple[float,float,float,int] , Tuple[float,int]]:
        """parse the question set txt file generated from "auto_create_questionset" function and then use llm model to generate response, 
    evaluate the performance of the given paramters based on similarity between responses and the default answers, use bert_score 
    and rouge_l to evaluate the response if you use essay type to generate questionset.  And use correct_count to evaluate 
    the response if you use single_choice type to generate questionset.  **Noted that the question_type must match the questionset_file's type**.
    

        Args:
        **questionset_flie (str)**: the path of question set txt file, accept .txt, .docx and .pdf.\n
        **question_type (str, optional)**: the type of question you want to generate, "essay" or "single_choice". Defaults to "essay".\n
        **eval_model (str, optional)**: llm model use to score the response. Defaults to "gpt-3.5-turbo".\n
        **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
            embeddings, chunk_size, model, verbose, topK, threshold, language , search_type, record_exp, 
            system_prompt, max_token, temperature.\n
        """
         
        
        ## set class variables ##
        self._set_model( **kwargs)
        self._change_variables(**kwargs)
        self.doc_path = doc_path
        if self.question_type=="essay" and self.language=='ch' and "用中文回答" not in self.system_prompt:
            self.system_prompt =  self.system_prompt + " 用中文回答 "
            
        ## check db ##
        self.db = akasha.helper.processMultiDB(self.doc_path, self.verbose, self.embeddings_obj, self.embeddings, self.chunk_size)
        if not self._check_db():
            return ""
        
        ## set local variables ##
        timestamp = datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S")
        self.timestamp_list.append(timestamp)
        start_time = time.time()
        self.doc_tokens, self.doc_length = 0, 0
        self.question, self.answer, self.docs = [], [], []
        if self.question_type.lower() == "essay":
            self.score = {"bert":[], "rouge":[], "llm_score":[]}
        else:
            self.score = {"correct_count":0}
        table = {}
        question, answer = akasha.helper.get_question_from_file(questionset_file, self.question_type)
        self.question_num = len(question)
        progress = tqdm(total = self.question, desc=f"Run Eval({self.question_type})")
        ## add logs ##
        
        self._add_basic_log(timestamp, "auto_evaluation")
        self.logs[timestamp]["questionset_path"] = questionset_file
        self.logs[timestamp]["question_num"] = self.question_num
        self.logs[timestamp]["question_type"] = self.question_type
        self.logs[timestamp]["search_type"] = self.search_type_str
         
        
        
        ### for each question and answer, use llm model to generate response, and evaluate the response by bert_score and rouge_l ###
        for i in range(self.question_num):
            
            progress.update(1)
            if self.question_type.lower() == "essay":
                query = question[i]
                query_with_prompt = akasha.prompts.format_sys_prompt(self.system_prompt, question[i])
                
            else:
                query, ans = akasha.prompts.format_question_query(question[i])
                query_with_prompt = akasha.prompts.format_llama_json(query)
                
            docs, docs_token = akasha.search.get_docs(self.db, self.embeddings_obj, query, self.topK, \
                self.threshold, self.language, self.search_type, self.verbose, self.model_obj, self.max_token, self.logs[timestamp])
            self.doc_length += akasha.helper.get_docs_length(self.language, docs)
            self.doc_tokens += docs_token
            
            

            try:
                chain = load_qa_chain(llm=self.model_obj, chain_type="stuff", verbose=self.verbose)
                response = chain.run(input_documents = docs, question =  query_with_prompt)
                response = akasha.helper.sim_to_trad(response)
                self.docs.extend(docs)
                self.response.append(response)
            except:
                print("running model error\n")
                response = ["running model error"]
                torch.cuda.empty_cache()

            if self.verbose:
                print("Question: ", question[i],"\n\n")
                if self.question_type.lower() == "essay":
                    print("Reference Answer: ", answer[i],"\n\n")
                else:    
                    print("Reference Answer: ", ans,"\n\n")
                print("Generated Response: ", response,"\n\n")
            
            # ---- #
            
            if self.question_type.lower() == "essay":
                self.score["bert"].append(eval.scores.get_bert_score(response, answer[i], self.language))
                self.score["rouge"].append(eval.scores.get_rouge_score(response, answer[i], self.language))
                self.score["llm_score"].append(eval.scores.get_llm_score(response, answer[i], eval_model))
            
                new_table = akasha.format.handle_table(question[i] + "\nAnswer:  "+ answer[i], docs, response)
                new_table = akasha.format.handle_score_table(new_table, self.score["bert"][-1], self.score["rouge"][-1], self.score["llm_score"][-1])
            else:
                new_table = akasha.format.handle_table(query + "\nAnswer:  " + ans, docs, response)
                result = akasha.helper.extract_result(response)
                if str(result) == str(ans):
                    self.score["correct_count"] += 1
            
            for key in new_table:
                if key not in table:
                    table[key] = []
                table[key].append(new_table[key])
                    
            
        progress.close() #end running llm progress bar
        
        
        
        ### record logs ###
        end_time = time.time()
        self._add_result_log(timestamp, end_time-start_time)
        self.logs[timestamp]["response"] = self.response
        
        if self.question_type.lower() == "essay":
            avg_bert = round(sum(self.score["bert"])/len(self.score["bert"]),3)
            avg_rouge = round(sum(self.score["rouge"])/len(self.score["rouge"]),3)
            avg_llm_score = round(sum(self.score["llm_score"])/len(self.score["llm_score"]),3)
            self.logs[timestamp]["bert"] = self.score["bert"]
            self.logs[timestamp]["rouge"] = self.score["rouge"]
            self.logs[timestamp]["llm_score"] = self.score["llm_score"]
            if self.record_exp != "":    
                params =  akasha.format.handle_params(self.model, self.embeddings, self.chunk_size, self.search_type_str,\
                    self.topK, self.threshold, self.language)           
                metrics = akasha.format.handle_metrics(self.doc_length, end_time - start_time, self.doc_tokens)
                metrics['avg_bert'] = avg_bert
                metrics['avg_rouge'] = avg_rouge
                metrics['avg_llm_score'] = avg_llm_score
                akasha.aiido_upload(self.record_exp, params, metrics, table)
            del self.db
            return avg_bert, avg_rouge, avg_llm_score, self.doc_tokens
        
        else:
            self.logs[timestamp]["correct_rate"] = self.score["correct_count"]/self.question_num
            if self.record_exp != "":  
                params =  akasha.format.handle_params(self.model, self.embeddings, self.chunk_size, self.search_type_str,\
                    self.topK, self.threshold, self.language)    
                metrics = akasha.format.handle_metrics(self.doc_length, end_time - start_time, self.doc_tokens)
                metrics['correct_rate'] = self.score["correct_count"]/self.question_num
                akasha.aiido_upload(self.record_exp, params, metrics, table)
            del self.db
            return self.logs[timestamp]["correct_rate"] , self.doc_tokens
        
        
        
        
    def optimum_combination(self,questionset_flie:str, doc_path:Union[List[str],str], embeddings_list:list = ["openai:text-embedding-ada-002"], chunk_size_list:list=[500]\
                    , model_list:list = ["openai:gpt-3.5-turbo"], topK_list:list = [2], search_type_list:list = ['svm','tfidf','mmr'], **kwargs\
                        )->(list,list):
        """test all combinations of giving lists, and run auto_evaluation to find parameters of the best result.

        Args:
            **questionset_flie (str)**: the path of question set txt file, accept .txt, .docx and .pdf.\n
            **doc_path (str)**: documents directory path\n
            **question_type (str, optional)**: the type of question you want to generate, "essay" or "single_choice". Defaults to "essay".\n
            **embeddings_list (_type_, optional)**: list of embeddings models. Defaults to ["openai:text-embedding-ada-002"].\n
            **chunk_size_list (list, optional)**: list of chunk sizes. Defaults to [500].\n
            **model_list (_type_, optional)**: list of models. Defaults to ["openai:gpt-3.5-turbo"].\n
            **topK_list (list, optional)**: list of topK. Defaults to [2].\n
            **threshold (float, optional)**: the similarity threshold of searching. Defaults to 0.2.\n
            **search_type_list (list, optional)**: list of search types, currently have "merge", "svm", "knn", "tfidf", "mmr". Defaults to ['svm','tfidf','mmr'].
        Returns:
            (list,list): return best score combination and best cost-effective combination
        """
        
        self._change_variables(**kwargs)
        start_time = time.time()
        combinations = akasha.helper.get_all_combine(embeddings_list, chunk_size_list, model_list, topK_list, search_type_list)
        progress = tqdm(len(combinations),total = len(combinations), desc="RUN LLM COMBINATION")
        print("\n\ntotal combinations: ", len(combinations))
        result_list = []
        if self.question_type.lower() == "essay":
            bcb = 0.0
            bcr = 0.0
            bcl = 0.0
        else:
            bcr = 0.0
        
        
        for embed, chk, mod, tK, st in combinations:
            progress.update(1)
            
            if self.question_type.lower() == "essay":
                cur_bert, cur_rouge, cur_llm, tokens = self.auto_evaluation(questionset_flie, doc_path, embeddings=embed, chunk_size=chk, model=mod, topK=tK,\
                    search_type=st)
                
                bcb = max(bcb,cur_bert)
                bcr = max(bcr,cur_rouge)
                bcl = max(bcl,cur_llm)
                cur_tup = (cur_bert, cur_rouge, cur_llm, embed, chk, mod, tK, self.search_type_str)
            else:
                cur_correct_rate, tokens = self.auto_evaluation(questionset_flie, doc_path, embeddings=embed, chunk_size=chk, model=mod, topK=tK,\
                    search_type=st)
                bcr = max(bcr,cur_correct_rate)
                cur_tup = (cur_correct_rate, cur_correct_rate/tokens, embed, chk, mod, tK, self.search_type_str)
            result_list.append(cur_tup)
            
        progress.close()


        if self.question_type.lower() == "essay":
            ### record bert score logs ###
            print("Best Bert Score: ", "{:.3f}".format(bcb))
            
            bs_combination = akasha.helper.get_best_combination(result_list, 0)
            print("\n\n")
            
            ### record rouge score logs ###
            print("Best Rouge Score: ", "{:.3f}".format(bcr))
            
            rs_combination = akasha.helper.get_best_combination(result_list, 1)
            print("\n\n")
            
            ### record llm_score logs ###
            print("Best llm score: ", "{:.3f}".format(bcl))
            # score_comb = "Best score combination: \n"
            # print(score_comb)
            # logs.append(score_comb)
            ls_combination = akasha.helper.get_best_combination(result_list, 2)
            print("\n\n")
            
            
        else:
            ### record logs ###
            print("Best correct rate: ", "{:.3f}".format(bcr))
            score_comb = "Best score combination: \n"
            print(score_comb)
            
            bs_combination = akasha.helper.get_best_combination(result_list, 0)


            print("\n\n")
            cost_comb = "Best cost-effective: \n"
            print(cost_comb)
            
            bc_combination = akasha.helper.get_best_combination(result_list, 1)



    
        end_time = time.time()
        format_time = "time spend: " + "{:.3f}".format(end_time-start_time)
        print( format_time )
        
        
        if self.question_type.lower() == "essay":
            return bs_combination, rs_combination, ls_combination
        return bs_combination, bc_combination   



