import pathlib
import time
from tqdm import tqdm
from typing import Callable, Union
from langchain.chains.question_answering import load_qa_chain, LLMChain
from langchain import PromptTemplate
from langchain.schema import Document
import akasha.helper as helper
import akasha.search as search
import akasha.format as format
import akasha.prompts as prompts
import datetime
from dotenv import load_dotenv

load_dotenv(pathlib.Path().cwd()/'.env') 


def aiido_upload(exp_name, params:dict={}, metrics:dict={}, table:dict={}, path_name:str=""):
    """upload params_metrics, table to mlflow server for tracking.

    Args:
        **exp_name (str)**: experiment name on the tracking server, if not found, will create one .\n
        **params (dict, optional)**: parameters dictionary. Defaults to {}.\n
        **metrics (dict, optional)**: metrics dictionary. Defaults to {}.\n
        **table (dict, optional)**: table dictionary, used to compare text context between different runs in the experiment. Defaults to {}.\n
    """
    import aiido
    
    if "model" not in params or "embeddings" not in params:
        aiido.init(experiment=exp_name, run = path_name)
        
    else:
        mod = params["model"].split(':')
        emb = params["embeddings"].split(':')[0]
        sea = params["search_type"]
        aiido.init(experiment=exp_name, run = emb + '-' + sea + '-' + '-'.join(mod))
    
    aiido.log_params_and_metrics(params=params, metrics=metrics)


    if len(table) > 0:
        aiido.mlflow.log_table(table,"table.json")
    aiido.mlflow.end_run()
    return






def detect_exploitation(texts:str, model:str = "openai:gpt-3.5-turbo", verbose:bool = False, record_exp:str = ""):
    """ check the given texts have harmful or sensitive information

    Args:
        **texts (str)**: texts that we want llm to check.\n
        **model (str, optional)**: llm model name. Defaults to "openai:gpt-3.5-turbo".\n
        **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
        **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set 
            record_exp as experiment name.  default ''.\n

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





class atman():
    def __init__(self, chunk_size:int=1000\
        , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
        language:str = 'ch' , search_type:Union[str,Callable] = 'svm', record_exp:str = "", \
        system_prompt:str = "", max_token:int=3000, temperature:float=0.0):
        
        
        self.chunk_size = chunk_size
        self.model = model
        self.verbose = verbose
        self.topK = topK
        self.threshold = threshold
        self.language = language
        self.search_type_str = helper.handle_search_type(search_type, self.verbose)
        self.record_exp = record_exp
        self.system_prompt = system_prompt
        self.max_token = max_token
        self.temperature = temperature
        
        self.timestamp_list = []
        
    
    def _change_variables(self, **kwargs):
        
        ## check if we need to change db, model_obj or embeddings_obj ##
        if "search_type" in kwargs:
            self.search_type_str = helper.handle_search_type(kwargs["search_type"], self.verbose)
            
        if  "embeddings" in kwargs:
            self.embeddings_obj = helper.handle_embeddings(kwargs["embeddings"], self.verbose)
            
        if "model" in kwargs or "temperature" in kwargs:
            new_temp = self.temperature
            new_model = self.model
            if "temperature" in kwargs:
                new_temp = kwargs["temperature"]
            if "model"  in kwargs:
                new_model = kwargs["model"]
            self.model_obj = helper.handle_model(new_model, self.verbose, new_temp)
                  
            
            
            
        ### check input argument is valid or not ###
        for key, value in kwargs.items():
            if key in self.__dict__: # check if variable exist
                if getattr(self, key, None) != value: # check if variable value is different
                    
                    self.__dict__[key] = value
            else:
                print(f"argument {key} not exist")
    
        return 
    
    
    def _check_db(self):
        
        if self.db is None:
            info = "document path not exist\n"
            print(info)
            return False
        else:
            return True
    
    def _add_basic_log(self, timestamp:str, fn_type:str):
        
        if timestamp not in self.logs:
            self.logs[timestamp] = {}
        self.logs[timestamp]["fn_type"] = fn_type
        self.logs[timestamp]["model"] = self.model
        self.logs[timestamp]["chunk_size"] = self.chunk_size
        self.logs[timestamp]["topK"] = self.topK
        self.logs[timestamp]["threshold"] = self.threshold
        self.logs[timestamp]["language"] = self.language
        self.logs[timestamp]["temperature"] = self.temperature
        self.logs[timestamp]["max_token"] = self.max_token
        self.logs[timestamp]["doc_path"] = self.doc_path
        print(self.logs[timestamp])
        return
    
    
    def _add_result_log(self, timestamp:str, time:float):
        
        self.logs[timestamp]["time"] = time
        self.logs[timestamp]["doc_length"] = self.doc_length
        self.logs[timestamp]["doc_tokens"] = self.doc_tokens
        try:
            self.logs[timestamp]["docs"] = '\n\n'.join([doc.page_content for doc in self.docs])
            self.logs[timestamp]["doc_metadata"] = '\n\n'.join([doc.metadata['source'] + "    page: " + str(doc.metadata['page']) for doc in self.docs])
        except:
            self.logs[timestamp]["doc_metadata"] = "none"
            self.logs[timestamp]["docs"]  = '\n\n'.join([doc for doc in self.docs])
            
        
        self.logs[timestamp]["system_prompt"] = self.system_prompt

        print(self.logs[timestamp])
        return
    
    









class Doc_QA(atman):
    def __init__(self, embeddings:str = "openai:text-embedding-ada-002", chunk_size:int=1000\
        , model:str = "openai:gpt-3.5-turbo", verbose:bool = False, topK:int = 2, threshold:float = 0.2,\
        language:str = 'ch' , search_type:Union[str,Callable] = 'svm', record_exp:str = "", \
        system_prompt:str = "", max_token:int=3000, temperature:float=0.0):
        
        super().__init__(chunk_size, model, verbose, topK, threshold,\
        language , search_type, record_exp, system_prompt, max_token, temperature)
        ### set argruments ###
        self.doc_path = ""
        self.embeddings = embeddings

        

        ### set variables ###
        self.logs = {}
        self.model_obj = helper.handle_model(model, self.verbose, self.temperature)
        self.embeddings_obj = helper.handle_embeddings(embeddings, self.verbose)
        self.search_type = search_type
        self.db = None
        self.docs = []
        self.doc_tokens = 0
        self.doc_length = 0
        self.response = ""
        self.prompt = ""
        
        
    
    
    
    def get_response(self,doc_path:str, prompt:str, **kwargs):
        
        
        self._change_variables(**kwargs)
        self.doc_path = doc_path
        self.db = helper.create_chromadb(self.doc_path, self.verbose, self.embeddings_obj, self.embeddings, self.chunk_size)
        timestamp = datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S")
        self.timestamp_list.append(timestamp)
        start_time = time.time()
        if not self._check_db():
            return ""
        
        self._add_basic_log(timestamp, "get_response")
        self.logs[timestamp]["search_type"] = self.search_type_str
        self.logs[timestamp]["embeddings"] = self.embeddings

        
        ### start to get response ###
        self.docs, self.doc_tokens = search.get_docs(self.db, self.embeddings_obj, prompt, self.topK, self.threshold, self.language,\
            self.search_type, self.verbose, self.model_obj, self.max_token, self.logs[timestamp])
    
        if self.docs is None:
            
            print("\n\nNo Relevant Documents.\n\n")
            return ""
        self.doc_length = helper.get_docs_length(self.language, self.docs)
        chain = load_qa_chain(llm=self.model_obj, chain_type="stuff",verbose=self.verbose)
        
        ## format prompt ##
        self.prompt = prompts.format_sys_prompt(self.system_prompt, prompt)
        
        try:
            self.response = chain.run(input_documents=self.docs, question=self.prompt)
            self.response =  helper.sim_to_trad(self.response)
            #response = res.split("Finished chain.")
        except Exception as e:
            
            print(e)
            print("\n\nllm error\n\n")
            
            self.response = ""
        if self.verbose:
            print(self.response)
        
        
        end_time = time.time()
        self._add_result_log(timestamp, end_time-start_time)
        self.logs[timestamp]["prompt"] = self.prompt
        self.logs[timestamp]["response"] = self.response
        if self.record_exp != "":
            params =  format.handle_params(self.model, self.embeddings, self.chunk_size, self.search_type_str,\
                self.topK, self.threshold, self.language)   
            metrics = format.handle_metrics(self.doc_length, end_time - start_time, self.doc_tokens)
            table = format.handle_table(prompt, self.docs, self.response)
            aiido_upload(self.record_exp, params, metrics, table)
        
        
        return self.response
        
    def chain_of_thought(self, doc_path:str, prompt_list:list, **kwargs)->list:
        
        self._change_variables(**kwargs)

        self.doc_path = doc_path
        table = {}
        self.db = helper.create_chromadb(self.doc_path, self.verbose, self.embeddings_obj, self.embeddings, self.chunk_size)
        timestamp = datetime.datetime.now().strftime( "%Y/%m/%d, %H:%M:%S")
        self.timestamp_list.append(timestamp)
        start_time = time.time()
        if not self._check_db():
            return ""
        
        
        self._add_basic_log(timestamp, "chain_of_thought")
        self.logs[timestamp]["search_type"] = self.search_type_str
        self.logs[timestamp]["embeddings"] = self.embeddings
        
        chain = load_qa_chain(llm=self.model_obj, chain_type="stuff",verbose=self.verbose)
        
        self.doc_tokens = 0
        self.doc_length = 0
        pre_result = []
        self.response = []
        self.docs = []
        
        for i in range(len(prompt_list)):
            
            question = prompts.format_sys_prompt(self.system_prompt, prompt_list[i])
            docs, tokens = search.get_docs(self.db, self.embeddings_obj, prompt_list[i], self.topK, self.threshold, \
                self.language, self.search_type, self.verbose, self.model_obj, self.max_token, self.logs[timestamp])
            
            self.docs.extend(docs)
            self.doc_length += helper.get_docs_length(self.language, docs)
            self.doc_tokens += tokens
            if self.verbose:
                print(docs)
                
            try:
                response = chain.run(input_documents=docs + pre_result, question = question)
                response = helper.sim_to_trad(response)
                
            except Exception as e:
                
                print(e)
                print("\n\nllm error\n\n")
            
                response = ""
                
            if self.verbose:
                print(response)
            self.response.append(response)
            pre_result.append(Document(page_content=''.join(response)))
            
            new_table = format.handle_table(prompt_list[i], docs, response)
            for key in new_table:
                if key not in table:
                    table[key] = []
                table[key].append(new_table[key])
            
            
        end_time = time.time()
        
        self._add_result_log(timestamp, end_time-start_time)
        self.logs[timestamp]["prompt"] = self.prompt
        self.logs[timestamp]["response"] = self.response
        
        if self.record_exp != "":
            params =  format.handle_params(self.model, self.embeddings, self.chunk_size, self.search_type_str,\
                self.topK, self.threshold, self.language)   
            metrics = format.handle_metrics(self.doc_length, end_time - start_time, self.doc_tokens)
            aiido_upload(self.record_exp, params, metrics, table)
        
        
        return self.response