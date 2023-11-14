from fastapi import FastAPI, Query, UploadFile, File
from routers import datasets, experts
import akasha
import akasha.helper
import akasha.db
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import json
import streamlit_utils as stu
app = FastAPI()
app.include_router(datasets.router)
app.include_router(experts.router)



OPENAI_CONFIG_PATH = "./config/openai.json"

### data class ###

class ConsultModel(BaseModel):
    data_path: Union[str, List[str]]
    prompt: Union[str, List[str]]
    chunk_size:Optional[int]=1000
    model:Optional[str] = "openai:gpt-3.5-turbo"
    topK:Optional[int] = 3 
    threshold:Optional[float] = 0.2
    search_type:Optional[str] = 'svm'
    system_prompt:Optional[str] = ""
    max_token:Optional[int]=3000
    temperature:Optional[float]=0.0
    use_chroma:Optional[bool]=True
    openai_config:Optional[Dict[str, Any]] = {}

class ConsultModelReturn(BaseModel):
    response: Union[str, List[str]]
    status: str
    logs: Dict[str, Any]



class OpenAIKey(BaseModel):
    openai_key: Optional[str] = ""
    azure_key: Optional[str] = ""
    azure_base: Optional[str] = ""
    



@app.post("/regular_consult")
async def regular_consult(user_input: ConsultModel):
    """load openai config and run get_response in akasha.Doc_QA

    Args:
        user_input (ConsultModel): data input class used in regular_consult and deep_consult
        data_path: Union[str, List[str]]
        prompt: Union[str, List[str]]
        chunk_size:Optional[int]=1000
        model:Optional[str] = "openai:gpt-3.5-turbo"
        topK:Optional[int] = 3 
        threshold:Optional[float] = 0.2
        search_type:Optional[str] = 'svm'
        system_prompt:Optional[str] = ""
        max_token:Optional[int]=3000
        temperature:Optional[float]=0.0
        use_chroma:Optional[bool]=True
        openai_config:Optional[Dict[str, Any]] = {}


    Returns:
        dict: status, response, logs
    """
    if not stu.load_openai(openai_config=user_input.openai_config):
        return {'status': 'fail', 'response': 'load openai config failed.\n\n'}
     
    qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, topK=user_input.topK, threshold=user_input.threshold\
        , model=user_input.model, temperature=user_input.temperature, max_token=user_input.max_token\
        ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt, use_chroma = user_input.use_chroma)
    response = qa.get_response(doc_path=user_input.data_path, prompt = user_input.prompt)
    
    ## get logs
    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        logs =  qa.logs[qa.timestamp_list[-1]]
    
    
    if response == None or response == "":
        user_output = ConsultModelReturn(response="Can not get response from get_resposne.\n", status="fail", logs=logs)
    else:
        user_output = ConsultModelReturn(response=response, status="success", logs=logs)
    
    del qa    
    return user_output



@app.post("/deep_consult")
async def deep_consult(user_input: ConsultModel):
    """load openai config and run chain_of_thought in akasha.Doc_QA

    Args:
        user_input (ConsultModel): data input class used in regular_consult and deep_consult
        data_path: Union[str, List[str]]
        prompt: Union[str, List[str]]
        chunk_size:Optional[int]=1000
        model:Optional[str] = "openai:gpt-3.5-turbo"
        topK:Optional[int] = 3 
        threshold:Optional[float] = 0.2
        search_type:Optional[str] = 'svm'
        system_prompt:Optional[str] = ""
        max_token:Optional[int]=3000
        temperature:Optional[float]=0.0
        use_chroma:Optional[bool]=True
        openai_config:Optional[Dict[str, Any]] = {}


    Returns:
        dict: status, response, logs
    """
    if not stu.load_openai(openai_config=user_input.openai_config):
        return {'status': 'fail', 'response': 'load openai config failed.\n\n'}
    
    
    qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, topK=user_input.topK, threshold=user_input.threshold\
        , model=user_input.model, temperature=user_input.temperature, max_token=user_input.max_token\
        ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt, use_chroma=user_input.use_chroma)
    response = qa.chain_of_thought(doc_path=user_input.data_path, prompt_list = user_input.prompt)
    ## get logs
    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        logs =  qa.logs[qa.timestamp_list[-1]]
    
    
    if response == None or response == [] or response == "":
        user_output = ConsultModelReturn(response="Can not get response from chain_of_thought.\n", status="fail", logs=logs)
    else:
        user_output = ConsultModelReturn(response=response, status="success", logs=logs)
    
    del qa    
    return user_output






@app.post("/openai/save")
async def save_openai_key(user_input:OpenAIKey):
    """save openai key into config file

    Args:
        user_input (_type_, optional): _description_.
        openai_key : str
        azure_key : str
        azure_base : str
    """
    
    if not stu.check_config(""):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}

    save_path = Path(OPENAI_CONFIG_PATH)
    if save_path.exists():
        with open(save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
    else :
        data = {}
        
    flag = False
    if user_input.openai_key.replace(' ','') != "":
        data['openai_key'] = user_input.openai_key.replace(' ','')
        flag = True
    if user_input.azure_base.replace(' ','') != "" and user_input.azure_key.replace(' ','') != "":
        flag = True        
        data['azure_key'] = user_input.azure_key.replace(' ','')
        data['azure_base'] = user_input.azure_base.replace(' ','')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    if not flag:
        return {'status': 'fail', 'response': 'can not save any openai configuration.\n\n'}
    return {'status': 'success', 'response': 'save openai key successfully.\n\n'}
    



@app.get("/openai/test_openai")
async def test_openai_key(user_input:OpenAIKey):
    """test openai key is valid or not

    Args:
        user_input (OpenAIKey): _description_
        openai_key : str
    Returns:
        dict: status, response
    """
    import openai
    openai_key = user_input.openai_key.replace(' ','')
    if openai_key != "":
        openai.api_type = "open_ai"
        openai.api_key = openai_key
        try:
            openai.Model.list()
        except openai.error.AuthenticationError as e:
            return {'status': 'fail', 'response': 'openai key is invalid.\n\n'}
        else:
            return {'status': 'success', 'response': 'openai key is valid.\n\n'}
        
    return {'status': 'fail', 'response': 'openai key empty.\n\n'}




@app.get("/openai/test_azure")
async def test_azure_key(user_input:OpenAIKey):
    """test azure key is valid or not

    Args:
        user_input (OpenAIKey): _description_
        azure_key : str
        azure_base : str
    Returns:
        dict: status, response
    """
    import openai
    azure_key = user_input.azure_key.replace(' ','')
    azure_base = user_input.azure_base.replace(' ','')
    if azure_key != "" and azure_base != "":
        openai.api_type = "azure"
        openai.api_base = azure_base
        openai.api_version = "2023-05-15"
        openai.api_key = azure_key

        try:
            openai.Model.list()
        except openai.error.AuthenticationError as e:
            return {'status': 'fail', 'response': 'azure openai key is invalid.\n\n'}
        else:
            return {'status': 'success', 'response': 'azure openai key is valid.\n\n'}
        
    return {'status': 'fail', 'response': 'azure openai key or base url is empty.\n\n'}