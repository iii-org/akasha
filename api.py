from fastapi import FastAPI, Query, UploadFile, File
import threading
from routers.watchdefault import start_observer
import akasha
from routers import datasets, experts
import akasha.helper
import akasha.db
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import json, os
import api_utils as apu
import gc, torch
## if default_key.json exist, create a thread to keep checking if the key is valid
if Path("./config/default_key.json").exists():
    thread = threading.Thread(target=start_observer)
    thread.start()

app = FastAPI()
app.include_router(datasets.router)
app.include_router(experts.router)

OPENAI_CONFIG_PATH = "./config/openai/"


def clean():
    gc.collect()
    torch.cuda.empty_cache()


### data class ###
class UserBase(BaseModel):
    user_name: str


class ConsultModel(BaseModel):
    data_path: Union[str, List[str]]
    prompt: Union[str, List[Any]]
    chunk_size: Optional[int] = 1000
    model: Optional[str] = "openai:gpt-3.5-turbo"
    embedding_model: Optional[str] = "openai:text-embedding-ada-002"
    topK: Optional[int] = 3
    threshold: Optional[float] = 0.2
    search_type: Optional[str] = 'svm'
    system_prompt: Optional[str] = ""
    max_doc_len: Optional[int] = 1500
    temperature: Optional[float] = 0.0
    use_chroma: Optional[bool] = True
    openai_config: Optional[Dict[str, Any]] = {}


class ConsultModelReturn(BaseModel):
    response: Union[str, List[str]]
    status: str
    logs: Dict[str, Any]
    timestamp: str


class OpenAIKey(BaseModel):
    owner: Optional[str] = ""
    openai_key: Optional[str] = ""
    azure_key: Optional[str] = ""
    azure_base: Optional[str] = ""


@app.post("/regular_consult")
def regular_consult(user_input: ConsultModel):
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
        max_doc_len:Optional[int]=1500
        temperature:Optional[float]=0.0
        use_chroma:Optional[bool]=True
        openai_config:Optional[Dict[str, Any]] = {}


    Returns:
        dict: status, response, logs
    """
    if user_input.model.split(
            ':')[0] == "openai" or user_input.embedding_model.split(
                ':')[0] == "openai":
        if not apu.load_openai(config=user_input.openai_config):
            return {
                'status': 'fail',
                'response': 'load openai config failed.\n\n'
            }

    qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, topK=user_input.topK, threshold=user_input.threshold\
        , model=user_input.model, temperature=user_input.temperature, max_doc_len=user_input.max_doc_len,embeddings=user_input.embedding_model\
        ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt, use_chroma = user_input.use_chroma)

    response = qa.get_response(doc_path=user_input.data_path,
                               prompt=user_input.prompt)

    ## get logs
    timesp = ''
    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        timesp = qa.timestamp_list[-1]
        if timesp in qa.logs:
            logs = qa.logs[timesp]

    if response == None or response == "":
        user_output = ConsultModelReturn(
            response="Can not get response from get_resposne.\n",
            status="fail",
            logs=logs,
            timestamp=timesp)
    else:
        user_output = ConsultModelReturn(response=response,
                                         status="success",
                                         logs=logs,
                                         timestamp=timesp)

    del qa.model_obj
    del qa
    clean()
    return user_output


@app.post("/deep_consult")
def deep_consult(user_input: ConsultModel):
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
        max_doc_len:Optional[int]=1500
        temperature:Optional[float]=0.0
        use_chroma:Optional[bool]=True
        openai_config:Optional[Dict[str, Any]] = {}


    Returns:
        dict: status, response, logs
    """
    if user_input.model.split(
            ':')[0] == "openai" or user_input.embedding_model.split(
                ':')[0] == "openai":
        if not apu.load_openai(config=user_input.openai_config):
            return {
                'status': 'fail',
                'response': 'load openai config failed.\n\n'
            }


    qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, topK=user_input.topK, threshold=user_input.threshold\
        , model=user_input.model, temperature=user_input.temperature, max_doc_len=user_input.max_doc_len,embeddings=user_input.embedding_model\
        ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt, use_chroma=user_input.use_chroma)
    response = qa.chain_of_thought(doc_path=user_input.data_path,
                                   prompt_list=user_input.prompt)
    ## get logs
    timesp = ''

    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        timesp = qa.timestamp_list[-1]
        if timesp in qa.logs:
            logs = qa.logs[timesp]

    if response == None or response == [] or response == "":
        user_output = ConsultModelReturn(
            response="Can not get response from chain_of_thought.\n",
            status="fail",
            logs=logs,
            timestamp=timesp)
    else:
        user_output = ConsultModelReturn(response=response,
                                         status="success",
                                         logs=logs,
                                         timestamp=timesp)

    del qa.model_obj
    del qa
    clean()
    return user_output


@app.post("/openai/save")
def save_openai_key(user_input: OpenAIKey):
    """save openai key into config file

    Args:
        user_input (_type_, optional): _description_.
        openai_key : str
        azure_key : str
        azure_base : str
        owner : str
    """
    owner = user_input.owner
    save_path = Path(OPENAI_CONFIG_PATH) / (owner + "_" + "openai.json")
    if not apu.check_config(OPENAI_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}

    if save_path.exists():
        with open(save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    else:
        data = {}

    flag = False
    if user_input.openai_key.replace(' ', '') != "":
        data['openai_key'] = user_input.openai_key.replace(' ', '')
        flag = True
    if user_input.azure_base.replace(
            ' ', '') != "" and user_input.azure_key.replace(' ', '') != "":
        flag = True
        data['azure_key'] = user_input.azure_key.replace(' ', '')
        data['azure_base'] = user_input.azure_base.replace(' ', '')

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    if not flag:
        return {
            'status': 'fail',
            'response': 'can not save any openai configuration.\n\n'
        }
    return {
        'status': 'success',
        'response': 'save openai key successfully.\n\n'
    }


@app.get("/openai/test_openai")
def test_openai_key(user_input: OpenAIKey):
    """test openai key is valid or not

    Args:
        user_input (OpenAIKey): _description_
        openai_key : str
    Returns:
        dict: status, response
    """
    import openai
    openai.api_type = "open_ai"
    openai.api_version = None

    if openai.base_url != "https://api.openai.com/v1":
        openai.base_url = "https://api.openai.com/v1"

    openai_key = user_input.openai_key.replace(' ', '')
    if openai_key != "":
        client = openai.OpenAI(api_key=openai_key, base_url=openai.base_url)
        try:
            client.models.list()
        except Exception as e:
            print(e.__str__())
            return {
                'status': 'fail',
                'response': 'openai key is invalid.\n\n' + e.__str__()
            }
        else:
            return {
                'status': 'success',
                'response': 'openai key is valid.\n\n'
            }

    return {'status': 'fail', 'response': 'openai key empty.\n\n'}


@app.get("/openai/test_azure")
def test_azure_key(user_input: OpenAIKey):
    """test azure key is valid or not

    Args:
        user_input (OpenAIKey): _description_
        azure_key : str
        azure_base : str
    Returns:
        dict: status, response
    """
    import openai
    azure_key = user_input.azure_key.replace(' ', '')
    azure_base = user_input.azure_base.replace(' ', '')
    if azure_key != "" and azure_base != "":
        client = openai.AzureOpenAI(azure_endpoint=azure_base,
                                    api_key=azure_key,
                                    api_version="2023-05-15")

        try:
            client.models.list()
        except Exception as e:
            return {
                'status': 'fail',
                'response': 'azure openai key is invalid.\n\n' + e.__str__()
            }
        else:
            return {
                'status': 'success',
                'response': 'azure openai key is valid.\n\n'
            }

    return {
        'status': 'fail',
        'response': 'azure openai key or base url is empty.\n\n'
    }


@app.get("/openai/load_openai")
def load_openai_from_file(user_input: UserBase):
    """load openai key from config file

    Args:
        user_name (str): user name
    Returns:
        dict: status, response
    """
    user_name = user_input.user_name
    openai_config_file_path = (Path(OPENAI_CONFIG_PATH) /
                               (user_name + "_" + "openai.json")).__str__()
    ret = {"azure_key": "", "azure_base": "", "openai_key": ""}
    try:
        if Path(openai_config_file_path).exists():
            file_path = Path(openai_config_file_path)
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "azure_key" in data and "azure_base" in data:

                    ret["azure_key"] = data["azure_key"]
                    ret["azure_base"] = data["azure_base"]

                if "openai_key" in data:

                    ret["openai_key"] = data["openai_key"]
    except Exception as e:
        return {'status': 'fail', 'response': ret}

    return {'status': 'success', 'response': ret}


@app.get("/openai/choose")
def choose_openai_key(user_input: OpenAIKey):
    """choose valid openai key from session_state and config file

    Args:
        user_input (_type_, optional): _description_.
        openai_key : str
        azure_key : str
        azure_base : str
        owner : str
    """

    openai_config_file_path = (
        Path(OPENAI_CONFIG_PATH) /
        (user_input.owner + "_" + "openai.json")).__str__()
    openai_key, azure_key, azure_base = "", "", ""

    if 'show_api_setting' in os.environ and os.environ[
            'show_api_setting'] == 'True':
        if 'default_openai_key' in os.environ:
            openai_key = os.environ['default_openai_key']
        if 'default_azure_key' in os.environ:
            azure_key = os.environ['default_azure_key']
        if 'default_azure_base' in os.environ:
            azure_base = os.environ['default_azure_base']
    else:
        openai_key = user_input.openai_key
        azure_key = user_input.azure_key
        azure_base = user_input.azure_base

    config = apu.choose_openai_key(openai_config_file_path, openai_key,
                                   azure_key, azure_base)

    if len(config) == 0:
        return {
            'status': 'fail',
            'response': 'can not find any valid openai key.\n\n'
        }

    return {'status': 'success', 'response': config}


@app.get("/openai/is_default_api")
def is_default_api():
    """ check if os.environ['show_api_setting'] == 'True'"""

    if 'show_api_setting' in os.environ and os.environ[
            'show_api_setting'] == 'True':
        return {'status': 'success', 'response': True}

    return {'status': 'success', 'response': False}
