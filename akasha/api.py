from fastapi import FastAPI, Query, UploadFile, File
from typing import Optional, List, Dict, Any, Union, Generator
from pydantic import BaseModel
import akasha
import gc, torch
import os

app = FastAPI()


def clean():
    try:
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    except:
        pass


class InfoModel(BaseModel):
    prompt: str
    info: Union[str, list] = ""
    model: Optional[str] = "openai:gpt-3.5-turbo"
    system_prompt: Optional[str] = ""
    max_doc_len: Optional[int] = 1500
    temperature: Optional[float] = 0.0
    openai_config: Optional[Dict[str, Any]] = {}


class ConsultModel(BaseModel):
    doc_path: Union[str, List[str]]
    prompt: Union[str, List[Any]]
    chunk_size: Optional[int] = 1000
    model: Optional[str] = "openai:gpt-3.5-turbo"
    embedding_model: Optional[str] = "openai:text-embedding-ada-002"
    threshold: Optional[float] = 0.1
    search_type: Optional[str] = 'auto'
    system_prompt: Optional[str] = ""
    max_doc_len: Optional[int] = 1500
    temperature: Optional[float] = 0.0
    openai_config: Optional[Dict[str, Any]] = {}


class ConsultModelReturn(BaseModel):
    response: Union[str, List[str]]
    status: str
    logs: Dict[str, Any]
    timestamp: str
    warnings: List[str] = []


class SummaryModel(BaseModel):
    file_path: str
    summary_type: Optional[str] = "map_reduce"
    summary_len: Optional[int] = 500
    max_doc_len: Optional[int] = 1600
    model: Optional[str] = "openai:gpt-3.5-turbo"
    system_prompt: Optional[str] = ""
    openai_config: Optional[Dict[str, Any]] = {}


def load_openai(config: dict) -> bool:
    """delete old environment variable and load new one.

    Args:
        config (dict): dictionary may contain openai_key, azure_key, azure_base.

    Returns:
        bool: load success or not
    """
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    if "AZURE_API_BASE" in os.environ:
        del os.environ["AZURE_API_BASE"]
    if "AZURE_API_KEY" in os.environ:
        del os.environ["AZURE_API_KEY"]
    if "AZURE_API_TYPE" in os.environ:
        del os.environ["AZURE_API_TYPE"]
    if "AZURE_API_VERSION" in os.environ:
        del os.environ["AZURE_API_VERSION"]

    if "openai_key" in config and config["openai_key"] != "":
        os.environ["OPENAI_API_KEY"] = config["openai_key"]

        return True

    if ("azure_key" in config and "azure_base" in config
            and config["azure_key"] != "" and config["azure_base"] != ""):
        os.environ["AZURE_API_KEY"] = config["azure_key"]
        os.environ["AZURE_API_BASE"] = config["azure_base"]
        os.environ["AZURE_API_TYPE"] = "azure"
        os.environ["AZURE_API_VERSION"] = "2023-05-15"
        return True

    return False


@app.post("/get_response")
def get_response(user_input: ConsultModel):
    """ run get_response in akasha.Doc_QA, load openai config if needed.

    Args:
        user_input (ConsultModel): data input class used in get_response
        doc_path: Union[str, List[str]]
        prompt: Union[str, List[str]]
        chunk_size:Optional[int]=1000
        model:Optional[str] = "openai:gpt-3.5-turbo"
        threshold:Optional[float] = 0.1
        search_type:Optional[str] = 'auto'
        system_prompt:Optional[str] = ""
        max_doc_len:Optional[int]=1500
        temperature:Optional[float]=0.0
        openai_config:Optional[Dict[str, Any]] = {}


    Returns:
        dict: status, response, logs
    """
    if user_input.model.split(
            ':')[0] == "openai" or user_input.embedding_model.split(
                ':')[0] == "openai":
        if not load_openai(config=user_input.openai_config):
            return {
                'status': 'fail',
                'response': 'load openai config failed.\n\n'
            }
    try:
        clean()
        qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, threshold=user_input.threshold\
            , model=user_input.model, temperature=user_input.temperature, max_doc_len=user_input.max_doc_len,embeddings=user_input.embedding_model\
            ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt)

        response = qa.get_response(doc_path=user_input.doc_path,
                                   prompt=user_input.prompt,
                                   keep_logs=True)
    except Exception as e:
        err_message = e.__str__()
        response = None

    try:
        ig_files = qa.ignored_files
    except:
        ig_files = []

    ## get logs
    timesp = ''
    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        timesp = qa.timestamp_list[-1]
        if timesp in qa.logs:
            logs = qa.logs[timesp]

    if response == None:
        user_output = ConsultModelReturn(
            response=f"text generation encounter errors, {err_message}\n",
            status="fail",
            logs=logs,
            timestamp=timesp,
            warnings=ig_files)
    else:
        user_output = ConsultModelReturn(response=response,
                                         status="success",
                                         logs=logs,
                                         timestamp=timesp,
                                         warnings=ig_files)

    del qa.model_obj
    del qa
    clean()
    return user_output


@app.post("/ask_self")
def ask_self(user_input: InfoModel):
    """ run ask_self in akasha.Doc_QA, load openai config if needed.

    Args:
        user_input (InfoModel): data input class used in get_response
        info: Union[str, List[str]]
        prompt: str
        model:Optional[str] = "openai:gpt-3.5-turbo"
        system_prompt:Optional[str] = ""
        max_doc_len:Optional[int]=1500
        temperature:Optional[float]=0.0
        openai_config:Optional[Dict[str, Any]] = {}


    Returns:
        dict: status, response, logs
    """
    if user_input.model.split(':')[0] == "openai":
        if not load_openai(config=user_input.openai_config):
            return {
                'status': 'fail',
                'response': 'load openai config failed.\n\n'
            }
    try:
        clean()
        qa = akasha.Doc_QA(verbose=True, model=user_input.model, temperature=user_input.temperature, max_doc_len=user_input.max_doc_len,\
            system_prompt=user_input.system_prompt)

        response = qa.ask_self(prompt=user_input.prompt,
                               info=user_input.info,
                               keep_logs=True)

    except Exception as e:
        err_message = e.__str__()
        response = None

    try:
        ig_files = qa.ignored_files
    except:
        ig_files = []

    ## get logs
    timesp = ''
    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        timesp = qa.timestamp_list[-1]
        if timesp in qa.logs:
            logs = qa.logs[timesp]

    if response == None:
        user_output = ConsultModelReturn(
            response=f"text generation encounter errors, {err_message}\n",
            status="fail",
            logs=logs,
            timestamp=timesp,
            warnings=ig_files)
    else:
        user_output = ConsultModelReturn(response=response,
                                         status="success",
                                         logs=logs,
                                         timestamp=timesp,
                                         warnings=ig_files)

    del qa.model_obj
    del qa
    clean()
    return user_output


@app.post("/ask_whole_file")
def ask_whole_file(user_input: ConsultModel):
    """ run ask_whole_file" in akasha.Doc_QA, load openai config if needed.

    Args:
        user_input (ConsultModel): data input class used in get_response
        doc_path: str
        prompt: str
        chunk_size:Optional[int]=1000
        model:Optional[str] = "openai:gpt-3.5-turbo"
        threshold:Optional[float] = 0.1
        search_type:Optional[str] = 'auto'
        system_prompt:Optional[str] = ""
        max_doc_len:Optional[int]=1500
        temperature:Optional[float]=0.0
        openai_config:Optional[Dict[str, Any]] = {}


    Returns:
        dict: status, response, logs
    """

    if isinstance(user_input.doc_path, list):
        user_input.doc_path = user_input.doc_path[0]
    if isinstance(user_input.prompt, list):
        user_input.prompt = user_input.prompt[0]

    if user_input.model.split(
            ':')[0] == "openai" or user_input.embedding_model.split(
                ':')[0] == "openai":
        if not load_openai(config=user_input.openai_config):
            return {
                'status': 'fail',
                'response': 'load openai config failed.\n\n'
            }
    try:
        clean()
        qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, threshold=user_input.threshold\
            , model=user_input.model, temperature=user_input.temperature, max_doc_len=user_input.max_doc_len,embeddings=user_input.embedding_model\
            ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt)

        response = qa.ask_whole_file(file_path=user_input.doc_path,
                                     prompt=user_input.prompt,
                                     keep_logs=True)
    except Exception as e:
        err_message = e.__str__()
        response = None

    try:
        ig_files = qa.ignored_files
    except:
        ig_files = []

    ## get logs
    timesp = ''
    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        timesp = qa.timestamp_list[-1]
        if timesp in qa.logs:
            logs = qa.logs[timesp]

    if response == None:
        user_output = ConsultModelReturn(
            response=f"text generation encounter errors, {err_message}\n",
            status="fail",
            logs=logs,
            timestamp=timesp,
            warnings=ig_files)
    else:
        user_output = ConsultModelReturn(response=response,
                                         status="success",
                                         logs=logs,
                                         timestamp=timesp,
                                         warnings=ig_files)

    del qa.model_obj
    del qa
    clean()
    return user_output


@app.post("/get_summary")
def get_summary(user_input: SummaryModel):
    """ get summary from akasha.Summary, load openai config if needed.

    Args:
        file_path(str): single file path
        summary_type: Optional[str] = "map_reduce" : summary method, "map_reduce" or "refine".
        summary_len: Optional[int] = 500 : the length of summary result
        model: Optional[str] = "openai:gpt-3.5-turbo"
        system_prompt: Optional[str] = ""
        openai_config: Optional[Dict[str, Any]] = {}

    Returns:
        _type_: _description_
    """

    if user_input.model.split(':')[0] == "openai":
        if not load_openai(config=user_input.openai_config):
            return {
                'status': 'fail',
                'response': 'load openai config failed.\n\n'
            }
    try:
        clean()
        sum = akasha.summary.Summary(
            chunk_size=500,
            chunk_overlap=50,
            model=user_input.model,
            verbose=True,
            system_prompt=user_input.system_prompt,
            max_doc_len=user_input.max_doc_len,
            temperature=0.0,
        )

        response = sum.summarize_file(file_path=user_input.file_path,
                                      summary_type=user_input.summary_type,
                                      summary_len=user_input.summary_len,
                                      keep_logs=True)
    except Exception as e:
        err_message = e.__str__()
        response = None

    ## get logs
    timesp = ''
    if len(sum.timestamp_list) == 0:
        logs = {}
    else:
        timesp = sum.timestamp_list[-1]
        if timesp in sum.logs:
            logs = sum.logs[timesp]

    if response == None:
        user_output = ConsultModelReturn(
            response=f"text generation encounter errors, {err_message}\n",
            status="fail",
            logs=logs,
            timestamp=timesp)
    else:
        user_output = ConsultModelReturn(response=response,
                                         status="success",
                                         logs=logs,
                                         timestamp=timesp)
    del sum.model_obj
    del sum
    clean()

    return user_output
