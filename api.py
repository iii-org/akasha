from fastapi import FastAPI, Query, UploadFile, File
import threading
from routers.watchdefault import start_observer
import akasha
import akasha.summary as summary
from routers import datasets, experts
import akasha.helper
import akasha.db
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union, Generator
from pathlib import Path
import json, os
import api_utils as apu
import gc, torch
import yaml
import logging, sys
from logging.handlers import TimedRotatingFileHandler
from fastapi.responses import StreamingResponse, Response
from langchain.llms.base import LLM
## if default_key.json exist, create a thread to keep checking if the key is valid
if not Path("./config").exists():
    os.mkdir("./config")
if Path("./config/default_key.json").exists():
    thread = threading.Thread(target=start_observer)
    thread.start()

if not Path(apu.get_accounts_path()).exists():
    with open(apu.get_accounts_path(), "w") as f:
        f.write("""cookie:
  expiry_days: 30
  key: random_signature_key
  name: random_cookie_name
credentials:
  usernames:
    cws:
      email: cws@gmail.com
      name: cws
      password: $2b$12$jCB8MeVqMc3jWDynjNyeVeLS8IWBduxnX362gLfJ1KIkeTPH9KYha
preauthorized:
  emails: []
""")


def loggings():
    if not Path("./logs").exists():
        os.mkdir("./logs")

    # Set up logging
    logger = logging.getLogger()  #__name__
    handler = TimedRotatingFileHandler('logs/app.log',
                                       when="midnight",
                                       interval=1,
                                       encoding='utf-8')
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    class StreamToLogger:
        """
        Fake file-like stream object that redirects writes to a logger instance.
        """

        def __init__(self, logger, log_level=logging.WARNING):
            self.logger = logger
            self.log_level = log_level

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())

        def flush(self):
            pass

    #sys.stdout = StreamToLogger(logger, logging.WARNING)
    #sys.stderr = StreamToLogger(logger, logging.ERROR)


loggings()

app = FastAPI()
app.include_router(datasets.router)
app.include_router(experts.router)

OPENAI_CONFIG_PATH = "./config/openai/"


def clean():
    try:
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    except:
        pass


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
    prompt_format_type: Optional[str] = "gpt"


class ChatModel(ConsultModel):
    history_messages: Optional[List[Dict]] = []


class ConsultModelReturn(BaseModel):
    response: Union[str, List[str]]
    status: str
    logs: Dict[str, Any]
    timestamp: str
    warnings: List[str] = []


class OpenAIKey(BaseModel):
    owner: Optional[str] = ""
    openai_key: Optional[str] = ""
    azure_key: Optional[str] = ""
    azure_base: Optional[str] = ""


class SummaryModel(BaseModel):
    file_path: str
    summary_type: Optional[str] = "map_reduce"
    summary_len: Optional[int] = 500
    model: Optional[str] = "openai:gpt-3.5-turbo"
    system_prompt: Optional[str] = ""
    openai_config: Optional[Dict[str, Any]] = {}


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
    try:
        clean()
        qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, threshold=user_input.threshold\
            , model=user_input.model, temperature=user_input.temperature, max_doc_len=user_input.max_doc_len,embeddings=user_input.embedding_model\
            ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt, use_chroma = user_input.use_chroma,prompt_format_type=user_input.prompt_format_type)

        response = qa.get_response(doc_path=user_input.data_path,
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


def run_llm(user_input: ConsultModel) -> Generator:

    try:
        clean()

        qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, threshold=user_input.threshold\
            , model=user_input.model, temperature=user_input.temperature, max_doc_len=user_input.max_doc_len,embeddings=user_input.embedding_model\
            ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt, use_chroma = user_input.use_chroma,\
                 prompt_format_type = user_input.prompt_format_type,)

        response_iter = qa.get_response(doc_path=user_input.data_path,
                                        prompt=user_input.prompt,
                                        stream=True)

        for response in response_iter:
            yield response

        ref_name = set()
        for doc in qa.docs:
            ref_name.add(doc.metadata['source'].split("/")[-1])
        doc_metadata = list(ref_name)
        yield json.dumps({"doc_metadata": doc_metadata})

        del qa.model_obj
        del qa
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        yield f"Error: {e}"
        gc.collect()
        torch.cuda.empty_cache()


def run_llm_chat(user_input: ChatModel) -> Generator:

    try:
        clean()

        message, chat_history_len = apu.retri_history_messages(
            user_input.history_messages,
            max_doc_len=user_input.max_doc_len // 2)

        qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, threshold=user_input.threshold\
            , model=user_input.model, temperature=user_input.temperature, embeddings=user_input.embedding_model\
            ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt, use_chroma = user_input.use_chroma,\
                 prompt_format_type = user_input.prompt_format_type,)
        response_iter = qa.get_response(doc_path=user_input.data_path,
                                        prompt=user_input.prompt,
                                        stream=True,
                                        history_messages=message)

        for response in response_iter:
            yield response

        ref_name = set()
        for doc in qa.docs:
            ref_name.add(doc.metadata['source'].split("/")[-1])
        doc_metadata = list(ref_name)
        yield json.dumps({"doc_metadata": doc_metadata})

        del qa.model_obj
        del qa
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        yield f"Error: {e}"
        gc.collect()
        torch.cuda.empty_cache()


@app.post("/regular_consult_stream")
def regular_consult_stream(user_input: ConsultModel):
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
            return Response(
                "load openai config failed.\n\n",
                status_code=500,
                media_type="text/event-plain",
            )

    try:

        return StreamingResponse(
            content=run_llm(user_input),
            media_type="text/event-stream",
        )

    except Exception as e:
        return Response(
            f"text generation encounter errors, {e.__str__()}\n",
            status_code=500,
        )


@app.post("/chat_stream")
def chat_stream(user_input: ChatModel):
    """user ask a question and use both history messages and documents to get response

    Args:
        user_input (ChatModel): data input class used in chat_stream and chat

    Returns:
        StreamingResponse: return the generator of response
    """

    if user_input.model.split(
            ':')[0] == "openai" or user_input.embedding_model.split(
                ':')[0] == "openai":
        if not apu.load_openai(config=user_input.openai_config):
            return {
                'status': 'fail',
                'response': 'load openai config failed.\n\n'
            }

    try:

        return StreamingResponse(
            content=run_llm_chat(user_input),
            media_type="text/event-stream",
        )

    except Exception as e:
        return Response(
            f"text generation encounter errors, {e.__str__()}\n",
            status_code=500,
        )


@app.post("/chat")
def chat(user_input: ChatModel):
    """user ask a question and use both history messages and documents to get response

    Args:
        user_input (ChatModel): data input class used in chat_stream and chat

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

    try:
        clean()
        message, chat_history_len = apu.retri_history_messages(
            user_input.history_messages,
            max_doc_len=user_input.max_doc_len // 2)

        qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, threshold=user_input.threshold\
            , model=user_input.model, temperature=user_input.temperature, embeddings=user_input.embedding_model\
            ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt, use_chroma = user_input.use_chroma,\
                 prompt_format_type = user_input.prompt_format_type,)
        response = qa.get_response(doc_path=user_input.data_path,
                                   prompt=user_input.prompt,
                                   stream=False,
                                   history_messages=message)

        ref_name = set()
        for doc in qa.docs:
            ref_name.add(doc.metadata['source'].split("/")[-1])

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

    logs['doc_metadata'] = list(ref_name)

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


@app.post("/deep_consult")
def deep_consult(user_input: ConsultModel):
    """load openai config and run chain_of_thought in akasha.Doc_QA

    Args:
        user_input (ConsultModel): data input class used in regular_consult and deep_consult
        data_path: Union[str, List[str]]
        prompt: Union[str, List[str]]
        chunk_size:Optional[int]=1000
        model:Optional[str] = "openai:gpt-3.5-turbo"
        topK:Optional[int] = -1 
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

    try:
        clean()
        qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, threshold=user_input.threshold\
            , model=user_input.model, temperature=user_input.temperature, max_doc_len=user_input.max_doc_len,embeddings=user_input.embedding_model\
            ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt, use_chroma=user_input.use_chroma)
        response = qa.chain_of_thought(doc_path=user_input.data_path,
                                       prompt_list=user_input.prompt,
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
    """load oprnai config and get summary from akasha.Summary

    Args:
        user_input (SummaryModel): _description_

    Returns:
        _type_: _description_
    """

    if user_input.model.split(':')[0] == "openai":
        if not apu.load_openai(config=user_input.openai_config):
            return {
                'status': 'fail',
                'response': 'load openai config failed.\n\n'
            }
    try:
        clean()
        sum = summary.Summary(
            chunk_size=500,
            chunk_overlap=50,
            model=user_input.model,
            verbose=True,
            system_prompt=user_input.system_prompt,
            max_doc_len=1600,
            temperature=0.0,
        )

        response = sum.summarize_file(file_path=user_input.file_path,
                                      summary_type=user_input.summary_type,
                                      summary_len=user_input.summary_len,
                                      keep_logs=True)
    except Exception as e:
        err_message = e.__str__()
        response = None
        logging.error(err_message)

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
            embedding = client.embeddings.create(
                input="<input>",
                model="text-embedding-ada-002"  # model = "deployment_name".
            )
            #print(client.models.list())
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


@app.get("/get_all_nicknames")
def get_all_nicknames(user_input: UserBase):
    """get all nicknames from database

    Args:
        user_input (UserBase): user name
    Returns:
        dict: status, response
    """
    username = user_input.user_name
    accounts_path = apu.get_accounts_path()

    try:
        with open(accounts_path, 'r') as f:
            accounts = yaml.safe_load(f)
            nicknames = {
                username: details.get('name')
                for username, details in accounts['credentials']
                ['usernames'].items()
            }

    except Exception as e:
        return {'status': 'fail', 'response': 'load account.yaml error.\n\n'}
    if not isinstance(nicknames, dict):
        return {'status': 'fail', 'response': 'can not find nickname.\n\n'}
    return {'status': 'success', 'response': nicknames}


@app.get("/get_nickname")
def get_nickname(user_input: UserBase):
    """get nickname from database

    Args:
        user_input (UserBase): user name
    Returns:
        dict: status, response
    """
    username = user_input.user_name
    accounts_path = apu.get_accounts_path()

    try:
        with open(accounts_path, 'r') as f:
            accounts = yaml.safe_load(f)
            nickname = accounts['credentials']['usernames'].get(username,
                                                                {}).get('name')
    except Exception as e:
        return {'status': 'fail', 'response': 'load account.yaml error.\n\n'}
    if nickname == None:
        return {'status': 'fail', 'response': 'can not find nickname.\n\n'}
    return {'status': 'success', 'response': nickname}
