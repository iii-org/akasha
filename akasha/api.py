from fastapi import FastAPI
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel
import akasha
import os

app = FastAPI()


class InfoModel(BaseModel):
    prompt: str
    info: Union[str, list] = ""
    model: Optional[str] = "openai:gpt-3.5-turbo"
    system_prompt: Optional[str] = ""
    max_input_tokens: Optional[int] = 3000
    temperature: Optional[float] = 0.0
    env_config: Optional[Dict[str, Any]] = {}


class webInfoModel(InfoModel):
    search_engine: str = "wiki"
    search_num: int = 5


class ConsultModel(BaseModel):
    data_source: Union[str, List[str]]
    prompt: Union[str, List[Any]]
    chunk_size: Optional[int] = 1000
    model: Optional[str] = "openai:gpt-3.5-turbo"
    embedding_model: Optional[str] = "openai:text-embedding-ada-002"
    threshold: Optional[float] = 0.0
    search_type: Optional[str] = "auto"
    system_prompt: Optional[str] = ""
    max_input_tokens: Optional[int] = 3000
    temperature: Optional[float] = 0.0
    env_config: Optional[Dict[str, Any]] = {}


class ConsultModelReturn(BaseModel):
    response: Union[str, List[str]]
    status: str
    logs: Dict[str, Any]
    timestamp: str
    warnings: List[str] = []


class SummaryModel(BaseModel):
    content: Union[str, list] = ""
    summary_type: Optional[str] = "map_reduce"
    summary_len: Optional[int] = 500
    max_input_tokens: Optional[int] = 3200
    model: Optional[str] = "openai:gpt-3.5-turbo"
    system_prompt: Optional[str] = ""
    env_config: Optional[Dict[str, Any]] = {}


OTHER_ENV = set(
    [
        "SERPER_API_KEY",
        "BRAVE_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
    ]
)


def load_env(config: dict) -> bool:
    """delete old environment variable and load new one.

    Args:
        config (dict): dictionary may contain openai_key, azure_key, azure_base.

    Returns:
        bool: load success or not
    """

    for key, val in config.items():
        if key in OTHER_ENV:
            os.environ[key] = val

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

    if "OPENAI_API_KEY" in config and config["OPENAI_API_KEY"] != "":
        os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

        return True

    if (
        "AZURE_API_KEY" in config
        and "AZURE_API_BASE" in config
        and config["AZURE_API_KEY"] != ""
        and config["AZURE_API_BASE"] != ""
    ):
        os.environ["AZURE_API_KEY"] = config["AZURE_API_KEY"]
        os.environ["AZURE_API_BASE"] = config["AZURE_API_BASE"]
        os.environ["AZURE_API_TYPE"] = "azure"
        os.environ["AZURE_API_VERSION"] = "2023-05-15"
        return True

    if (
        os.environ.get("ANTHROPIC_API_KEY", "") != ""
        or os.environ.get("GEMINI_API_KEY", "") != ""
    ):
        return True
    return False


@app.post("/RAG")
def RAG(user_input: ConsultModel):
    """run RAG for document search, load openai config if needed.

    Args:
        user_input (ConsultModel): data input class used in get_response
        data_source: Union[str, List[str]]
        prompt: Union[str, List[str]]
        chunk_size:Optional[int]=1000
        model:Optional[str] = "openai:gpt-3.5-turbo"
        threshold:Optional[float] (deprecated) = 0.0
        search_type:Optional[str] = 'auto'
        system_prompt:Optional[str] = ""
        temperature:Optional[float]=0.0
        env_config:Optional[Dict[str, Any]] = {}
        max_input_tokens: Optional[int] = 3000


    Returns:
        dict: status, response, logs
    """
    if user_input.model.split(":")[0] in [
        "openai",
        "gemini",
        "anthropic",
    ] or user_input.embedding_model.split(":")[0] in ["openai", "gemini", "anthropic"]:
        if not load_env(config=user_input.env_config):
            return {"status": "fail", "response": "load env config failed.\n\n"}
    try:
        qa = akasha.RAG(
            verbose=True,
            search_type=user_input.search_type,
            model=user_input.model,
            temperature=user_input.temperature,
            max_input_tokens=user_input.max_input_tokens,
            embeddings=user_input.embedding_model,
            chunk_size=user_input.chunk_size,
            system_prompt=user_input.system_prompt,
        )

        response = qa(
            data_source=user_input.data_source, prompt=user_input.prompt, keep_logs=True
        )
    except Exception as e:
        err_message = e.__str__()
        response = None

    try:
        ig_files = qa.ignored_files
    except Exception:
        ig_files = []

    ## get logs
    timesp = ""
    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        timesp = qa.timestamp_list[-1]
        if timesp in qa.logs:
            logs = qa.logs[timesp]

    if response is None:
        user_output = ConsultModelReturn(
            response=f"text generation encounter errors, {err_message}\n",
            status="fail",
            logs=logs,
            timestamp=timesp,
            warnings=ig_files,
        )
    else:
        user_output = ConsultModelReturn(
            response=response,
            status="success",
            logs=logs,
            timestamp=timesp,
            warnings=ig_files,
        )

    del qa.model_obj
    del qa
    return user_output


@app.post("/ask")
def ask(user_input: InfoModel):
    """run ask to ask llm, load openai config if needed.

    Args:
        user_input (InfoModel): data input class used in get_response
        info: Union[str, List[str]]
        prompt: str
        model:Optional[str] = "openai:gpt-3.5-turbo"
        system_prompt:Optional[str] = ""
        max_input_tokens:Optional[int]=3000
        temperature:Optional[float]=0.0
        env_config:Optional[Dict[str, Any]] = {}


    Returns:
        dict: status, response, logs
    """
    if user_input.model.split(":")[0] in ["openai", "gemini", "anthropic"]:
        if not load_env(config=user_input.env_config):
            return {"status": "fail", "response": "load env config failed.\n\n"}
    try:
        qa = akasha.ask(
            verbose=True,
            model=user_input.model,
            temperature=user_input.temperature,
            max_input_tokens=user_input.max_input_tokens,
            system_prompt=user_input.system_prompt,
        )

        response = qa(prompt=user_input.prompt, info=user_input.info, keep_logs=True)

    except Exception as e:
        err_message = e.__str__()
        response = None

    ig_files = []

    ## get logs
    timesp = ""
    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        timesp = qa.timestamp_list[-1]
        if timesp in qa.logs:
            logs = qa.logs[timesp]

    if response is None:
        user_output = ConsultModelReturn(
            response=f"text generation encounter errors, {err_message}\n",
            status="fail",
            logs=logs,
            timestamp=timesp,
            warnings=ig_files,
        )
    else:
        user_output = ConsultModelReturn(
            response=response,
            status="success",
            logs=logs,
            timestamp=timesp,
            warnings=ig_files,
        )

    del qa.model_obj
    del qa
    return user_output


@app.post("/summary")
def summary(user_input: SummaryModel):
    """get summary to summarize articles or files, load openai config if needed.

    Args:
        content(Union[str,List[str]]): file paths or urls, or text content
        summary_type: Optional[str] = "map_reduce" : summary method, "map_reduce" or "refine".
        summary_len: Optional[int] = 500 : the length of summary result
        model: Optional[str] = "openai:gpt-3.5-turbo"
        system_prompt: Optional[str] = ""
        env_config: Optional[Dict[str, Any]] = {}

    Returns:
        _type_: _description_
    """

    if user_input.model.split(":")[0] in ["openai", "gemini", "anthropic"]:
        if not load_env(config=user_input.env_config):
            return {"status": "fail", "response": "load env config failed.\n\n"}
    try:
        sum = akasha.summary(
            chunk_size=500,
            chunk_overlap=50,
            model=user_input.model,
            verbose=True,
            system_prompt=user_input.system_prompt,
            max_input_tokens=user_input.max_input_tokens,
            sum_type=user_input.summary_type,
            sum_len=user_input.summary_len,
            temperature=0.0,
        )

        response = sum(content=user_input.content, keep_logs=True)
    except Exception as e:
        err_message = e.__str__()
        response = None

    ## get logs
    timesp = ""
    if len(sum.timestamp_list) == 0:
        logs = {}
    else:
        timesp = sum.timestamp_list[-1]
        if timesp in sum.logs:
            logs = sum.logs[timesp]

    if response is None:
        user_output = ConsultModelReturn(
            response=f"text generation encounter errors, {err_message}\n",
            status="fail",
            logs=logs,
            timestamp=timesp,
        )
    else:
        user_output = ConsultModelReturn(
            response=response, status="success", logs=logs, timestamp=timesp
        )
    del sum.model_obj
    del sum

    return user_output


@app.post("/websearch")
def websearch(user_input: webInfoModel):
    """run websearch to use prompt search in web and answer the user question, load openai config if needed.

    Args:
        user_input (InfoModel): data input class used in get_response
        info: Union[str, List[str]]
        prompt: str
        model:Optional[str] = "openai:gpt-3.5-turbo"
        system_prompt:Optional[str] = ""
        max_input_tokens:Optional[int]=3000
        temperature:Optional[float]=0.0
        search_engine: str = "wiki"
        search_num: int = 5
        env_config:Optional[Dict[str, Any]] = {}


    Returns:
        dict: status, response, logs
    """

    load_env(config=user_input.env_config)

    try:
        qa = akasha.websearch(
            verbose=True,
            model=user_input.model,
            temperature=user_input.temperature,
            max_input_tokens=user_input.max_input_tokens,
            system_prompt=user_input.system_prompt,
            search_engine=user_input.search_engine,
            search_num=user_input.search_num,
        )

        response = qa(prompt=user_input.prompt, keep_logs=True)

    except Exception as e:
        err_message = e.__str__()
        response = None

    ig_files = []

    ## get logs
    timesp = ""
    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        timesp = qa.timestamp_list[-1]
        if timesp in qa.logs:
            logs = qa.logs[timesp]

    if response is None:
        user_output = ConsultModelReturn(
            response=f"text generation encounter errors, {err_message}\n",
            status="fail",
            logs=logs,
            timestamp=timesp,
            warnings=ig_files,
        )
    else:
        user_output = ConsultModelReturn(
            response=response,
            status="success",
            logs=logs,
            timestamp=timesp,
            warnings=ig_files,
        )

    del qa.model_obj
    del qa
    return user_output
