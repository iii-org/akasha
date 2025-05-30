import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
import numpy as np
import jieba
import json, re, time

from pathlib import Path
import opencc
from typing import Callable, Union, Tuple, List, Generator
from langchain.schema import Document
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages.ai import AIMessage
# from langchain_core.messages import HumanMessage, SystemMessage
import os, traceback, logging
import shutil

from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI
from langchain_core.embeddings import Embeddings
import akasha.format as afr
import akasha.prompts
import tiktoken
from pathlib import Path
import os
from dotenv import dotenv_values
import uuid
from charset_normalizer import detect

jieba.setLogLevel(
    jieba.logging.INFO)  ## ignore logging jieba model information

cc = opencc.OpenCC("s2twp")


def del_path(path, tag="temp_c&r@md&"):
    p = Path(path)
    for file in p.glob("*"):
        if tag in file.name:
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)

    return


def is_path_exist(path: str) -> bool:
    try:
        des_path = Path(path)
        if not des_path.exists():
            raise FileNotFoundError("can not find the path")
    except FileNotFoundError as err:
        traceback.print_exc()
        print(err, path)
        return False
    return True


def _separate_name(name: str):
    """separate type:name by ':'

    Args:
        **name (str)**: string with format "type:name" \n

    Returns:
        (str, str): res_type , res_name
    """
    sep = name.split(":")

    if len(sep) > 2:
        res_type = sep[0].lower()
        res_name = ':'.join(sep[1:])
    elif len(sep) < 2:
        ### if the format type not equal to type:name ###
        res_type = sep[0].lower()
        res_name = ""
    else:
        res_type = sep[0].lower()
        res_name = sep[1]

    return res_type, res_name


def get_env_var(env_file: str = "") -> dict:
    """if env_file is not empty, get the environment variable from the file
        else get the environment variable from the os.environ

    Args:
        env_file (str, optional): the path of .env file. Defaults to "".

    Returns:
        dict: return the environment variable dictionary
    """

    require_env = ["OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_API_VERSION","OPEANI_API_TYPE", \
        "AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION", "AZURE_API_TYPE", "SERPER_API_KEY",\
            "GEMINI_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN","ANTHROPIC_API_KEY", "REMOTE_API_KEY"]
    if env_file == "" or not Path(env_file).exists():
        env_dict = {}
        os_env_dict = os.environ.copy()
        for req in require_env:
            if req in os_env_dict:
                env_dict[req] = os_env_dict[req]

    else:
        env_dict = dotenv_values(env_file)
    return env_dict


def _handle_azure_env(env_dict: dict) -> Tuple[str, str, str]:
    """from environment variable dictionary env_dict get the api_base, api_key, api_version

    Returns:
        (str, str, str): api_base, api_key, api_version
    """
    check_env, ret, count = ["BASE", "KEY", "VERSION"], ["", "", ""], 0
    try:
        for check in check_env:
            if f"AZURE_API_{check}" in env_dict:
                ret[count] = env_dict[f"AZURE_API_{check}"]
                #if "OPENAI_API_BASE" in os.environ:
                #os.environ.pop("OPENAI_API_BASE", None)
            elif f"OPENAI_API_{check}" in os.environ:
                ret[count] = env_dict[f"OPENAI_API_{check}"]
                if check == "BASE":
                    env_dict["AZURE_API_BASE"] = os.environ["OPENAI_API_BASE"]
                    env_dict.pop("OPENAI_API_BASE", None)
            else:
                if check == "VERSION":
                    ret[count] = "2023-05-15"
                else:
                    raise Exception(
                        f"can not find the openai {check} in environment variable.\n\n"
                    )
            count += 1
    except Exception as err:
        traceback.print_exc()
        print(err)

    return ret[0], ret[1], ret[2]


def handle_embeddings(embedding_name: str = "openai:text-embedding-ada-002",
                      verbose: bool = False,
                      env_file: str = "") -> Embeddings:
    """create model client used in document QA, default if openai "gpt-3.5-turbo"
        use openai:text-embedding-ada-002 as default.
    Args:
        **embedding_name (str)**: embeddings client you want to use.
            format is (type:name), which is the model type and model name.\n
            for example, "openai:text-embedding-ada-002", "huggingface:all-MiniLM-L6-v2".\n
        **logs (list)**: list that store logs\n
        **verbose (bool)**: print logs or not\n

    Returns:
        vars: embeddings client
    """
    if isinstance(embedding_name, Embeddings):

        return embedding_name

    if isinstance(embedding_name, Callable):

        from akasha.models.custom import custom_embed
        embeddings = custom_embed(func=embedding_name)
        if verbose:
            print("selected custom embedding.")
        return embeddings

    embedding_type, embedding_name = _separate_name(embedding_name)
    env_dict = get_env_var(env_file)
    if embedding_type in [
            "text-embedding-ada-002", "openai", "openaiembeddings"
    ]:
        import openai

        if ("AZURE_API_TYPE" in env_dict and env_dict["AZURE_API_TYPE"]
                == "azure") or ("OPENAI_API_TYPE" in env_dict
                                and env_dict["OPENAI_API_TYPE"] == "azure"):

            embedding_name = embedding_name.replace(".", "")
            api_base, api_key, api_version = _handle_azure_env(env_dict)
            embeddings = AzureOpenAIEmbeddings(model=embedding_name,
                                               azure_endpoint=api_base,
                                               api_key=api_key,
                                               api_version=api_version,
                                               validate_base_url=False)

        else:

            if "OPENAI_API_KEY" not in env_dict:
                raise Exception(
                    "can not find the OPENAI_API_KEY in environment variable.\n\n"
                )

            openai.api_type = "open_ai"
            embeddings = OpenAIEmbeddings(
                model=embedding_name,
                openai_api_base="https://api.openai.com/v1",
                api_key=env_dict["OPENAI_API_KEY"],
                openai_api_type="open_ai",
            )
        info = "selected openai embeddings.\n"

    elif embedding_type in [
            "huggingface",
            "huggingfaceembeddings",
            "transformers",
            "transformer",
            "hf",
    ]:

        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_name,
            model_kwargs={"trust_remote_code": True})
        info = "selected hugging face embeddings.\n"

    elif embedding_type in [
            "tf",
            "tensorflow",
            "tensorflowhub",
            "tensorflowhubembeddings",
            "tensorflowembeddings",
    ]:
        from langchain_community.embeddings import TensorflowHubEmbeddings

        embeddings = TensorflowHubEmbeddings()
        info = "selected tensorflow embeddings.\n"

    elif embedding_type in ["gemini", "gemi", "google"]:

        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_name, google_api_key=os.environ["GEMINI_API_KEY"])
        info = "selected gemini embeddings.\n"

    else:
        import openai

        if ("AZURE_API_TYPE" in env_dict and env_dict["AZURE_API_TYPE"]
                == "azure") or ("OPENAI_API_TYPE" in env_dict
                                and env_dict["OPENAI_API_TYPE"] == "azure"):

            embedding_name = embedding_name.replace(".", "")
            api_base, api_key, api_version = _handle_azure_env(env_dict)
            embeddings = AzureOpenAIEmbeddings(model=embedding_name,
                                               azure_endpoint=api_base,
                                               api_key=api_key,
                                               api_version=api_version,
                                               validate_base_url=False)

        else:

            if "OPENAI_API_KEY" not in env_dict:
                raise Exception(
                    "can not find the OPENAI_API_KEY in environment variable.\n\n"
                )
            openai.api_type = "open_ai"

            embeddings = OpenAIEmbeddings(
                model=embedding_name,
                openai_api_base="https://api.openai.com/v1",
                api_key=env_dict["OPENAI_API_KEY"],
                openai_api_type="open_ai",
            )

        info = "can not find the embeddings, use openai as default.\n"

    if verbose:
        print(info)
    return embeddings


def handle_embeddings_and_name(embed: Union[
    str, Embeddings, Callable] = "openai:text-embedding-ada-002",
                               verbose: bool = False,
                               env_file: str = "") -> Tuple[Embeddings, str]:
    """get the embeddings object and embed name

    Args:
        embed (_type_, optional): _description_. Defaults to "openai:text-embedding-ada-002".
        verbose (bool, optional): _description_. Defaults to False.
        env_file (str, optional): _description_. Defaults to "".

    Returns:
        Tuple[Embeddings, str]: _description_
    """
    if isinstance(embed, Embeddings):
        return embed, _decide_embedding_type(embed)

    if callable(embed):
        model_name = embed.__name__
    else:
        model_name = embed

    model_obj = handle_embeddings(embed, verbose, env_file)

    return model_obj, model_name


def handle_model(model_name: Union[str, Callable] = "openai:gpt-3.5-turbo",
                 verbose: bool = False,
                 temperature: float = 0.0,
                 max_output_tokens: int = 1024,
                 env_file: str = "") -> BaseLanguageModel:
    """create model client used in document QA, default if openai "gpt-3.5-turbo"

    Args:
       ** model_name (str)**: open ai model name like "gpt-3.5-turbo","text-davinci-003", "text-davinci-002"\n
        **logs (list)**: list that store logs\n
        **verbose (bool)**: print logs or not\n

    Returns:
        vars: model client
    """
    if isinstance(model_name, BaseLanguageModel):
        return model_name

    if isinstance(model_name, Callable):

        from akasha.models.custom import custom_model

        model = custom_model(func=model_name, temperature=temperature)
        if verbose:
            print("selected custom model.")
        return model

    model_type, model_name = _separate_name(model_name)
    env_dict = get_env_var(env_file)
    if model_type in ["remote", "server", "tgi", "text-generation-inference"]:

        from akasha.models.remo import remote_model

        remote_api_key = "123"
        remote_model_name = "remote_model"
        base_url = model_name
        if "REMOTE_API_KEY" in env_dict:
            remote_api_key = env_dict["REMOTE_API_KEY"]
        if "@" in model_name:
            base_url, remote_model_name = model_name.split("@")

        info = f"selected remote model. \n"
        model = remote_model(base_url,
                             temperature,
                             api_key=remote_api_key,
                             model_name=remote_model_name,
                             max_output_tokens=max_output_tokens)

    elif model_type in ["google", "gemini", "gemi"]:

        from akasha.models.gemi import gemini_model
        if "GEMINI_API_KEY" not in env_dict:
            raise Exception(
                "can not find the GEMINI_API_KEY in environment variable.\n\n")
        info = f"selected gemini model. \n"
        model = gemini_model(model_name=model_name,
                             api_key=env_dict["GEMINI_API_KEY"],
                             temperature=temperature,
                             max_output_tokens=max_output_tokens)

    elif model_type in ["anthropic", "anthropicai", "claude", "anthro"]:
        if "ANTHROPIC_API_KEY" not in env_dict:
            raise Exception(
                "can not find the ANTHROPIC_API_KEY in environment variable.\n\n"
            )
        info = f"selected anthropic model. \n"

        from akasha.models.anthro import anthropic_model
        model = anthropic_model(model_name=model_name,
                                api_key=env_dict["ANTHROPIC_API_KEY"],
                                temperature=temperature,
                                max_output_tokens=max_output_tokens)

    elif (model_type
          in ["llama-cpu", "llama-gpu", "llama", "llama2", "llama-cpp"]
          and model_name != ""):

        from akasha.models.llamacpp2 import LlamaCPP

        model = LlamaCPP(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        info = "selected llama-cpp model\n"
    elif model_type in [
            "huggingface",
            "huggingfacehub",
            "transformers",
            "transformer",
            "huggingface-hub",
            "hf",
    ]:
        from akasha.models.hf import hf_model

        model = hf_model(model_name=model_name,
                         env_dict=env_dict,
                         temperature=temperature,
                         max_output_tokens=max_output_tokens)
        info = f"selected huggingface model {model_name}.\n"

    elif model_type in ["chatglm", "chatglm2", "glm"]:
        from akasha.models.chglm import chatGLM

        model = chatGLM(model_name=model_name,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens)
        info = f"selected chatglm model {model_name}.\n"

    elif model_type in ["lora", "peft"]:

        from akasha.models.gtq import peft_Llama2

        model = peft_Llama2(model_name_or_path=model_name,
                            temperature=temperature)
        info = f"selected peft model {model_name}.\n"

    elif model_type in ["gptq"]:
        if model_name.lower().find("taiwan-llama") != -1:

            from akasha.models.gtq import TaiwanLLaMaGPTQ

            model = TaiwanLLaMaGPTQ(model_name_or_path=model_name,
                                    temperature=temperature)

        else:

            from akasha.models.gtq import gptq
            model = gptq(
                model_name_or_path=model_name,
                temperature=temperature,
                bit4=True,
                max_token=4096,
            )
        info = f"selected gptq model {model_name}.\n"
    else:
        if model_type not in ["openai", "gpt-3.5", "gpt"]:
            info = f"can not find the model {model_type}:{model_name}, use openai as default.\n"
            model_name = "gpt-3.5-turbo"
            print(info)
        import openai

        if verbose:
            call_back = [StreamingStdOutCallbackHandler()]
        else:
            call_back = None

        if ("AZURE_API_TYPE" in env_dict and env_dict["AZURE_API_TYPE"]
                == "azure") or ("OPENAI_API_TYPE" in env_dict
                                and env_dict["OPENAI_API_TYPE"] == "azure"):
            model_name = model_name.replace(".", "")
            api_base, api_key, api_version = _handle_azure_env(env_dict)

            model = AzureChatOpenAI(
                model=model_name,
                deployment_name=model_name,
                temperature=temperature,
                azure_endpoint=api_base,
                api_key=api_key,
                api_version=api_version,
                validate_base_url=False,
                streaming=True,
                callbacks=call_back,
            )
        else:
            if "OPENAI_API_KEY" not in env_dict:
                raise Exception(
                    "can not find the OPENAI_API_KEY in environment variable.\n\n"
                )
            openai.api_type = "open_ai"
            model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=env_dict["OPENAI_API_KEY"],
                streaming=True,
                callbacks=call_back,
            )
        info = f"selected openai model {model_name}.\n"
    if verbose:
        print(info)

    return model


def handle_model_and_name(model: Union[
    str, Callable, BaseLanguageModel] = "openai:gpt-3.5-turbo",
                          verbose: bool = False,
                          temperature: float = 0.0,
                          max_output_tokens: int = 1024,
                          env_file: str = "") -> Tuple[BaseLanguageModel, str]:
    """get the model object and model name

    Args:
        model (_type_, optional): _description_. Defaults to "openai:gpt-3.5-turbo".
        verbose (bool, optional): _description_. Defaults to False.
        temperature (float, optional): _description_. Defaults to 0.0.
        max_output_tokens (int, optional): _description_. Defaults to 1024.
        env_file (str, optional): _description_. Defaults to "".

    Returns:
        Tuple[BaseLanguageModel, str]: _description_
    """
    if isinstance(model, BaseLanguageModel):
        return model, model._llm_type

    if callable(model):
        model_name = model.__name__
    else:
        model_name = model

    model_obj = handle_model(model, verbose, temperature, max_output_tokens,
                             env_file)

    return model_obj, model_name


def handle_search_type(search_type: Union[str, BaseLanguageModel, Embeddings],
                       verbose: bool = False) -> str:

    if isinstance(search_type, BaseLanguageModel):
        search_type_str = search_type._llm_type

    elif isinstance(search_type, Embeddings):
        search_type_str = _decide_embedding_type(search_type)

    elif callable(search_type):
        search_type_str = search_type.__name__

    else:
        search_type_str = search_type

    # if verbose:
    #     print("search type is :", search_type_str)

    return search_type_str


def get_doc_length(language: str, text: str) -> int:
    """calculate the length of terms in a giving Document

    Args:
        **language (str)**: 'ch' for chinese and 'en' for others, default 'ch'\n
        **doc (Document)**: Document object\n

    Returns:
        doc_length: int Docuemtn length
    """

    if "chinese" in afr.language_dict[language]:
        doc_length = len(list(jieba.cut(text)))
    else:
        doc_length = len(text.split())
    return doc_length


def get_docs_length(language: str, docs: list) -> int:
    """calculate the total length of terms in giving documents

    Args:
        language (str): 'ch' for chinese and 'en' for others, default 'ch'\n
        docs (list): list of Documents\n

    Returns:
        docs_length: int total Document length
    """
    docs_length = 0
    for doc in docs:
        docs_length += get_doc_length(language, doc.page_content)
    return docs_length


def get_question_from_file(path: str, question_style: str):
    """load questions from file and save the questions into lists.
    a question list include the question, mutiple options, and the answer (the number of the option),
      and they are all separate by space in the file.

    Args:
        **path (str)**: path of the question file\n

    Returns:
        list: list of question list
    """
    f_path = Path(path)
    with f_path.open(mode="r", encoding="utf-8") as file:
        content = file.read()
    questions = []
    answers = []

    if question_style.lower() == "essay":
        content = content.split("\n\n")
        for i in range(len(content)):
            if content[i] == "":
                continue

            try:
                process = "".join(content[i].split("問題：")).split("答案：")
                if len(process) < 2:
                    raise SyntaxError("Question Format Error")
            except:
                process = "".join(content[i].split("問題:")).split("答案:")
                if len(process) < 2:
                    continue

            questions.append(process[0])
            answers.append(process[1])
        return questions, answers

    for con in content.split("\n"):
        if con == "":
            continue
        words = [word for word in con.split("\t") if word != ""]

        questions.append(words[:-1])
        answers.append(words[-1])

    return questions, answers


def extract_result(response: str):
    """to prevent the output of llm format is not what we want, try to extract the answer (digit) from the llm output

    Args:
        **response (str)**: llm output\n

    Returns:
        int: digit of answer
    """
    try:
        res = extract_json(response)
        #res = str(json.loads(response)["ans"]).replace(" ", "")
        if res == None:
            raise Exception("can not find the json format in the response")
        res = res["ans"]
    except:
        res = -1
        for c in response:
            if c.isdigit():
                res = c

                break
    return res


def extract_json(s: str) -> Union[dict, None]:
    """parse the JSON part of the string

    Args:
        s (str): string that contains JSON part

    Returns:
        Union[dict, None]: return the JSON part of the string, if not found return None
    """
    # Use a regular expression to find the JSON part of the string
    match = re.search(r'\{.*\}', s, re.DOTALL)
    stack = []
    start = 0
    s = s.replace("\n", " ").replace("\r", " ").strip()
    if match is None:
        return None

    s = match.group()
    for i, c in enumerate(s):
        if c == '{':
            stack.append(i)
        elif c == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    try:
                        json_part = s[start:i + 1]
                        json_part = json_part.replace("\n", "")
                        json_part = json_part.replace("'", '"')
                        # Try to parse the JSON part of the string
                        return json.loads(json_part)
                    except json.JSONDecodeError:
                        traceback.print_exc()
                        print(s[start:i + 1])
                        print(
                            "The JSON part of the string is not well-formatted"
                        )
                        return None
    return None


def get_all_combine(
    embeddings_list: list,
    chunk_size_list: list,
    model_list: list,
    search_type_list: list,
) -> list:
    """record all combinations of giving lists

    Args:
        **embeddings_list (list)**: list of embeddings(str)\n
        **chunk_size_list (list)**: list of chunk sizes(int)\n
        **model_list (list)**: list of models(str)\n
        **search_type_list (list)**: list of search types(str)\n

    Returns:
        list: list of tuples of all different combinations
    """
    res = []
    for embed in embeddings_list:
        for chk in chunk_size_list:
            for mod in model_list:
                for st in search_type_list:
                    res.append((embed, chk, mod, st))

    return res


def get_best_combination(result_list: list, idx: int) -> list:
    """input list of tuples and find the greatest tuple based on score or cost-effective (index 0 or index 1)
    tuple looks like (score, cost-effective, embeddings, chunk size, model, topK, search type)

    Args:
        **result_list (list)**: list of tuples that save the information of running experiments\n
        **idx (int)**: the index used to find the greatest result 0 is based on score and 1 is based on cost-effective\n

    Returns:
        list: return list of tuples that have same highest criteria
    """
    res = []
    sorted_res = sorted(result_list, key=lambda x: x[idx], reverse=True)
    max_score = sorted_res[0][idx]
    for tup in sorted_res:
        if tup[idx] < max_score:
            break
        res_str = ("embeddings: " + tup[-4] + ", chunk size: " + str(tup[-3]) +
                   ", model: " + tup[-2] + ", search type: " + tup[-1] + "\n")
        print(res_str)
        res.append(tup[-4:])

    return res


def sim_to_trad(text: str) -> str:
    """convert simplified chinese to traditional chinese

    Args:
        **text (str)**: simplified chinese\n

    Returns:
        str: traditional chinese
    """
    global cc
    return cc.convert(text)


def _get_text(
        texts: list,
        previous_summary: str,
        i: int,
        max_input_tokens: int,
        model_name: str = "openai:gpt-3.5-turbo") -> Tuple[int, str, int]:
    """used in summary, combine chunks of texts into one chunk that can fit into llm model

    Args:
        texts (list): chunks of texts
        previous_summary (str): _description_
        i (int): start from i-th chunk
        max_input_tokens (int): the max tokens we want to fit into llm model at one time
        model_name (str): model name(to calculate tokens) default "openai:gpt-3.5-turbo"\n

    Returns:
        (int, str, int): return the total tokens of combined chunks, combined chunks of texts, and the index of next chunk
    """
    cur_count = myTokenizer.compute_tokens(previous_summary, model_name)
    words_len = myTokenizer.compute_tokens(texts[i], model_name)
    cur_text = ""
    while cur_count + words_len < max_input_tokens and i < len(texts):
        cur_count += words_len
        cur_text += texts[i] + "\n"
        i += 1
        if i < len(texts):
            words_len = myTokenizer.compute_tokens(texts[i], model_name)

    return cur_count, cur_text, i


def call_model(
    model: BaseLanguageModel,
    input_text: Union[str, list],
) -> str:
    """call llm model and return the response

    Args:
        model (BaseLanguageModel): llm model
        input_text (str): the input_text that send to llm model
        prompt_type (str): the type of prompt, default "gpt"

    Returns:
        str: llm response
    """

    ### for openai, change system prompt and prompt into system meg and human meg ###

    response = ""
    print_flag = True

    model, model_name = handle_model_and_name(model)

    try:
        try:
            model_type = model._llm_type
        except:
            print_flag = False
            model_type = "unknown"

        if ("openai" in model_type):
            print_flag = False
            response = model.invoke(input_text)

        elif "remote" in model_type:
            print_flag = False
            response = model._call(input_text)
        else:
            try:
                response = model._call(input_text)
            except:
                response = model._generate(input_text)

        if isinstance(response, AIMessage):
            response = response.content
            if isinstance(response, dict):
                response = response.__str__()
            if isinstance(response, list):
                response = '\n'.join(response)

        if ("huggingface" in model_type) or ("llama cpp" in model_type) or (
                "gemini" in model_type) or ("anthropic" in model_type):
            print_flag = False

        if response is None or response == "":
            print_flag = False
            raise Exception("LLM response is empty.")

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(trace_text + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n")
        raise e

    if print_flag:
        print("llm response:", "\n\n" + response)

    if isinstance(response, str):
        response = sim_to_trad(response)

    return response


def call_batch_model(
    model: BaseLanguageModel,
    input_text: list,
) -> List[str]:
    """call llm model in batch and return the response 

    Args:
        model (BaseLanguageModel): llm model
        input_text: list

    Returns:
        str: llm response
    """

    ### check the input prompt and system prompt ###
    if isinstance(input_text, str):
        input_text = [input_text]

    response = ""
    responses = []
    model, model_name = handle_model_and_name(model)

    try:

        response = model.batch(input_text)
        for res in response:
            if isinstance(res, AIMessage):
                res = res.content
            if isinstance(res, dict):
                res = res.__str__()
            if isinstance(res, list):
                res = '\n'.join(res)
            responses.append(res)

        if response is None or response == "" or ''.join(responses) == "":
            print_flag = False
            raise Exception("LLM response is empty.")

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(trace_text + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n")
        raise e

    # if print_flag:
    #     print("llm response:", "\n\n" + response)

    return responses


def call_stream_model(
    model: BaseLanguageModel,
    input_text: Union[str, list],
) -> Generator[str, None, None]:
    """call llm model and yield the response

    Args:
        model (BaseLanguageModel): llm model
        input_text (str): the input_text that send to llm model
        prompt_type (str): the type of prompt, default "gpt"

    Returns:
        str: llm response
    """

    ### for openai, change system prompt and prompt into system meg and human meg ###

    response = None
    texts = ""
    model, model_name = handle_model_and_name(model)
    try:

        try:
            response = model.stream(input_text)
        except:
            response = model._call(input_text)

        for r in response:
            if isinstance(r, AIMessage):
                r = r.content
                if isinstance(r, dict):
                    r = r.__str__()
                if isinstance(r, list):
                    r = '\n'.join(r)
            texts += r
            yield sim_to_trad(r)

        if texts == "":
            yield "ERROR! LLM response is empty.\n\n"

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(trace_text + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n")
        yield e


def call_image_model(
    model: BaseLanguageModel,
    input_text: Union[str, list],
) -> str:

    response = ""
    print_flag = True
    model, model_name = handle_model_and_name(model)
    try:

        model_type = model._llm_type

        if ("openai" in model_type) or ("remote" in model_type):
            print_flag = False
            response = model.invoke(input_text)

        else:

            try:
                response = model.call_image(input_text)
            except:
                response = model._generate(input_text)

        if isinstance(response, AIMessage):
            response = response.content
            if isinstance(response, dict):
                response = response.__str__()
            if isinstance(response, list):
                response = '\n'.join(response)

        if response is None or response == "":
            print_flag = False
            raise Exception("LLM response is empty.")

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(trace_text + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n")
        raise e

    response = sim_to_trad(response)

    if print_flag:
        print("llm response:", "\n\n" + response)
    return response


def get_non_repeat_rand_int(vis: set, num: int, doc_range: int):
    temp = np.random.randint(num)
    if len(vis) >= num // 2:
        vis = set()
    if (temp not in vis) and (temp + doc_range - 1 not in vis):
        for i in range(doc_range):
            vis.add(temp + i)
        return temp
    return get_non_repeat_rand_int(vis, num, doc_range)


def get_text_md5(text):
    import hashlib

    md5_hash = hashlib.md5(text.encode()).hexdigest()

    return md5_hash


def image_to_base64(image_path: str) -> str:
    """convert image to base64 string

    Args:
        image_path (str): path of image

    Returns:
        str: base64 string
    """
    import base64

    with open(image_path, "rb") as img_file:
        img_str = base64.b64encode(img_file.read())
    return img_str.decode("utf-8")


def call_translator(model_obj: BaseLanguageModel,
                    texts: str,
                    prompt_format_type: str = "auto",
                    language: str = "zh") -> str:
    """translate texts to target language

    Args:
        model_obj (BaseLanguageModel): LLM that used to translate
        texts (str): texts that need to be translated
        prompt_format_type (str, optional): system prompt format. Defaults to "auto".
        language (str, optional): target language. Defaults to "zh".

    Returns:
        str: translated texts
    """
    model_obj, model_name = handle_model_and_name(model_obj)
    sys_prompt = akasha.prompts.default_translate_prompt(language)
    prod_prompt = akasha.prompts.format_sys_prompt(sys_prompt, texts,
                                                   prompt_format_type,
                                                   model_name)

    response = call_model(model_obj, prod_prompt)

    return response


def call_JSON_formatter(
    model_obj: BaseLanguageModel,
    texts: str,
    keys: Union[str, list] = "",
    prompt_format_type: str = "auto",
) -> Union[dict, None]:
    """use LLM to transfer texts into JSON format

    Args:
        model_obj (BaseLanguageModel): LLM that used to transfer
        texts (str): texts that need to be transferred
        keys (Union[str, list], optional): keys name of output dictionary. Defaults to "".
        prompt_format_type (str, optional): system prompt format. Defaults to "auto".

    Returns:
        Union[dict, None]: return the JSON part of the string, if not found return None
    """

    if keys == "":
        sys_prompt = "Format the following TEXTS into a single JSON instance that conforms to the JSON schema."
    elif isinstance(keys, str):
        keys = [keys]

    if keys != "":
        sys_prompt = f"Format the following TEXTS into a single JSON instance that conforms to the JSON schema which includes: {', '.join(keys)}\n\n"

    model_obj, model_name = handle_model_and_name(model_obj)

    prod_prompt = akasha.prompts.format_sys_prompt(sys_prompt,
                                                   "TEXTS: " + texts,
                                                   prompt_format_type,
                                                   model_name)

    response = call_model(model_obj, prod_prompt)
    return extract_json(response)


def retri_history_messages(
    messages: list,
    pairs: int = 10,
    max_input_tokens: int = 1500,
    model_name: str = "openai:gpt-3.5-turbo",
    role1: str = "User",
    role2: str = "Assistant",
) -> Tuple[str, int]:
    """from messages dict list, get pairs of user question and assistant response from most recent and not exceed max_input_tokens and pairs, and return the text with total length

    Args:
        messages (list): history messages list, each index is a dict with keys: role("user", "assistant"), content(content of message)
        pairs (int, optional): the maximum number of messages. Defaults to 10.
        max_input_tokens (int, optional): the maximum number of messages tokens. Defaults to 1500.
        model_name (str, optional): model name. Defaults to "openai:gpt-3.5-turbo".
        role1 (str, optional): role1 name. Defaults to "User".
        role2 (str, optional): role2 name. Defaults to "Assistant".

    Returns:
        Tuple[str, int]: return the text with total length.
    """
    cur_len = 0
    count = 0
    ret = []
    splitter = '\n----------------\n'

    for i in range(len(messages) - 1, -1, -2):
        if count >= pairs:
            break
        if (messages[i]["role"] != role2) or (messages[i - 1]["role"]
                                              != role1):
            i += 1
            continue
        texta = f"{role2}: " + messages[i]["content"].replace('\n', '') + "\n"
        textq = f"{role1}: " + messages[i - 1]["content"].replace(
            '\n', '')  # {(i+1)//2}.
        len_texta = myTokenizer.compute_tokens(texta, model_name)
        len_textq = myTokenizer.compute_tokens(textq, model_name)

        if cur_len + len_texta > max_input_tokens:
            break
        cur_len += len_texta
        ret.append(texta)

        if cur_len + len_textq > max_input_tokens:
            break
        cur_len += len_textq
        ret.append(textq)

        count += 1

    if count == 0:
        return "", 0

    ret.reverse()
    ret_str = splitter + ''.join(ret) + splitter

    return ret_str, cur_len


def _decide_embedding_type(embeddings: Embeddings) -> str:

    if isinstance(embeddings, OpenAIEmbeddings) or isinstance(
            embeddings, AzureOpenAIEmbeddings):
        return "openai:" + embeddings.model
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        if isinstance(embeddings, GoogleGenerativeAIEmbeddings):
            return "gemini:" + embeddings.model

        from akasha.models.custom import custom_embed
        if isinstance(embeddings, custom_embed):
            return embeddings.model_name

        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            if isinstance(embeddings, HuggingFaceEmbeddings):
                return "hf:" + embeddings.model_name

            from langchain_community.embeddings import TensorflowHubEmbeddings
            if isinstance(embeddings, TensorflowHubEmbeddings):
                return "tf:" + embeddings.model_url

            else:
                raise Exception("can not find the embeddings type.")


def self_RAG(model_obj: BaseLanguageModel,
             question: str,
             docs: List[Document],
             process_num: int = 10,
             earlyend_num: int = 8,
             max_view_num: int = 100,
             prompt_format_type: str = "auto") -> List[Document]:
    """self RAG model to get the answer

    Args:
        model (BaseLanguageModel): LLM model
        question (str): input prompt
        docs (List[Document]): list of documents
        process_num (int, optional): number of documents to process at one time. Defaults to 10.
        earlyend_num (int, optional): number of irrelevant documents to end the process at each time. Defaults to 8.
        max_view_num (int, optional): number of max documents to view. Defaults to 100.

    Returns:
        List[Document]: relevant documents
    """
    sys_prompt = akasha.prompts.default_doc_grader_prompt()

    results = []

    count = 0
    model_obj, model_name = handle_model_and_name(model_obj)
    while count < len(docs) and count < max_view_num:

        txts = []
        for idx in range(min(process_num, len(docs) - count)):
            prod_prompt = f"Retrieved document: \n\n {docs[count+idx].page_content} \n\n User question: {question}"
            input_text = akasha.prompts.format_sys_prompt(
                sys_prompt, prod_prompt, prompt_format_type, model_name)
            txts.append(input_text)

        irre_count = 0

        response_list = call_batch_model(model_obj, txts)
        for idx, response in enumerate(response_list):
            if 'yes' in response.lower():
                results.append(docs[count + idx])
            else:
                irre_count += 1
        if irre_count >= earlyend_num:
            break

        count += process_num

    return results


def check_relevant_answer(model_obj: BaseLanguageModel,
                          batch_responses: List[str],
                          question: str,
                          prompt_format_type: str = "auto") -> List[str]:
    """ask LLM that each of the retrieved answers list is relevant to the question or not"""
    results = []
    txts = []
    sys_prompt = akasha.prompts.default_answer_grader_prompt()
    model_obj, model_name = handle_model_and_name(model_obj)
    for idx in range(len(batch_responses)):
        prod_prompt = f"Retrieved answer: \n\n {batch_responses[idx]} \n\n User question: {question}"
        text_input = akasha.prompts.format_sys_prompt(sys_prompt, prod_prompt,
                                                      prompt_format_type,
                                                      model_name)
        txts.append(text_input)

    response_list = call_batch_model(model_obj, txts)
    for idx, response in enumerate(response_list):
        if 'yes' in response.lower():
            results.append(batch_responses[idx])

    return results


def merge_history_and_prompt(history_messages: list,
                             system_prompt: str,
                             prompt: str,
                             prompt_format_type: str = "auto",
                             user_tag: str = "user",
                             assistant_tag: str = "assistant",
                             model: str = "remote:xxx") -> Union[str, list]:
    """merge system prompt, history messages, and prompt based on the prompt format type, if history_messages is empty, return the prompt
    if prompt_format_type is start with "chat_", it will become list of dictionary, otherwise it will become string

    Args:
        history_messages (list): _description_
        system_prompt (str): _description_
        prompt (str): _description_
        prompt_format_type (str, optional): _description_. Defaults to "gpt".
        user_tag (str, optional): _description_. Defaults to "user".
        assistant_tag (str, optional): _description_. Defaults to "assistant".
        model (_type_, optional): _description_. Defaults to "remote:xxx".

    Returns:
        Union[str, list]: _description_
    """
    ### decide prompt format type if auto###
    if prompt_format_type == "auto":
        prompt_format_type = akasha.prompts.decide_auto_prompt_format_type(
            model)

    ### if history_messages is empty, return the prompt ###
    if history_messages == [] or history_messages == None or history_messages == "":
        return akasha.prompts.format_sys_prompt(system_prompt, prompt,
                                                prompt_format_type)

    if "chat_" in prompt_format_type and prompt_format_type != "chat_gemma":

        if prompt_format_type == "chat_gemini":
            assistant_tag = "model"

        text_input = akasha.prompts.format_sys_prompt(system_prompt, "",
                                                      prompt_format_type)

        prod_prompt = akasha.prompts.format_sys_prompt("", prompt,
                                                       prompt_format_type)

        prod_history = akasha.prompts.format_history_prompt(
            history_messages, prompt_format_type, user_tag, assistant_tag)

        text_input.extend(prod_history)

        text_input.extend(prod_prompt)

        return text_input

    else:
        history_str = ""

        for i in range(len(history_messages)):

            if i % 2 == 0:
                history_str += user_tag + ": " + history_messages[i] + "\n"

            else:
                history_str += assistant_tag + ": " + history_messages[i] + "\n"

        history_str += "\n\n"
        return akasha.prompts.format_sys_prompt(system_prompt,
                                                history_str + prompt,
                                                prompt_format_type)


class myTokenizer(object):
    """this class is for computing the number of tokens in a given text using different tokenizers.

    Args:
        object (_type_): _description_
    """

    def __init__(self,
                 model_id: str,
                 tokenizer: object,
                 path: str = './tokenizers'):
        """
        Initialize a Tokenizer object.

        Args:
            model_id (str): The name of the model for the tokenizer.
            tokenizer (object): The tokenizer object from HuggingFace.
            path (str, optional): The path to save the tokenizer. Defaults to './tokenizers'.
        """
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.path = path

    def compute_tokens_huggingface(self, text: str) -> int:
        """
        Compute the number of tokens in a given text using huggingface tokenizer.

        Args:
            text (str): The text to be tokenized.

        Returns:
            int: The number of tokens in the text.
        """
        tokens = self.tokenizer(text)
        num_tokens = len(tokens['input_ids'])
        return num_tokens

    @staticmethod
    def compute_tokens_anthropic(text: str, model_name: str) -> int:
        ### take too long time to load the model, skip for now ###
        import anthropic

        token_count = anthropic.Anthropic().beta.messages.count_tokens(
            model=model_name, messages=[{
                'role': 'user',
                'content': text
            }])

        return token_count.input_tokens

    @staticmethod
    def compute_tokens_openai(text: str, model_name: str) -> int:
        """
        Compute the number of tokens in a given text using OpenAI tiktoken.

        Args:
            text (str): The text to be tokenized.
            model_name (str): The name of the OpenAI model. ex. 'openai:gpt-3.5-turbo'
                            Reminder: '/' should not be included in model_name 

        Returns:
            int: The number of tokens in the text.

        Raises:
            ValueError: If the model_name is not a valid OpenAI model.
        """
        if '/' in model_name:
            raise ValueError('Non-OpenAI models are not supported')
        if model_name.lower().startswith('openai:'):
            model_name = model_name.lower().lstrip('openai:')
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(text)
        num_tokens = len(tokens)
        return num_tokens

    @classmethod
    def compute_tokens(cls,
                       text: str,
                       model_id: str,
                       model_path: str = './tokenizers',
                       save_tokenizer: bool = True) -> int:
        """
        Compute the number of tokens in a given text using either huggingface or OpenAI tiktoken.

        If the model_id is an OpenAI model, the tiktoken library is used to tokenize the text.
        If the model_id is a Non-OpenAI model, the huggingface tokenizer is used to tokenize the text.

        Args:
            text (str): The text to be tokenized.
            model_id (str): The name of the model. ex. 'gpt-2', 'openai:gpt-3.5-turbo'
            model_path (str, optional): The path to the tokenizer. Defaults to './tokenizers'.
            save_tokenizer (bool, optional): Whether to save the tokenizer locally. Defaults to True.

        Returns:
            int: The number of tokens in the text.

        """
        model_type, model_name = _separate_name(model_id)
        if model_type in ["openai", "gpt-3.5", "gpt"]:
            return cls.compute_tokens_openai(text, model_name)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(text))
            return num_tokens


def get_mac_address() -> str:
    # Get the MAC address
    mac = uuid.getnode()
    # Convert the MAC address to a readable format without colons
    mac_address = ''.join([
        '{:02x}'.format((mac >> elements) & 0xff)
        for elements in range(0, 2 * 6, 2)
    ][::-1])
    return get_text_md5(mac_address)


def generate_keyword(
        text: str,
        keyword_num: int = 5,
        keyword_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """use keybert to extract keywords from texts

    Args:
        texts (str): _description_
        keyword_model (str, optional): _description_. Defaults to "paraphrase-multilingual-MiniLM-L12-v2".
    """

    from keybert import KeyBERT

    kw_model = KeyBERT(keyword_model)

    kw_list = kw_model.extract_keywords(text, top_n=keyword_num)

    keyword_list = [kwww[0] for kwww in kw_list]

    return keyword_list


def detect_encoding(file_path: Union[str, Path]) -> str:
    with open(file_path, 'rb') as file:
        raw_data = file.read(1000)  # Read a portion of the file
        result = detect(raw_data)
        return result['encoding']
