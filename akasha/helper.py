import numpy as np
import jieba
import json, re, time
from pathlib import Path
import opencc
from typing import Callable, Union, Tuple, List, Generator
from langchain.schema import Document
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
    TensorflowHubEmbeddings,
)
from akasha.models.hf import chatGLM, hf_model, custom_model, custom_embed, remote_model, gptq
from akasha.models.llama2 import peft_Llama2, get_llama_cpp_model, TaiwanLLaMaGPTQ
import os, traceback, logging
import shutil
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
import akasha.format as afr
import akasha.prompts

jieba.setLogLevel(
    jieba.logging.INFO)  ## ignore logging jieba model information


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


def _handle_azure_env() -> Tuple[str, str, str]:
    """from environment variable get the api_base, api_key, api_version

    Returns:
        (str, str, str): api_base, api_key, api_version
    """
    check_env, ret, count = ["BASE", "KEY", "VERSION"], ["", "", ""], 0
    try:
        for check in check_env:
            if f"AZURE_API_{check}" in os.environ:
                ret[count] = os.environ[f"AZURE_API_{check}"]
                if "OPENAI_API_BASE" in os.environ:
                    os.environ.pop("OPENAI_API_BASE", None)
            elif f"OPENAI_API_{check}" in os.environ:
                ret[count] = os.environ[f"OPENAI_API_{check}"]
                if check == "BASE":
                    os.environ["AZURE_API_BASE"] = os.environ[
                        "OPENAI_API_BASE"]
                    os.environ.pop("OPENAI_API_BASE", None)
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
                      verbose: bool = False) -> vars:
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
        embeddings = custom_embed(func=embedding_name)
        if verbose:
            print("selected custom embedding.")
        return embeddings

    embedding_type, embedding_name = _separate_name(embedding_name)

    if embedding_type in [
            "text-embedding-ada-002", "openai", "openaiembeddings"
    ]:
        import openai

        if ("AZURE_API_TYPE" in os.environ and os.environ["AZURE_API_TYPE"]
                == "azure") or ("OPENAI_API_TYPE" in os.environ
                                and os.environ["OPENAI_API_TYPE"] == "azure"):
            embedding_name = embedding_name.replace(".", "")
            api_base, api_key, api_version = _handle_azure_env()
            embeddings = AzureOpenAIEmbeddings(azure_deployment=embedding_name,
                                               azure_endpoint=api_base,
                                               api_key=api_key,
                                               api_version=api_version,
                                               validate_base_url=False)

        else:
            openai.api_type = "open_ai"

            embeddings = OpenAIEmbeddings(
                model=embedding_name,
                openai_api_base="https://api.openai.com/v1",
                api_key=os.environ["OPENAI_API_KEY"],
                openai_api_type="open_ai",
            )
        info = "selected openai embeddings.\n"

    elif embedding_type in ["rerank", "re"]:
        if embedding_name == "":
            embedding_name = "BAAI/bge-reranker-base"

        embeddings = "rerank:" + embedding_name
        info = "selected rerank embeddings.\n"
    elif embedding_type in [
            "huggingface",
            "huggingfaceembeddings",
            "transformers",
            "transformer",
            "hf",
    ]:

        embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
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

    else:
        embeddings = OpenAIEmbeddings()
        info = "can not find the embeddings, use openai as default.\n"

    if verbose:
        print(info)
    return embeddings


def handle_model(model_name: Union[str, Callable] = "openai:gpt-3.5-turbo",
                 verbose: bool = False,
                 temperature: float = 0.0,
                 max_output_tokens: int = 1024) -> BaseLanguageModel:
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
        model = custom_model(func=model_name, temperature=temperature)
        if verbose:
            print("selected custom model.")
        return model

    model_type, model_name = _separate_name(model_name)

    if model_type in ["remote", "server", "tgi", "text-generation-inference"]:

        base_url = model_name
        info = f"selected remote model. \n"
        model = remote_model(base_url,
                             temperature,
                             max_output_tokens=max_output_tokens)

    elif (model_type
          in ["llama-cpu", "llama-gpu", "llama", "llama2", "llama-cpp"]
          and model_name != ""):
        model = get_llama_cpp_model(model_type, model_name, temperature)
        info = "selected llama-cpp model\n"
    elif model_type in [
            "huggingface",
            "huggingfacehub",
            "transformers",
            "transformer",
            "huggingface-hub",
            "hf",
    ]:
        model = hf_model(model_name=model_name,
                         temperature=temperature,
                         max_output_tokens=max_output_tokens)
        info = f"selected huggingface model {model_name}.\n"

    elif model_type in ["chatglm", "chatglm2", "glm"]:
        model = chatGLM(model_name=model_name,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens)
        info = f"selected chatglm model {model_name}.\n"

    elif model_type in ["lora", "peft"]:
        model = peft_Llama2(model_name_or_path=model_name,
                            temperature=temperature)
        info = f"selected peft model {model_name}.\n"

    elif model_type in ["gptq"]:
        if model_name.lower().find("taiwan-llama") != -1:
            model = TaiwanLLaMaGPTQ(model_name_or_path=model_name,
                                    temperature=temperature)

        else:
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
        if ("AZURE_API_TYPE" in os.environ and os.environ["AZURE_API_TYPE"]
                == "azure") or ("OPENAI_API_TYPE" in os.environ
                                and os.environ["OPENAI_API_TYPE"] == "azure"):
            model_name = model_name.replace(".", "")
            api_base, api_key, api_version = _handle_azure_env()
            model = AzureChatOpenAI(
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
            openai.api_type = "open_ai"
            model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.environ["OPENAI_API_KEY"],
                streaming=True,
                callbacks=call_back,
            )
        info = f"selected openai model {model_name}.\n"
    if verbose:
        print(info)

    return model


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
    cc = opencc.OpenCC("s2t.json")
    return cc.convert(text)


def _get_text(texts: list,
              previous_summary: str,
              i: int,
              max_doc_len: int,
              language: str = "ch") -> Tuple[int, str, int]:
    """used in summary, combine chunks of texts into one chunk that can fit into llm model

    Args:
        texts (list): chunks of texts
        previous_summary (str): _description_
        i (int): start from i-th chunk
        max_doc_len (int): the max doc length we want to fit into llm model at one time
        language (str): 'ch' for chinese and 'en' for others, default 'ch'\n

    Returns:
        (int, str, int): return the total tokens of combined chunks, combined chunks of texts, and the index of next chunk
    """
    cur_count = get_doc_length(language, previous_summary)
    words_len = get_doc_length(language, texts[i])
    cur_text = ""
    while cur_count + words_len < max_doc_len and i < len(texts):
        cur_count += words_len
        cur_text += texts[i] + "\n"
        i += 1
        if i < len(texts):
            words_len = get_doc_length(language, texts[i])

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
    try:
        try:
            model_type = model._llm_type
        except:
            print_flag = False
            try:
                response = model._call(input_text)
            except:
                response = model._generate(input_text)

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

        if "huggingface" in model_type:
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
    try:
        try:
            model_type = model._llm_type
        except:

            try:
                response = model.stream(input_text)
            except:
                response = model._call(input_text)

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
                    prompt_format_type: str = "gpt",
                    language: str = "zh") -> str:
    """translate texts to target language

    Args:
        model_obj (BaseLanguageModel): LLM that used to translate
        texts (str): texts that need to be translated
        prompt_format_type (str, optional): system prompt format. Defaults to "gpt".
        language (str, optional): target language. Defaults to "zh".

    Returns:
        str: translated texts
    """
    sys_prompt = akasha.prompts.default_translate_prompt(language)
    prod_prompt = akasha.prompts.format_sys_prompt(sys_prompt, texts,
                                                   prompt_format_type)

    response = call_model(model_obj, prod_prompt)

    return response


def call_JSON_formatter(
    model_obj: BaseLanguageModel,
    texts: str,
    keys: Union[str, list] = "",
    prompt_format_type: str = "gpt",
) -> Union[dict, None]:
    """use LLM to transfer texts into JSON format

    Args:
        model_obj (BaseLanguageModel): LLM that used to transfer
        texts (str): texts that need to be transferred
        keys (Union[str, list], optional): keys name of output dictionary. Defaults to "".
        prompt_format_type (str, optional): system prompt format. Defaults to "gpt". Defaults to "gpt".

    Returns:
        Union[dict, None]: return the JSON part of the string, if not found return None
    """

    if keys == "":
        sys_prompt = "Format the following TEXTS into a single JSON instance that conforms to the JSON schema."
    elif isinstance(keys, str):
        keys = [keys]

    if keys != "":
        sys_prompt = f"Format the following TEXTS into a single JSON instance that conforms to the JSON schema which includes: {', '.join(keys)}\n\n"

    prod_prompt = akasha.prompts.format_sys_prompt(sys_prompt,
                                                   "TEXTS: " + texts,
                                                   prompt_format_type)

    response = call_model(model_obj, prod_prompt)
    return extract_json(response)


def retri_history_messages(messages: list,
                           pairs: int = 10,
                           max_doc_len: int = 750,
                           role1: str = "User",
                           role2: str = "Assistant",
                           language: str = "ch") -> Tuple[str, int]:
    """from messages dict list, get pairs of user question and assistant response from most recent and not exceed max_doc_len and pairs, and return the text with total length

    Args:
        messages (list): history messages list, each index is a dict with keys: role("user", "assistant"), content(content of message)
        pairs (int, optional): the maximum number of messages. Defaults to 10.
        max_doc_len (int, optional): the maximum number of messages length. Defaults to 750.
        language (str, optional): message language. Defaults to "ch".

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
        len_texta = akasha.helper.get_doc_length(language, texta)  #)
        len_textq = akasha.helper.get_doc_length(language, textq)

        if cur_len + len_texta > max_doc_len:
            break
        cur_len += len_texta
        ret.append(texta)

        if cur_len + len_textq > max_doc_len:
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

    if isinstance(embeddings, custom_embed):
        return embeddings.model_name

    elif isinstance(embeddings, OpenAIEmbeddings) or isinstance(
            embeddings, AzureOpenAIEmbeddings):
        return "openai:" + embeddings.model

    elif isinstance(embeddings, HuggingFaceEmbeddings):
        return "hf:" + embeddings.model_name

    elif isinstance(embeddings, TensorflowHubEmbeddings):
        return "tf:" + embeddings.model_url

    else:
        raise Exception("can not find the embeddings type.")


def self_RAG(model_obj: BaseLanguageModel,
             question: str,
             docs: List[Document],
             process_num: int = 10,
             earlyend_num: int = 8,
             max_view_num: int = 100,
             prompt_format_type: str = "gpt") -> List[Document]:
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
    while count < len(docs) and count < max_view_num:

        txts = []
        for idx in range(min(process_num, len(docs) - count)):
            prod_prompt = f"Retrieved document: \n\n {docs[count+idx].page_content} \n\n User question: {question}"
            input_text = akasha.prompts.format_sys_prompt(
                sys_prompt, prod_prompt, prompt_format_type)
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
                          batch_responses: List[str], question: str,
                          prompt_format_type: str) -> List[str]:
    """ask LLM that each of the retrieved answers list is relevant to the question or not"""
    results = []
    txts = []
    sys_prompt = akasha.prompts.default_answer_grader_prompt()
    for idx in range(len(batch_responses)):
        prod_prompt = f"Retrieved answer: \n\n {batch_responses[idx]} \n\n User question: {question}"
        text_input = akasha.prompts.format_sys_prompt(sys_prompt, prod_prompt,
                                                      prompt_format_type)
        txts.append(text_input)

    response_list = call_batch_model(model_obj, txts)
    for idx, response in enumerate(response_list):
        if 'yes' in response.lower():
            results.append(batch_responses[idx])

    return results


def merge_history_and_prompt(
        history_messages: list,
        system_prompt: str,
        prompt: str,
        prompt_format_type: str = "gpt",
        user_tag: str = "user",
        assistant_tag: str = "assistant") -> Union[str, list]:

    if history_messages == [] or history_messages == None or history_messages == "":
        return akasha.prompts.format_sys_prompt(system_prompt, prompt,
                                                prompt_format_type)

    if prompt_format_type == "chat_gpt":
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


def retri_max_texts(texts_list: list,
                    left_doc_len: int,
                    language: str = "ch") -> Tuple[list, int]:
    """return list of texts that do not exceed the left_doc_len

    Args:
        texts_list (list): _description_
        left_doc_len (int): _description_

    Returns:
        Tuple[list, int]: _description_
    """
    ret = []
    cur_len = 0
    for text in texts_list:
        txt_len = get_doc_length(language, text)
        if cur_len + txt_len > left_doc_len:
            break
        cur_len += txt_len
        ret.append(text)
    return ret, cur_len
