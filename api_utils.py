from pathlib import Path
import requests
import os
import shutil
import time
import json
import hashlib
import akasha, akasha.db
from typing import Generator, Union, List, Callable, Optional, Any, Tuple
import warnings

warnings.filterwarnings("ignore")

from transformers import pipeline, AutoTokenizer, TextStreamer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from langchain.llms.base import LLM
from threading import Thread
import openai
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from akasha.models.hf import remote_model

HOST = os.getenv("API_HOST", "127.0.0.1")
PORT = os.getenv("API_PORT", "8000")
api_url = {
    "delete_expert": f"{HOST}:{PORT}/expert/delete",
    "test_openai": f"{HOST}:{PORT}/openai/test_openai",
    "test_azure": f"{HOST}:{PORT}/openai/test_azure",
}
ACCOUNTS_PATH = "./config/accounts.yaml"
DOCS_PATH = "./docs"
CONFIG_PATH = "./config"
EXPERT_CONFIG_PATH = "./config/experts"
DATASET_CONFIG_PATH = "./config/datasets/"
DB_PATH = "./chromadb/"
MODEL_NAME_PATH = "./config/default_model_name.txt"
DEFAULT_CONFIG = {
    "system_prompt": "",
    "language_model": "openai:gpt-3.5-turbo",
    "search_type": "svm",
    "top_k": 5,
    "threshold": 0.1,
    "max_doc_len": 1500,
    "temperature": 0.0,
    "use_compression": 0,  # 0 for False, 1 for True
    "compression_language_model": "",
    "prompt_format_type": "gpt",
}


def get_docs_path():
    return DOCS_PATH


def get_db_path():
    return DB_PATH


def get_default_config():
    return DEFAULT_CONFIG


def get_config_path():
    return CONFIG_PATH


def get_expert_config_path():
    return EXPERT_CONFIG_PATH


def get_dataset_config_path():
    return DATASET_CONFIG_PATH


def get_model_path():
    return "./model"


def get_accounts_path():
    return ACCOUNTS_PATH


def get_model_name_path():
    return MODEL_NAME_PATH


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


def generate_hash(owner: str, dataset_name: str) -> str:
    """use hashlib sha256 to generate hash value of owner and dataset_name/expert_name {owner}-{dataset_name/expert_name}

    Args:
        owner (str): owner name
        dataset_name (str): dataset name or expert name

    Returns:
        str: hash value
    """
    combined_string = f"{owner}-{dataset_name}"
    sha256 = hashlib.sha256()
    sha256.update(combined_string.encode("utf-8"))
    hex_digest = sha256.hexdigest()
    return hex_digest


def check_dir(path: str) -> bool:
    """check if directory exist, if not, create it."""
    try:
        p = Path(path)
        if not p.exists():
            p.mkdir()
    except:
        return False
    return True


def check_config(path: str) -> bool:
    """check if config directory exist, if not, create it.
    also check sub directory exist(path), if not, create it.

    Args:
        path (str): sub directory path

    Returns:
        bool: check and create success or not
    """
    try:
        config_p = Path(CONFIG_PATH)

        if not config_p.exists():
            config_p.mkdir()

        if path == "":
            return True
        if not Path(path).exists():
            Path(path).mkdir()
    except:
        return False
    return True


def get_file_list_of_dataset(docs_path: str) -> list:
    """get all file names in a dataset

    Args:
        docs_path (str): _description_

    Returns:
        list[str]: list of file names
    """
    if not Path(docs_path).exists():
        return []

    files = os.listdir(docs_path)
    return files


def get_lastupdate_of_file_in_dataset(docs_path: str, file_name: str) -> float:
    """get the last update time of a file in dataset

    Args:
        docs_path (str): docs directory path
        file_name (str): file path

    Returns:
        (float): float number of last update time
    """
    file_path = os.path.join(docs_path, file_name)
    last_update = os.path.getmtime(file_path)
    return last_update


def get_lastupdate_of_dataset(dataset_name: str, owner: str) -> str:
    """get the last update time of each file in dataset, and find out the lastest one

    Args:
        dataset_name (str): dataset name
        owner (str): owner name

    Raises:
        Exception: cano find DOCS_PATH directory and can not create it.
        Exception: can not create owner directory in DOCS_PATH

    Returns:
        (str): return "" if no file in dataset, else return the last update time of the latest file in str format
    """
    last_updates = []
    try:
        if not check_dir(DOCS_PATH):
            raise Exception(f"can not create {DOCS_PATH} directory")

        owner_path = Path(DOCS_PATH) / owner

        if not check_dir(owner_path):
            raise Exception(f"can not create {owner} directory in {DOCS_PATH}")

        docs_path = os.path.join(owner_path.__str__(), dataset_name)
        dataset_files = get_file_list_of_dataset(docs_path)
        for file in dataset_files:
            last_updates.append(
                get_lastupdate_of_file_in_dataset(docs_path, file))

        if len(last_updates) == 0:
            raise Exception(f"Dataset={dataset_name} has not file.")

    except Exception as e:
        return ""

    return time.ctime(max(last_updates))


def update_dataset_name_from_chromadb(old_dataset_name: str, dataset_name: str,
                                      md5_list: list):
    """change dataset name in chromadb file, in chromadn directory, change directory name of {old_dataset_name}_md5 to {dataset_name}_md5

    Args:
        old_dataset_name (str): _description_
        dataset_name (str): _description_
        md5_list (list):
    """

    for md5 in md5_list:
        p = Path(DB_PATH)
        tag = old_dataset_name + "_" + md5
        for file in p.glob("*"):
            if tag in file.name:
                new_name = (dataset_name + "_" + md5 + "_" +
                            "_".join(file.name.split("_")[2:]))
                file.rename(Path(DB_PATH) / new_name)
    return


def check_and_delete_files_from_expert(dataset_name: str, owner: str,
                                       delete_files: list,
                                       old_dataset_name: str) -> bool:
    """check every expert if they use this dataset, if yes, delete files that are in delete_files from expert's dataset list.
    change dataset name to new dataset name if dataset_name != old_dataset_name.
        if the expert has no dataset after deletion, delete the expert file.

    Args:
        dataset_name (str): dataset name
        owner (str): owner name
        delete_files (list): list of file names that delete from dataset

    Returns:
        bool: delete any file from expert or not
    """

    p = Path(EXPERT_CONFIG_PATH)
    if not p.exists():
        return False
    flag = False
    delete_set = set(delete_files)
    for file in p.glob("*"):
        with open(file, "r", encoding="utf-8") as ff:
            expert = json.load(ff)
        for dataset in expert["datasets"]:
            if dataset["name"] == old_dataset_name and dataset[
                    "owner"] == owner:
                dataset["name"] = dataset_name
                for file in dataset["files"]:
                    if file in delete_set:
                        dataset["files"].remove(file)
                        flag = True
                if len(dataset["files"]) == 0:
                    ## delete dataset if no file in dataset
                    expert["datasets"].remove(dataset)
                break
        if len(expert["datasets"]) == 0:
            ## delete file if no dataset in expert
            res = requests.post(
                api_url["delete_expert"],
                json={
                    "owner": expert["owner"],
                    "expert_name": expert["name"]
                },
            ).json()
        else:
            with open(file, "w", encoding="utf-8") as f:
                json.dump(expert, f, ensure_ascii=False, indent=4)

    if flag:
        return True

    return False


def check_and_delete_dataset(dataset_name: str, owner: str) -> bool:
    """check every expert if they use this dataset, if yes, delete this dataset from expert's dataset list.
        if the expert has no dataset after deletion, delete the expert file.

    Args:
        dataset_name (str): dataset name
        owner (str): owner of dataset
    """

    p = Path(EXPERT_CONFIG_PATH)
    if not p.exists():
        return False
    flag = False
    for file in p.glob("*"):
        with open(file, "r", encoding="utf-8") as ff:
            expert = json.load(ff)
        for dataset in expert["datasets"]:
            if dataset["name"] == dataset_name and dataset["owner"] == owner:
                expert["datasets"].remove(dataset)
                flag = True
                break
        if len(expert["datasets"]) == 0:
            ## delete file if no dataset in expert
            res = requests.post(
                api_url["delete_expert"],
                json={
                    "owner": expert["owner"],
                    "expert_name": expert["name"]
                },
            ).json()
        else:
            with open(file, "w", encoding="utf-8") as f:
                json.dump(expert, f, ensure_ascii=False, indent=4)

    if flag:
        return True

    return False


def check_and_delete_chromadb(chunk_size: int, embedding_model: str,
                              filename: str, dataset_name: str, owner: str,
                              id: str, data_path: Path) -> str:
    """check if any of other expert use the same file with same chunk size and embedding model, if not, delete the chromadb file

    Args:
        chunk_size (int): _description_
        embedding_model (str): _description_
        filename (str): _description_
        dataset_name (str): _description_
        owner (str): _description_
        id (str): _description_

    Returns:
        _type_: _description_
    """

    embed_type, embed_name = _separate_name(embedding_model)

    p = Path(EXPERT_CONFIG_PATH)
    if not p.exists():
        return ""

    for file in p.glob("*"):
        if file == data_path:
            continue
        with open(file, "r", encoding="utf-8") as ff:
            expert = json.load(ff)

        if (expert["chunk_size"] == chunk_size
                and expert["embedding_model"] == embedding_model):
            for dataset in expert["datasets"]:
                if dataset["name"] == dataset_name and dataset[
                        "owner"] == owner:
                    if filename in dataset["files"]:
                        return ""

    ## not find, delete

    # get MD5 of file
    md5 = id

    target_path = Path(DATASET_CONFIG_PATH) / (id + ".json")
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for file in data["files"]:
            if file["filename"] == filename:
                md5 = file["MD5"]
                break

    db_storage_path = (DB_PATH + dataset_name + "_" +
                       filename.split(".")[0].replace(" ", "").replace(
                           "_", "") + "_" + md5 + "_" + embed_type + "_" +
                       embed_name.replace("/", "-") + "_" + str(chunk_size))

    if Path(db_storage_path).exists():
        return db_storage_path

    return ""


def delete_datasets_from_expert(ori_datasets: list, delete_datasets: list):
    """from ori_datasets remove files in delete_datasets

    Args:
        ori_datasets (list): original datasets in expert
        delete_datasets (list): datasets(include name, owner, file) that wants to delete from expert

    Returns:
        list: list of datasets after deletion
    """
    delete_hash = {
        (dataset["owner"], dataset["name"]): dataset["files"]
        for dataset in delete_datasets
    }

    for dataset in ori_datasets:
        if (dataset["owner"], dataset["name"]) in delete_hash:
            for file in delete_hash[(dataset["owner"], dataset["name"])]:
                if file in dataset["files"]:
                    dataset["files"].remove(file)
            ## if no file in dataset, delete dataset
            if len(dataset["files"]) == 0:
                ori_datasets.remove(dataset)

    return ori_datasets


def add_datasets_to_expert(ori_datasets: list, add_datasets: list) -> list:
    """merge ori_datasets and add_datasets,  if dataset already exist, append files to it, else create new dataset in ori_datasets

    Args:
        ori_datasets (list): original datasets in expert
        add_datasets (list): datasets(include name, owner, files) that wants to add to expert

    Returns:
        list[dict]: merged datasets
    """
    append_hash = {
        (dataset["owner"], dataset["name"]): dataset["files"]
        for dataset in ori_datasets
    }

    for dataset in add_datasets:
        if (dataset["owner"], dataset["name"]) in append_hash:
            for file in dataset["files"]:
                if file not in append_hash[(dataset["owner"],
                                            dataset["name"])]:
                    append_hash[(dataset["owner"],
                                 dataset["name"])].append(file)
        else:
            append_hash[(dataset["owner"], dataset["name"])] = dataset["files"]

    ori_datasets = []

    ## rebuild ori_datasets
    for key in append_hash:
        ori_datasets.append({
            "owner": key[0],
            "name": key[1],
            "files": append_hash[key]
        })

    return ori_datasets


def choose_openai_key(
    config_file_path: str,
    openai_key: str = "",
    azure_key: str = "",
    azure_base: str = "",
) -> dict:
    """test the openai key, azure openai key, or keys in openai.json file and choose it if valid

    Args:
        openai_key (str, optional): openai key. Defaults to "".
        azure_key (str, optional): azure openai key. Defaults to "".
        azure_base (str, optional): azure base url. Defaults to "".
    """
    openai_key = openai_key.replace(" ", "")
    azure_key = azure_key.replace(" ", "")
    azure_base = azure_base.replace(" ", "")

    if azure_key != "" and azure_base != "":
        # res = requests.get(api_url['test_azure'], json = {'azure_key':azure_key, 'azure_base':azure_base}).json()
        # if res['status'] == 'success':
        return {"azure_key": azure_key, "azure_base": azure_base}

    if openai_key != "":
        # res = requests.get(api_url['test_openai'], json = {'openai_key':openai_key}).json()
        # if res['status'] == 'success':
        return {"openai_key": openai_key}

    if Path(CONFIG_PATH).exists():
        file_path = Path(config_file_path)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "azure_key" in data and "azure_base" in data:
                # res = requests.get(api_url['test_azure'], json = {'azure_key':data['azure_key'], 'azure_base':data['azure_base']}).json()
                # if res['status'] == 'success':
                return {
                    "azure_key": data["azure_key"],
                    "azure_base": data["azure_base"],
                }

            if "openai_key" in data:
                # res = requests.get(api_url['test_openai'], json = {'openai_key':data['openai_key']}).json()
                # if res['status'] == 'success':
                return {"openai_key": data["openai_key"]}

    return {}


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


def retri_history_messages(messages: list,
                           pairs: int = 10,
                           max_doc_len: int = 750,
                           language: str = "ch") -> Tuple[list, int]:
    """from messages dict list, get pairs of user question and assistant response from most recent and not exceed max_doc_len and pairs, and return the text with total length

    Args:
        messages (list): history messages list, each index is a dict with keys: role("user", "assistant"), content(content of message)
        pairs (int, optional): the maximum number of messages. Defaults to 10.
        max_doc_len (int, optional): the maximum number of messages length. Defaults to 750.
        language (str, optional): message language. Defaults to "ch".

    Returns:
        Tuple[list, int]: return the text with total length.
    """
    cur_len = 0
    count = 0
    ret = []
    splitter = '\n----------------\n'

    for i in range(len(messages) - 1, -1, -2):
        if count >= pairs:
            break
        if (messages[i]["role"] != "assistant") or (messages[i - 1]["role"]
                                                    != "user"):
            i += 1
            continue
        texta = messages[i]["content"] + "\n"
        textq = messages[i - 1]["content"] + "\n"
        temp = akasha.helper.get_doc_length(language, textq + texta)
        if cur_len + temp > max_doc_len:
            break
        cur_len += temp
        ret.append(texta)
        ret.append(textq)
        count += 1

    if count == 0:
        return "", 0

    ret.reverse()
    #ret_str = splitter + "chat history: \n\n" + ''.join(ret) + splitter

    return ret, cur_len


class hf_model(LLM):

    max_token: int = 4096
    tokenizer: Any
    model: Any
    streamer: Any
    device: Any

    def __init__(self, model_name: str, temperature: float, **kwargs):
        """define custom model, input func and temperature

        Args:
            **func (Callable)**: the function return response from llm\n
        """
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if temperature == 0.0:
            temperature = 0.01
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            temperature=temperature,
            repetition_penalty=1.2,
            top_p=0.95,
            torch_dtype=torch.float16,
            device_map="auto").to(self.device)

    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return "huggingface text generation model"

    def stream(self,
               prompt: str,
               stop: Optional[List[str]] = None) -> Generator[str, None, None]:

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        gerneration_kwargs = dict(inputs,
                                  streamer=self.streamer,
                                  max_new_tokens=1024,
                                  do_sample=True,
                                  min_new_tokens=10)
        #self.model.generate(**inputs, streamer= self.streamer, max_new_tokens=1024, do_sample=True)

        thread = Thread(target=self.model.generate, kwargs=gerneration_kwargs)
        thread.start()
        # for text in self.streamer:
        #     yield text
        yield from self.streamer

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        gerneration_kwargs = dict(
            inputs,
            streamer=self.streamer,
            max_new_tokens=1024,
            do_sample=True,
        )
        thread = Thread(target=self.model.generate, kwargs=gerneration_kwargs)
        thread.start()
        generated_text = ""
        for new_text in self.streamer:
            generated_text += new_text
        return generated_text

    def _generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        return self._call(prompt, stop)


def _handle_stream_model(model_name: str, verbose: bool,
                         temperature: float) -> LLM:
    """handle each model type, including openai, remote, llama, chatglm, lora, gptq, huggingface, and return the model object
    for remote, gpt and huggingface models, we are using streaming mode; for others, we are using non-streaming mode.

    Args:
        model_name (str): {model_type}:{model_name)
        verbose (bool): print the processing text or not
        temperature (float): temperature for language model

    Returns:
        LLM: model object
    """
    model_type, model_name = _separate_name(model_name)

    if model_type in ["openai", "gpt-3.5", "gpt"]:

        if ("AZURE_API_TYPE" in os.environ and os.environ["AZURE_API_TYPE"]
                == "azure") or ("OPENAI_API_TYPE" in os.environ
                                and os.environ["OPENAI_API_TYPE"] == "azure"):
            model_name = model_name.replace(".", "")
            api_base, api_key, api_version = akasha.helper._handle_azure_env()
            model = AzureChatOpenAI(
                deployment_name=model_name,
                temperature=temperature,
                azure_endpoint=api_base,
                api_key=api_key,
                api_version=api_version,
                validate_base_url=False,
                streaming=True,
            )
        else:
            openai.api_type = "open_ai"
            model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.environ["OPENAI_API_KEY"],
                streaming=True,
            )
        info = f"selected openai model {model_name}.\n"

    elif model_type in [
            "remote", "server", "tgi", "text-generation-inference"
    ]:
        model = remote_model(model_name, temperature=temperature)
        info = f"selected remote model {model_name}.\n"

    elif (model_type
          in ["llama-cpu", "llama-gpu", "llama", "llama2", "llama-cpp"]
          and model_name != ""):
        model = akasha.helper.get_llama_cpp_model(model_type, model_name,
                                                  temperature)
        info = "selected llama-cpp model\n"

    elif model_type in ["chatglm", "chatglm2", "glm"]:
        model = akasha.helper.chatGLM(model_name=model_name,
                                      temperature=temperature)
        info = f"selected chatglm model {model_name}.\n"

    elif model_type in ["lora", "peft"]:
        model = akasha.helper.peft_Llama2(model_name_or_path=model_name,
                                          temperature=temperature)
        info = f"selected peft model {model_name}.\n"

    elif model_type in ["gptq"]:
        if model_name.lower().find("taiwan-llama") != -1:
            model = akasha.helper.TaiwanLLaMaGPTQ(
                model_name_or_path=model_name, temperature=temperature)

        else:
            model = akasha.helper.gptq(
                model_name_or_path=model_name,
                temperature=temperature,
                bit4=True,
                max_token=4096,
            )
        info = f"selected gptq model {model_name}.\n"

    else:
        model = hf_model(model_name=model_name, temperature=temperature)
        info = f"selected huggingface model {model_name}.\n"

    if verbose:
        print(info)

    return model


class Doc_QA_stream(akasha.atman):
    """class for implement search db based on user prompt and generate response from llm model, include get_response and chain_of_thoughts."""

    def __init__(
        self,
        embeddings: str = "openai:text-embedding-ada-002",
        chunk_size: int = 1000,
        model: str = "openai:gpt-3.5-turbo",
        verbose: bool = False,
        topK: int = -1,
        threshold: float = 0.2,
        language: str = "ch",
        search_type: Union[str, Callable] = "svm",
        system_prompt: str = "",
        prompt_format_type: str = "gpt",
        max_doc_len: int = 1500,
        temperature: float = 0.0,
        use_chroma: bool = False,
        use_rerank: bool = False,
        ignore_check: bool = False,
    ):
        """initials of Doc_QA_stream class

        Args:
            embeddings (_type_, optional): embedding model, including two types(openai and huggingface). Defaults to "openai:text-embedding-ada-002".
            chunk_size (int, optional): the max length of each text segments. Defaults to 1000.
            model (_type_, optional): language model. Defaults to "openai:gpt-3.5-turbo".
            verbose (bool, optional): print the processing text or not. Defaults to False.
            topK (int, optional): the number of documents to be selected. Defaults to 2.
            threshold (float, optional): threshold of similarity for searching relavant documents. Defaults to 0.2.
            language (str, optional): "ch" chinese or "en" english. Defaults to "ch".
            search_type (Union[str, Callable], optional): _description_. Defaults to "svm".
            system_prompt (str, optional): the prompt you want llm to output in certain format. Defaults to "".
            prompt_format_type (str, optional): the prompt and system prompt format for the language model, including two types(gpt and llama). Defaults to "gpt".
            max_doc_len (int, optional): max total length of selected documents. Defaults to 1500.
            temperature (float, optional): temperature for language model. Defaults to 0.0.
            compression (bool, optional): compress the selected documents or not. Defaults to False.
            use_chroma (bool, optional): use chroma db name instead of documents path to load data or not. Defaults to False.
            use_rerank (bool, optional): use rerank model to re-rank the selected documents or not. Defaults to False.
            ignore_check (bool, optional): speed up loading data if the chroma db is already existed. Defaults to False.
        """

        super().__init__(chunk_size, model, verbose, topK, threshold, language,
                         search_type, "", system_prompt, max_doc_len,
                         temperature)

        ### set argruments ###
        self.doc_path = ""
        self.compression = False
        self.use_chroma = use_chroma
        self.ignore_check = ignore_check
        self.use_rerank = use_rerank
        self.prompt_format_type = prompt_format_type
        ### set variables ###
        self.logs = {}
        self.model_obj = _handle_stream_model(model, self.verbose,
                                              self.temperature)
        self.embeddings_obj = akasha.helper.handle_embeddings(
            embeddings, self.verbose)
        self.embeddings = akasha.helper.handle_search_type(embeddings)
        self.model = akasha.helper.handle_search_type(model)
        self.search_type = search_type
        self.db = None
        self.docs = []
        self.doc_tokens = 0
        self.doc_length = 0
        self.response = ""
        self.prompt = ""
        self.ignored_files = []

    def _set_model(self, **kwargs):
        """change model, embeddings, search_type, temperature if user use **kwargs to change them."""
        ## check if we need to change db, model_obj or embeddings_obj ##
        if "search_type" in kwargs:
            self.search_type_str = akasha.helper.handle_search_type(
                kwargs["search_type"], self.verbose)

        if "embeddings" in kwargs:
            self.embeddings_obj = akasha.helper.handle_embeddings(
                kwargs["embeddings"], self.verbose)

        if "model" in kwargs or "temperature" in kwargs:
            new_temp = self.temperature
            new_model = self.model
            if "temperature" in kwargs:
                new_temp = kwargs["temperature"]
            if "model" in kwargs:
                new_model = kwargs["model"]
            if new_model != self.model or new_temp != self.temperature:
                self.model_obj = _handle_stream_model(new_model, self.verbose,
                                                      new_temp)

    def search_docs(self, doc_path: Union[List[str], str], prompt: str,
                    **kwargs) -> Tuple[str, str]:

        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.doc_path = doc_path
        self.prompt = prompt
        search_dict = {}
        if self.use_chroma:
            self.db, self.ignored_files = akasha.db.get_db_from_chromadb(
                self.doc_path, self.embeddings)
        else:
            self.db, self.ignored_files = akasha.db.processMultiDB(
                self.doc_path, self.verbose, self.embeddings_obj,
                self.embeddings, self.chunk_size, self.ignore_check)

        ### start to get response ###
        retrivers_list = akasha.search.get_retrivers(
            self.db, self.embeddings_obj, self.use_rerank, self.threshold,
            self.search_type, search_dict)
        self.docs, self.doc_length, self.doc_tokens = akasha.search.get_docs(
            self.db,
            self.embeddings_obj,
            retrivers_list,
            self.prompt,
            self.use_rerank,
            self.language,
            self.search_type,
            self.verbose,
            self.model_obj,
            self.max_doc_len,
            compression=self.compression,
        )

        #end_time = time.time()
        if self.system_prompt.replace(' ', '') == "":
            self.system_prompt = akasha.prompts.default_doc_ask_prompt()
        prod_sys_prompt, prod_prompt = akasha.prompts.format_sys_prompt(
            self.system_prompt, self.prompt, self.prompt_format_type)

        splitter = '\n----------------\n'


        text_input = prod_sys_prompt + "\n----------------\n" + splitter.join([doc.page_content for doc in self.docs]) +\
            splitter

        if self.verbose:
            print("Prompt after formatting:", "\n\n" + text_input)

        return text_input, prod_prompt
