import pathlib
import time
from typing import Callable, Union, List, Tuple, Generator
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import akasha.helper as helper
import akasha.search as search
import akasha.format as format
import akasha.prompts as prompts
import akasha.db
import datetime, traceback
import warnings, logging
import os
from dotenv import load_dotenv
from warnings import warn

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
_DEFAULT_MAX_DOC_LEN = 1500
load_dotenv(pathlib.Path().cwd() / ".env")


def aiido_upload(
    exp_name,
    params: dict = {},
    metrics: dict = {},
    table: dict = {},
    path_name: str = "",
):
    """upload params_metrics, table to mlflow server for tracking.

    Args:
        **exp_name (str)**: experiment name on the tracking server, if not found, will create one .\n
        **params (dict, optional)**: parameters dictionary. Defaults to {}.\n
        **metrics (dict, optional)**: metrics dictionary. Defaults to {}.\n
        **table (dict, optional)**: table dictionary, used to compare text context between different runs in the experiment. Defaults to {}.\n
    """
    import aiido
    if path_name is None:
        path_name = ""
    if "model" not in params or "embeddings" not in params:
        aiido.init(experiment=exp_name, run=path_name)

    else:
        mod = params["model"].split(":")
        emb = params["embeddings"].split(":")[0]
        sea = params["search_type"]
        aiido.init(experiment=exp_name,
                   run=emb + "-" + sea + "-" + "-".join(mod))

    aiido.log_params_and_metrics(params=params, metrics=metrics)

    if len(table) > 0:
        aiido.mlflow.log_table(table, "table.json")
    aiido.mlflow.end_run()


def detect_exploitation(
    texts: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
):
    """check the given texts have harmful or sensitive information

    Args:
        **texts (str)**: texts that we want llm to check.\n
        **model (str, optional)**: llm model name. Defaults to "openai:gpt-3.5-turbo".\n
        **verbose (bool, optional)**: show log texts or not. Defaults to False.\n

    Returns:
        str: response from llm
    """

    model = helper.handle_model(model, verbose, 0.0)
    sys_b, sys_e = "<<SYS>>\n", "\n<</SYS>>\n\n"
    system_prompt = (
        "[INST]" + sys_b +
        "check if below texts have any of Ethical Concerns, discrimination, hate speech, "
        +
        "illegal information, harmful content, Offensive Language, or encourages users to share or access copyrighted materials"
        + " And return true or false. Texts are: " + sys_e + "[/INST]")

    template = system_prompt + f""" 
    
    Texts: {texts}
    Answer: """

    response = helper.call_model(model, template)

    print(response)
    return response


def openai_vision(
    pic_path: Union[str, List[str]],
    prompt: str,
    model: str = "gpt-4-vision-preview",
    max_token: int = 3000,
    verbose: bool = False,
    record_exp: str = "",
):
    start_time = time.time()

    ### process input message ###
    base64_pic = []
    pic_message = []
    if isinstance(pic_path, str):
        pic_path = [pic_path]

    for path in pic_path:
        if not pathlib.Path(path).exists():
            print(f"image path {path} not exist")
        else:
            base64_pic.append(helper.image_to_base64(path))

    for pic in base64_pic:
        pic_message.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{pic}",
                "detail": "auto",
            },
        })
    content = [{"type": "text", "text": prompt}]
    content.extend(pic_message)

    ### call model ###
    import os
    from langchain_openai import ChatOpenAI, AzureChatOpenAI
    from langchain.schema.messages import HumanMessage, SystemMessage
    from langchain.callbacks import get_openai_callback

    if ("AZURE_API_TYPE" in os.environ and os.environ["AZURE_API_TYPE"]
            == "azure") or ("OPENAI_API_TYPE" in os.environ
                            and os.environ["OPENAI_API_TYPE"] == "azure"):
        modeln = model.replace(".", "")
        api_base, api_key, api_version = helper._handle_azure_env()
        chat = AzureChatOpenAI(
            name=modeln,
            deployment_name=modeln,
            temperature=0.0,
            base_url=api_base,
            api_key=api_key,
            api_version=api_version,
            max_tokens=max_token,
        )
    else:
        chat = ChatOpenAI(model=model,
                          max_tokens=max_token,
                          temperature=0.0,
                          verbose=verbose)
    input_message = [HumanMessage(content=content)]

    with get_openai_callback() as cb:
        try:
            ret = chat.invoke(input_message).content
        except:
            chat = ChatOpenAI(model="gpt-4-vision-preview",
                              max_tokens=max_token,
                              temperature=0.0)
            ret = chat.invoke(input_message).content

        tokens, prices = cb.total_tokens, cb.total_cost

    end_time = time.time()
    if record_exp != "":
        params = format.handle_params(model, "", "", "", "", "", "ch")
        metrics = format.handle_metrics(0, end_time - start_time, tokens)
        table = format.handle_table(prompt, "\n".join(pic_path), ret)
        aiido_upload(record_exp, params, metrics, table)
    print("\n\n\ncost:", round(prices, 3))

    return ret


class atman:
    """basic class for akasha, implement _set_model, _change_variables, _check_db, add_log and save_logs function."""

    def __init__(
        self,
        chunk_size: int = 1000,
        model: str = DEFAULT_MODEL,
        verbose: bool = False,
        topK: int = -1,
        threshold: float = 0.1,
        language: str = "ch",
        search_type: Union[str, Callable] = "svm",
        record_exp: str = "",
        system_prompt: str = "",
        max_doc_len: int = 1500,
        temperature: float = 0.0,
        keep_logs: bool = False,
        max_output_tokens: int = 1024,
        max_input_tokens: int = 3000,
        env_file: str = "",
    ):
        """initials of atman class

        Args:
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
            **max_doc_len (int, optional)**: max document size of llm input. Defaults to 1500.\n (deprecated in 1.0.0)
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n
            **keep_logs (bool, optional)**: record logs or not. Defaults to False.\n
            **max_output_tokens (int, optional)**: max output tokens of llm model. Defaults to 1024.\n
            **max_input_tokens (int, optional)**: max input tokens of llm model. Defaults to 3000.\n
            **env_file (str, optional)**: the path of the .env file. Defaults to "".\n
        """
        if max_doc_len != _DEFAULT_MAX_DOC_LEN:
            warn(
                "max_doc_len is deprecated and will be removed in future 1.0.0 version",
                DeprecationWarning)
        self.chunk_size = chunk_size
        self.model = model
        self.verbose = verbose
        self.topK = topK
        self.threshold = threshold
        self.language = format.handle_language(language)
        self.search_type_str = helper.handle_search_type(
            search_type, self.verbose)
        self.record_exp = record_exp
        self.system_prompt = system_prompt
        self.max_doc_len = max_doc_len
        self.temperature = temperature
        self.keep_logs = keep_logs
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens
        self.env_file = env_file
        self.timestamp_list = []
        if topK != -1:
            warnings.warn(
                "The 'topK' parameter is deprecated and will be removed in future versions",
                DeprecationWarning)

    def _set_model(self, **kwargs):
        """change model, embeddings, search_type, temperature if user use **kwargs to change them."""
        ## check if we need to change db, model_obj or embeddings_obj ##
        if "search_type" in kwargs:
            self.search_type_str = helper.handle_search_type(
                kwargs["search_type"], self.verbose)

        if ("embeddings" in kwargs) or ("env_file" in kwargs):
            new_embeddings = self.embeddings
            new_env_file = self.env_file
            if "embeddings" in kwargs:
                new_embeddings = kwargs["embeddings"]
            if "env_file" in kwargs:
                new_env_file = kwargs["env_file"]

            if new_embeddings != self.embeddings or new_env_file != self.env_file:
                self.embeddings_obj = helper.handle_embeddings(
                    new_embeddings, self.verbose, new_env_file)

        if ("model" in kwargs) or ("temperature" in kwargs) or (
                "max_output_tokens" in kwargs) or ("env_file" in kwargs):
            new_temp = self.temperature
            new_model = self.model
            new_tokens = self.max_output_tokens
            new_env_file = self.env_file
            if "temperature" in kwargs:
                new_temp = kwargs["temperature"]
            if "model" in kwargs:
                new_model = kwargs["model"]
            if "max_output_tokens" in kwargs:
                new_tokens = kwargs["max_output_tokens"]
            if "env_file" in kwargs:
                new_env_file = kwargs["env_file"]
            if (new_model != self.model) or (new_temp != self.temperature) or (
                    new_tokens
                    != self.max_output_tokens) or (new_env_file
                                                   != self.env_file):
                self.model_obj = helper.handle_model(new_model, self.verbose,
                                                     new_temp, new_tokens,
                                                     new_env_file)

    def _change_variables(self, **kwargs):
        """change other arguments if user use **kwargs to change them."""
        ### check input argument is valid or not ###
        for key, value in kwargs.items():
            if (key == "max_doc_len") and value != _DEFAULT_MAX_DOC_LEN:

                warn(
                    "max_doc_len is deprecated and will be removed in future 1.0.0 version",
                    DeprecationWarning)
            if (key == "model"
                    or key == "embeddings") and key in self.__dict__:
                self.__dict__[key] = helper.handle_search_type(value)

            elif key == "language":
                self.language = format.handle_language(value)

            elif key in self.__dict__:  # check if variable exist
                if (getattr(self, key, None)
                        != value):  # check if variable value is different
                    self.__dict__[key] = value
            else:
                if key != "dbs":
                    logging.warning(f"argument {key} not exist")

    def _check_db(self) -> bool:
        """check if user input doc_path is exist or not

        Returns:
            _type_: _description_
        """
        if self.db is None:
            info = "document path not exist or don't have any file.\n"
            raise OSError(info)

        return True

    def _add_basic_log(self, timestamp: str, fn_type: str):
        """add pre-process log to self.logs

        Args:
            timestamp (str): timestamp of this run
            fn_type (str): function type of this run
        """
        if self.keep_logs == False:
            return

        if timestamp not in self.logs:
            self.logs[timestamp] = {}
        self.logs[timestamp]["fn_type"] = fn_type
        self.logs[timestamp]["model"] = self.model
        self.logs[timestamp]["chunk_size"] = self.chunk_size
        self.logs[timestamp]["topK"] = self.topK
        self.logs[timestamp]["threshold"] = self.threshold
        self.logs[timestamp]["language"] = format.language_dict[self.language]
        self.logs[timestamp]["temperature"] = self.temperature
        self.logs[timestamp]["max_input_tokens"] = self.max_input_tokens
        self.logs[timestamp]["doc_path"] = self.doc_path

    def _add_result_log(self, timestamp: str, time: float):
        """add post-process log to self.logs

        Args:
            timestamp (str): timestamp of this run
            time (float): spent time of this run
        """

        if self.keep_logs == False:
            return

        self.logs[timestamp]["time"] = time
        self.logs[timestamp]["doc_length"] = self.doc_length
        self.logs[timestamp]["doc_tokens"] = self.doc_tokens
        if hasattr(self, 'question') and self.question:
            self.logs[timestamp]["question"] = self.question
        if hasattr(self, 'answer') and self.answer:
            self.logs[timestamp]["answer"] = self.answer
        try:
            self.logs[timestamp]["docs"] = "\n\n".join(
                [doc.page_content for doc in self.docs])
            self.logs[timestamp]["doc_metadata"] = "\n\n".join([
                doc.metadata["source"] + "    page: " +
                str(doc.metadata["page"]) for doc in self.docs
            ])
        except:
            try:
                self.logs[timestamp]["doc_metadata"] = "none"
                self.logs[timestamp]["docs"] = "\n\n".join(
                    [doc for doc in self.docs])
            except:
                self.logs[timestamp]["doc_metadata"] = "none"
                self.logs[timestamp]["docs"] = "\n\n".join(
                    [doc.page_content for doc in self.docs])
        self.logs[timestamp]["system_prompt"] = self.system_prompt

    def save_logs(self, file_name: str = "", file_type: str = "json"):
        """save logs into json or txt file

        Args:
            file_name (str, optional): file path and the file name. if not assign, use logs/{current time}. Defaults to "".
            file_type (str, optional): the extension of the file, can be txt or json. Defaults to "json".

        Returns:
            plain_text(str): string of the log
        """

        extension = ""
        ## set extension ##
        if file_name != "":
            tst_file = file_name.split(".")[-1]
            if file_type == "json":
                if tst_file != "json" and tst_file != "JSON":
                    extension = ".json"
            else:
                if tst_file != "txt" and tst_file != "TXT":
                    extension = ".txt"

        ## set filename if not given ##
        from pathlib import Path

        if file_name == "":
            file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            logs_path = Path("logs")
            if not logs_path.exists():
                logs_path.mkdir()
            file_path = Path("logs/" + file_name + extension)
        else:
            file_path = Path(file_name + extension)

        ## write file ##
        if file_type == "json":
            import json

            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(self.logs, fp, indent=4, ensure_ascii=False)

        else:
            with open(file_path, "w", encoding="utf-8") as fp:
                for key in self.logs:
                    text = key + ":\n"
                    fp.write(text)
                    for k in self.logs[key]:
                        if type(self.logs[key][k]) == list:
                            text = (k + ": " + "\n".join(
                                [str(w) for w in self.logs[key][k]]) + "\n\n")

                        else:
                            text = k + ": " + str(self.logs[key][k]) + "\n\n"

                        fp.write(text)

                    fp.write("\n\n\n\n")

        print("save logs to " + str(file_path))
        return

    def _display_docs(self, ) -> str:
        splitter = '\n----------------\n'
        return "----------------\n" + splitter.join(
            [doc.page_content for doc in self.docs]) + splitter


class Doc_QA(atman):
    """class for implement search db based on user prompt and generate response from llm model, include get_response and chain_of_thoughts."""

    def __init__(
        self,
        embeddings: str = "openai:text-embedding-ada-002",
        chunk_size: int = 1000,
        model: str = DEFAULT_MODEL,
        verbose: bool = False,
        topK: int = -1,
        threshold: float = 0.1,
        language: str = "ch",
        search_type: Union[str, Callable] = "svm",
        record_exp: str = "",
        system_prompt: str = "",
        prompt_format_type: str = "gpt",
        max_doc_len: int = 1500,
        temperature: float = 0.0,
        keep_logs: bool = False,
        max_output_tokens: int = 1024,
        compression: bool = False,
        use_chroma: bool = False,
        use_rerank: bool = False,
        ignore_check: bool = False,
        stream: bool = False,
        max_input_tokens: int = 3000,
        env_file: str = "",
    ):
        """initials of Doc_QA class

        Args:
            embeddings (_type_, optional): embedding model, including two types(openai and huggingface). Defaults to "openai:text-embedding-ada-002".
            chunk_size (int, optional): the max length of each text segments. Defaults to 1000.
            model (_type_, optional): language model. Defaults to "openai:gpt-3.5-turbo".
            verbose (bool, optional): print the processing text or not. Defaults to False.
            topK (int, optional): the number of documents to be selected. Defaults to 2.
            threshold (float, optional): threshold of similarity for searching relavant documents. Defaults to 0.2.
            language (str, optional): "ch" chinese or "en" english. Defaults to "ch".
            search_type (Union[str, Callable], optional): _description_. Defaults to "svm".
            record_exp (str, optional): experiment name of aiido. Defaults to "".
            system_prompt (str, optional): the prompt you want llm to output in certain format. Defaults to "".
            prompt_format_type (str, optional): the prompt and system prompt format for the language model, including two types(gpt and llama). Defaults to "gpt".
            max_doc_len (int, optional): max total length of selected documents. Defaults to 1500. (will deprecated in 1.0.0)
            temperature (float, optional): temperature for language model. Defaults to 0.0.
            keep_logs (bool, optional): record logs or not. Defaults to False.
            compression (bool, optional): compress the selected documents or not. Defaults to False.
            use_chroma (bool, optional): use chroma db name instead of documents path to load data or not. Defaults to False.
            use_rerank (bool, optional): use rerank model to re-rank the selected documents or not. Defaults to False.
            ignore_check (bool, optional): speed up loading data if the chroma db is already existed. Defaults to False.
            max_output_tokens (int, optional): max output tokens of llm model. Defaults to 1024.\n
            max_input_tokens (int, optional): max input tokens of llm model. Defaults to 3000.\n
            env_file (str, optional): the path of the .env file. Defaults to "".\n
        """

        super().__init__(chunk_size, model, verbose, topK, threshold, language,
                         search_type, record_exp, system_prompt, max_doc_len,
                         temperature, keep_logs, max_output_tokens,
                         max_input_tokens, env_file)
        ### set argruments ###
        self.doc_path = ""
        self.compression = compression
        self.use_chroma = use_chroma
        self.ignore_check = ignore_check
        self.use_rerank = use_rerank
        self.prompt_format_type = prompt_format_type
        ### set variables ###
        self.logs = {}
        self.model_obj = helper.handle_model(model, self.verbose,
                                             self.temperature,
                                             self.max_output_tokens,
                                             self.env_file)
        self.embeddings_obj = helper.handle_embeddings(embeddings,
                                                       self.verbose,
                                                       self.env_file)
        self.embeddings = helper.handle_search_type(embeddings)
        self.model = helper.handle_search_type(model)
        self.search_type = search_type
        self.db = None
        self.docs = []
        self.doc_tokens = 0
        self.doc_length = 0
        self.response = ""
        self.prompt = ""
        self.ignored_files = []
        self.stream = stream

    def _truncate_docs(self, text: str) -> List[str]:
        """truncate documents if the total length of documents exceed the max_input_tokens

        Returns:
            text (str): string of documents texts
            
        """

        new_docs = []
        tot_len = len(text)
        idx = 2
        truncate_content = text[:(tot_len // idx)]
        # truncate_len = helper.get_doc_length(self.language, truncate_content)
        truncated_token_len = helper.myTokenizer.compute_tokens(
            truncate_content, self.model)
        while truncated_token_len > self.max_input_tokens:
            idx *= 2
            truncate_content = text[:(tot_len // idx)]
            truncated_token_len = helper.myTokenizer.compute_tokens(
                truncate_content, self.model)

        rge = tot_len // idx
        st = 0
        ed = rge
        while st < tot_len:
            new_docs.append(text[st:ed])
            st += rge
            ed += rge

        return new_docs

    def _separate_docs(self,
                       history_messages: list = []) -> Tuple[List[str], int]:
        """separate documents if the total length of documents exceed the max_input_tokens

        Returns:
            ret (List[str]): list of string of separated documents texts
            tot_len (int): the length of total documents
            
        """
        tot_len = 0
        cur_len = 0

        left_tokens = self.max_input_tokens - helper.myTokenizer.compute_tokens(
            self.prompt, self.model) - helper.myTokenizer.compute_tokens(
                '\n\n'.join(history_messages), self.model)
        ret = [""]
        for db_doc in self.docs:

            cur_token_len = helper.myTokenizer.compute_tokens(
                db_doc.page_content, self.model)
            if cur_len + cur_token_len > left_tokens:
                if cur_token_len <= left_tokens:
                    cur_len = cur_token_len
                    ret.append(db_doc.page_content)
                else:
                    new_docs = self._truncate_docs(db_doc.page_content)
                    ret.extend(new_docs)
                    ret.append("")
                    cur_len = 0

                tot_len += cur_token_len
                continue

            cur_len += cur_token_len
            ret[-1] += db_doc.page_content + "\n"
            tot_len += cur_token_len

        ## remove the last empty string ##
        if ret[-1] == "":
            ret = ret[:-1]

        return ret, tot_len

    def get_response(self,
                     doc_path: Union[List[str], str, akasha.db.dbs],
                     prompt: str,
                     history_messages: list = [],
                     **kwargs):
        """input the documents directory path and question, will first store the documents
        into vectors db (chromadb), then search similar documents based on the prompt question.
        llm model will use these documents to generate the response of the question.

            Args:
                **doc_path (str)**: documents directory path\n
                **prompt (str)**:question you want to ask.\n
                **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
                embeddings, chunk_size, model, verbose, topK, threshold, language , search_type, record_exp,
                system_prompt, max_doc_len, temperature.

            Returns:
                response (str): the response from llm model.
        """

        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        if isinstance(doc_path, akasha.db.dbs):
            self.doc_path = "use dbs object"
        else:
            self.doc_path = doc_path
        self.prompt = prompt
        search_dict = {}

        if isinstance(doc_path, akasha.db.dbs):
            self.db = doc_path
            self.ignored_files = []
        elif self.use_chroma:
            self.db, self.ignored_files = akasha.db.get_db_from_chromadb(
                self.doc_path, self.embeddings)
        else:
            self.db, self.ignored_files = akasha.db.processMultiDB(
                self.doc_path, self.verbose, self.embeddings_obj,
                self.embeddings, self.chunk_size, self.ignore_check)

        start_time = time.time()
        self._check_db()
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        if self.keep_logs == True:
            self.timestamp_list.append(timestamp)
            self._add_basic_log(timestamp, "get_response")
            self.logs[timestamp]["search_type"] = self.search_type_str
            self.logs[timestamp]["embeddings"] = self.embeddings
            self.logs[timestamp]["compression"] = self.compression

        ### start to get response ###
        retrivers_list = search.get_retrivers(self.db, self.embeddings_obj,
                                              self.use_rerank, self.threshold,
                                              self.search_type, search_dict)
        self.docs, self.doc_length, self.doc_tokens = search.get_docs(
            self.db,
            self.embeddings_obj,
            retrivers_list,
            self.prompt,
            self.use_rerank,
            self.language,
            self.search_type,
            self.verbose,
            self.model,
            self.max_input_tokens -
            helper.myTokenizer.compute_tokens(self.prompt, self.model) -
            helper.myTokenizer.compute_tokens('\n\n'.join(history_messages),
                                              self.model),
            compression=self.compression,
        )
        if self.docs is None:

            print("\n\nNo Relevant Documents.\n\n")
            self.docs = []

        ## format prompt ##
        if self.system_prompt.replace(' ', '') == "":
            self.system_prompt = prompts.default_doc_ask_prompt(self.language)

        text_input = helper.merge_history_and_prompt(
            history_messages, self.system_prompt,
            self._display_docs() + "User question: " + self.prompt,
            self.prompt_format_type)

        end_time = time.time()
        if self.stream:
            return helper.call_stream_model(
                self.model_obj,
                text_input,
            )
        else:

            self.response = helper.call_model(
                self.model_obj,
                text_input,
            )

            if self.keep_logs == True:
                self._add_result_log(timestamp, end_time - start_time)
                self.logs[timestamp]["prompt"] = self.prompt
                self.logs[timestamp]["response"] = self.response
                for k, v in search_dict.items():
                    self.logs[timestamp][k] = v

            if self.record_exp != "":
                params = format.handle_params(
                    self.model,
                    self.embeddings,
                    self.chunk_size,
                    self.search_type_str,
                    self.topK,
                    self.threshold,
                    self.language,
                )
                metrics = format.handle_metrics(self.doc_length,
                                                end_time - start_time,
                                                self.doc_tokens)
                table = format.handle_table(prompt, self.docs, self.response)
                aiido_upload(self.record_exp, params, metrics, table)

            return self.response

    def chain_of_thought(self,
                         doc_path: Union[List[str], str],
                         prompt_list: list,
                         history_messages: list = [],
                         **kwargs) -> list:
        """input the documents directory path and question, will first store the documents
        into vectors db (chromadb), then search similar documents based on the prompt question.
        llm model will use these documents to generate the response of the question.

        In chain_of_thought function, you can separate your question into multiple small steps so that llm can have better response.
        chain_of_thought function will use all responses from the previous prompts, and combine the documents search from current prompt to generate
        response.

        Args:
           **doc_path (str)**: documents directory path\n
            **prompt (list)**:questions you want to ask.\n
            **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
            embeddings, chunk_size, model, verbose, topK, threshold, language , search_type, record_exp,
            system_prompt, max_input_tokens, temperature.

        Returns:
            response (list): the responses from llm model.
        """

        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        if self.system_prompt.replace(' ', '') == "":
            self.system_prompt = prompts.default_doc_ask_prompt(self.language)

        if isinstance(doc_path, akasha.db.dbs):
            self.doc_path = "use dbs object"
        else:
            self.doc_path = doc_path
        table = {}
        search_dict = {}

        if isinstance(doc_path, akasha.db.dbs):
            self.db = doc_path
            self.ignored_files = []
        elif self.use_chroma:
            self.db, self.ignored_files = akasha.db.get_db_from_chromadb(
                self.doc_path, self.embeddings)
        else:
            self.db, self.ignored_files = akasha.db.processMultiDB(
                self.doc_path, self.verbose, self.embeddings_obj,
                self.embeddings, self.chunk_size, self.ignore_check)

        start_time = time.time()
        self._check_db()
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        if self.keep_logs == True:
            self.timestamp_list.append(timestamp)
            self._add_basic_log(timestamp, "chain_of_thought")
            self.logs[timestamp]["search_type"] = self.search_type_str
            self.logs[timestamp]["embeddings"] = self.embeddings
            self.logs[timestamp]["compression"] = self.compression

        self.doc_tokens = 0
        self.doc_length = 0
        self.response = []
        self.docs = []
        self.prompt = []
        total_docs = []
        retrivers_list = search.get_retrivers(self.db, self.embeddings_obj,
                                              self.use_rerank, self.threshold,
                                              self.search_type, search_dict)

        def recursive_get_response(prompt_list):
            pre_result = []
            for prompt in prompt_list:
                if isinstance(prompt, list):
                    response = recursive_get_response(prompt)
                    pre_result.append(Document(page_content="".join(response)))
                else:
                    self.prompt.append(prompt)
                    merge_prompts = ''.join(self.prompt)
                    docs, docs_len, tokens = search.get_docs(
                        self.db,
                        self.embeddings_obj,
                        retrivers_list,
                        prompt,
                        self.use_rerank,
                        self.language,
                        self.search_type,
                        self.verbose,
                        self.model,
                        self.max_input_tokens -
                        helper.myTokenizer.compute_tokens(
                            merge_prompts, self.model) -
                        helper.myTokenizer.compute_tokens(
                            '\n\n'.join(history_messages), self.model),
                        compression=self.compression,
                    )
                    total_docs.extend(docs)
                    self.doc_length += docs_len
                    self.doc_tokens += tokens
                    self.docs = docs + pre_result
                    ## format prompt ##

                    text_input = helper.merge_history_and_prompt(
                        history_messages, self.system_prompt,
                        self._display_docs() + "User question: " + prompt,
                        self.prompt_format_type)

                    response = helper.call_model(
                        self.model_obj,
                        text_input,
                    )

                    self.response.append(response)
                    pre_result.append(Document(page_content="".join(response)))

                    new_table = format.handle_table(prompt, docs, response)
                    for key in new_table:
                        if key not in table:
                            table[key] = []
                        table[key].append(new_table[key])
            pre_result = []
            return response

        recursive_get_response(prompt_list)
        end_time = time.time()
        self.docs = total_docs
        if self.keep_logs == True:
            self._add_result_log(timestamp, end_time - start_time)
            self.logs[timestamp]["prompt"] = self.prompt
            self.logs[timestamp]["response"] = self.response
            for k, v in search_dict.items():
                self.logs[timestamp][k] = v

        if self.record_exp != "":
            params = format.handle_params(
                self.model,
                self.embeddings,
                self.chunk_size,
                self.search_type_str,
                self.topK,
                self.threshold,
                self.language,
            )
            metrics = format.handle_metrics(self.doc_length,
                                            end_time - start_time,
                                            self.doc_tokens)
            aiido_upload(self.record_exp, params, metrics, table)

        return self.response

    def ask_whole_file(self, file_path: str, prompt: str, **kwargs) -> str:
        """input the documents directory path and question, will first store the documents
        into vectors db (chromadb), then search similar documents based on the prompt question.
        llm model will use these documents to generate the response of the question.

            Args:
                **file_path (str)**: document file path\n
                **prompt (str)**:question you want to ask.\n
                **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
                embeddings, chunk_size, model, verbose, topK, threshold, language , search_type, record_exp,
                system_prompt, max_input_tokens, temperature.

            Returns:
                response (str): the response from llm model.
        """
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.prompt = prompt
        self.file_path = file_path
        self.docs = akasha.db._load_file(file_path, file_path.split('.')[-1])

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        start_time = time.time()
        if self.keep_logs == True:
            self.timestamp_list.append(timestamp)
            self._add_basic_log(timestamp, "ask_whole_file")
            self.logs[timestamp]["embeddings"] = "ask_whole_file"

        if self.docs is None:
            logging.error("No Relevant Documents.")
            raise AttributeError("No Relevant Documents.")

        ### start to get response ###
        cur_documents, self.doc_tokens = self._separate_docs()

        self.doc_length = helper.get_doc_length(self.language,
                                                ''.join(cur_documents))

        ## format prompt ##
        if self.system_prompt.replace(' ', '') == "":
            self.system_prompt = prompts.default_doc_ask_prompt(self.language)
        prod_sys_prompts = []

        for d_count in range(len(cur_documents)):
            prod_sys_prompt = prompts.format_sys_prompt(
                self.system_prompt + cur_documents[d_count], prompt,
                self.prompt_format_type)
            prod_sys_prompts.append(prod_sys_prompt)

        if len(cur_documents) == 1:

            if self.stream:
                return helper.call_stream_model(
                    self.model_obj,
                    prod_sys_prompts[0],
                )
            else:
                self.response = helper.call_model(self.model_obj,
                                                  prod_sys_prompts[0])

        else:
            ### call batch model ##
            batch_responses = helper.call_batch_model(
                self.model_obj,
                prod_sys_prompts,
            )

            ## check relevant answer if batch_responses > 5 ##
            if len(batch_responses) > 5:
                batch_responses = helper.check_relevant_answer(
                    self.model_obj, batch_responses, self.prompt,
                    self.prompt_format_type)

            fnl_input = akasha.prompts.format_sys_prompt(
                prompts.default_conclusion_prompt(prompt, self.language),
                "\n\n".join(batch_responses), self.prompt_format_type)

            if self.stream:
                return helper.call_stream_model(
                    self.model_obj,
                    fnl_input,
                )

            self.response = helper.call_model(self.model_obj, fnl_input)

            end_time = time.time()
            if self.keep_logs == True:
                self._add_result_log(timestamp, end_time - start_time)
                self.logs[timestamp]["prompt"] = self.prompt
                self.logs[timestamp]["response"] = self.response

            if self.record_exp != "":
                params = format.handle_params(
                    self.model,
                    self.embeddings,
                    self.chunk_size,
                    self.search_type_str,
                    self.topK,
                    self.threshold,
                    self.language,
                )
                metrics = format.handle_metrics(self.doc_length,
                                                end_time - start_time,
                                                self.doc_tokens)
                table = format.handle_table(prompt, self.docs, self.response)
                aiido_upload(self.record_exp, params, metrics, table)

        return self.response

    def ask_self(self,
                 prompt: str,
                 info: Union[str, list] = "",
                 history_messages: list = [],
                 **kwargs) -> str:
        """input information and question, llm model will use the information to generate the response of the question.

            Args:
                **info (str,list)**: document file path\n
                **prompt (str)**:question you want to ask.\n
                **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
                embeddings, chunk_size, model, verbose, topK, threshold, language , search_type, record_exp,
                system_prompt, max_input_tokens, temperature.

            Returns:
                response (str): the response from llm model.
        """
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.prompt = prompt
        if isinstance(info, str):
            self.docs = [Document(page_content=info)]
        else:
            self.docs = [Document(page_content=i) for i in info]

        start_time = time.time()

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")

        if self.keep_logs == True:
            self.timestamp_list.append(timestamp)
            self._add_basic_log(timestamp, "ask_self")
            self.logs[timestamp]["embeddings"] = "ask_self"

        ### start to get response ###
        cur_documents, self.doc_tokens = self._separate_docs(history_messages)

        self.doc_length = helper.get_doc_length(self.language,
                                                ''.join(cur_documents))

        ## format prompt ##
        if self.system_prompt.replace(' ', '') == "":
            self.system_prompt = prompts.default_doc_ask_prompt(self.language)
        prod_sys_prompts = []

        for d_count in range(len(cur_documents)):

            prod_sys_prompt = helper.merge_history_and_prompt(
                history_messages, self.system_prompt,
                cur_documents[d_count] + "\n\nUser question: " + self.prompt,
                self.prompt_format_type)
            prod_sys_prompts.append(prod_sys_prompt)

        ### start to get response ###
        if len(cur_documents) > 1:
            ## call batch model ##
            batch_responses = helper.call_batch_model(
                self.model_obj,
                prod_sys_prompts,
            )
            fnl_conclusion_prompt = prompts.default_conclusion_prompt(
                prompt, self.language)
            ## check relevant answer if batch_responses > 10 ##
            if len(batch_responses) > 10:
                batch_responses = helper.check_relevant_answer(
                    self.model_obj, batch_responses, self.prompt,
                    self.prompt_format_type)

            batch_responses, cur_len = retri_max_texts(
                batch_responses,
                self.max_input_tokens - helper.myTokenizer.compute_tokens(
                    fnl_conclusion_prompt, self.model), self.model)
            fnl_input = prompts.format_sys_prompt(fnl_conclusion_prompt,
                                                  "\n\n".join(batch_responses),
                                                  self.prompt_format_type)

            if self.stream:
                return helper.call_stream_model(
                    self.model_obj,
                    fnl_input,
                )

            self.response = helper.call_model(self.model_obj, fnl_input)

        else:
            prod_sys_prompt = helper.merge_history_and_prompt(
                history_messages, self.system_prompt,
                '\n\n'.join(cur_documents) + "\n\nUser question: " +
                self.prompt, self.prompt_format_type)
            if self.stream:
                return helper.call_stream_model(
                    self.model_obj,
                    prod_sys_prompt,
                )

            self.response = helper.call_model(self.model_obj, prod_sys_prompt)

        end_time = time.time()
        if self.keep_logs == True:
            self._add_result_log(timestamp, end_time - start_time)
            self.logs[timestamp]["prompt"] = self.prompt
            self.logs[timestamp]["response"] = self.response

        if self.record_exp != "":
            params = format.handle_params(
                self.model,
                self.embeddings,
                self.chunk_size,
                self.search_type_str,
                self.topK,
                self.threshold,
                self.language,
            )
            metrics = format.handle_metrics(self.doc_length,
                                            end_time - start_time,
                                            self.doc_tokens)
            table = format.handle_table(prompt, self.docs, self.response)
            aiido_upload(self.record_exp, params, metrics, table)

        return self.response

    def ask_agent(self, doc_path: Union[List[str], str], prompt: str,
                  **kwargs) -> str:
        """input the documents directory path and question, will first store the documents
        into vectors db (chromadb), then search similar documents based on the prompt question.
        question will use self-ask with search to solve complex question.
        llm model will use these documents to generate the response of the question.

            Args:
                **doc_path (str)**: documents directory path\n
                **prompt (str)**:question you want to ask.\n
                **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
                embeddings, chunk_size, model, verbose, topK, threshold, language , search_type, record_exp,
                system_prompt, max_input_tokens, temperature.

            Returns:
                response (str): the response from llm model.
        """
        self._set_model(**kwargs)
        self._change_variables(**kwargs)

        if isinstance(doc_path, akasha.db.dbs):
            self.doc_path = "use dbs object"
        else:
            self.doc_path = doc_path
        self.prompt = prompt
        self.follow_up = []
        self.intermediate_ans = []
        original_sys_prompt = self.system_prompt

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        start_time = time.time()
        ### start to get response ###

        self_ask_prompt = prompts.default_self_ask_prompt(
        ) + "Question: " + prompt

        formatter = [
            prompts.OutputSchema(
                name="need",
                description=
                "the 0 or 1 value that the question need follow up questions to answer or not.",
                type="bool"),
            prompts.OutputSchema(
                name="follow_up",
                description=
                "if need follow up questions, list of string of the follow up questions, else return empty list.",
                type="list"),
        ]

        ## format sys prompt ##
        self_ask_sys_prompt = ""
        if "chinese" in format.language_dict[self.language]:
            self_ask_sys_prompt = "用中文回答 "
        self_ask_sys_prompt += prompts.JSON_formatter(formatter)
        prod_sys_prompt = prompts.format_sys_prompt(self_ask_sys_prompt, "",
                                                    self.prompt_format_type)
        prod_prompt = prompts.format_sys_prompt("", self_ask_prompt,
                                                self.prompt_format_type)

        stream_status = self.stream
        ret = self.ask_self(prompt=prod_prompt,
                            system_prompt=prod_sys_prompt,
                            stream=False)
        self.stream = stream_status

        parse_json = akasha.helper.extract_json(ret)
        self.system_prompt = original_sys_prompt  # reset system prompt back, since the self-ask is done

        if parse_json is None or int(parse_json["need"]) == 0:

            return self.get_response(doc_path, prompt)

        self.follow_up = parse_json["follow_up"]

        self.rerun_ask_agent(doc_path, prompt, self.follow_up)

        end_time = time.time()
        self.prompt = prompt

        if self.keep_logs == True:
            self.timestamp_list.append(timestamp)
            self._add_basic_log(timestamp, "ask_agent")
            self.logs[timestamp]["search_type"] = self.search_type_str
            self.logs[timestamp]["embeddings"] = self.embeddings
            self.logs[timestamp]["compression"] = self.compression
            self._add_result_log(timestamp, end_time - start_time)
            self.logs[timestamp]["prompt"] = prompt
            self.logs[timestamp]["response"] = self.response
            self.logs[timestamp]["follow_up"] = self.follow_up
            self.logs[timestamp["intermediate_ans"]] = self.intermediate_ans

        if self.record_exp != "":
            params = format.handle_params(
                self.model,
                self.embeddings,
                self.chunk_size,
                self.search_type_str,
                self.threshold,
                self.language,
            )
            metrics = format.handle_metrics(self.doc_length,
                                            end_time - start_time,
                                            self.doc_tokens)
            table = format.handle_table(prompt, self.docs, self.response)
            aiido_upload(self.record_exp, params, metrics, table)

        return self.response

    def rerun_ask_agent(self, doc_path: Union[List[str], str], prompt: str,
                        follow_up: list, **kwargs) -> Tuple[str, list, list]:

        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        if isinstance(doc_path, akasha.db.dbs):
            self.doc_path = "use dbs object"
        else:
            self.doc_path = doc_path
        self.prompt = prompt
        self.follow_up = follow_up
        self.intermediate_ans = []
        original_sys_prompt = self.system_prompt

        if isinstance(doc_path, akasha.db.dbs):
            self.db = doc_path
            self.ignored_files = []
        elif self.use_chroma:
            self.db, self.ignored_files = akasha.db.get_db_from_chromadb(
                self.doc_path, self.embeddings)
        else:
            self.db, self.ignored_files = akasha.db.processMultiDB(
                self.doc_path, self.verbose, self.embeddings_obj,
                self.embeddings, self.chunk_size, self.ignore_check)

        for each_follow_up in self.follow_up:
            #ret = self.ask_self(prompt=follow_up, system_prompt=self.system_prompt)
            each_follow_up = '\n'.join(
                self.intermediate_ans) + "\n" + each_follow_up
            follow_up_response = self.get_response(
                doc_path,
                each_follow_up,
            )

            check = self.ask_self(
                prompt=
                f"help me check if the below Response answer the Question or not, return 1 if yes, 0 if no.\
            \nQuestion: {each_follow_up}\n\nResponse: " + follow_up_response,
                system_prompt="return 1 or 0 only")
            self.system_prompt = original_sys_prompt  # reset system prompt back, since the self-ask is done
            if int(check) == 1:
                self.intermediate_ans.append(follow_up_response)
            else:
                # agent = get_agent_buildin_tool(self.model_obj)
                # self.intermediate_ans.append(agent(follow_up)['output'])
                pass

        self.response = self.get_response(
            doc_path,
            prompt='\n'.join(self.intermediate_ans) + "\n" + prompt,
            dbs=self.db)

        return self.response, self.follow_up, self.intermediate_ans

    def ask_image(self, image_path: str, prompt: str, **kwargs) -> str:
        """ask model with image and prompt

        Args:
            image_path (str): image path or url (recommand jpeg or png file)
            prompt (str): user question

        Returns:
            str: _description_
        """
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.prompt = prompt
        fnl_input = []
        start_time = time.time()

        ## check model ##
        model_prefix = self.model.split(":")[0]
        if model_prefix in ["hf", "hugginface"] and self.stream == True:
            raise ValueError(
                "Currently huggingface model does not support stream mode.\n\n"
            )

        if model_prefix in [
                "llama-cpu", "llama-gpu", "llama", "llama2", "llama-cpp",
                "chatglm", "chatglm2", "glm", "lora", "peft", "gptq", "gptq2"
        ]:
            raise ValueError(
                f"Currently {self.model} model does not support image input.\n\n"
            )

        ## decide prompt format ##
        if model_prefix in ["hf", "huggingface"]:
            fnl_input = prompts.format_image_prompt(image_path, prompt,
                                                    "image_llama")

        else:
            fnl_input = prompts.format_image_prompt(image_path, prompt,
                                                    "image_gpt")

        if self.stream:
            return helper.call_stream_model(
                self.model_obj,
                fnl_input,
            )

        self.response = helper.call_image_model(self.model_obj, fnl_input)

        return self.response


def retri_max_texts(
        texts_list: list,
        left_token_len: int,
        model_name: str = "openai:gpt-3.5-turbo") -> Tuple[list, int]:
    """return list of texts that do not exceed the left_token_len

    Args:
        texts_list (list): _description_
        left_doc_len (int): _description_

    Returns:
        Tuple[list, int]: _description_
    """
    ret = []
    cur_len = 0
    for text in texts_list:
        txt_len = helper.myTokenizer.compute_tokens(text, model_name)
        if cur_len + txt_len > left_token_len:
            break
        cur_len += txt_len
        ret.append(text)
    return ret, cur_len
