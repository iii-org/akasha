from typing import Callable, Union, List, Tuple, Generator
from akasha.helper import handle_embeddings, handle_search_type, handle_model
from akasha.utils.prompts.format import handle_language, language_dict

import datetime, traceback
import warnings, logging
from akasha.utils.base import DEFAULT_CHUNK_SIZE, DEFAULT_SEARCH_TYPE, DEFAULT_EMBED, DEFAULT_MODEL, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_INPUT_TOKENS
from akasha.utils.db.db_structure import dbs


class atman:
    """basic class for akasha, implement _set_model, _change_variables, _check_db, add_log and save_logs function."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embeddings: str = DEFAULT_EMBED,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        search_type: Union[str, Callable] = DEFAULT_SEARCH_TYPE,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        threshold: float = 0.0,
        language: str = "ch",
        record_exp: str = "",
        system_prompt: str = "",
        keep_logs: bool = False,
        verbose: bool = False,
        env_file: str = "",
    ):
        """initials of atman class

        Args:
            **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.\n
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
            embeddings (_type_, optional): embedding model, including two types(openai and huggingface). Defaults to "openai:text-embedding-ada-002".
            **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
            **threshold (float, optional)**: (deprecated) the similarity threshold of searching. Defaults to 0.0.\n
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
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n
            **keep_logs (bool, optional)**: record logs or not. Defaults to False.\n
            **max_output_tokens (int, optional)**: max output tokens of llm model. Defaults to 1024.\n
            **max_input_tokens (int, optional)**: max input tokens of llm model. Defaults to 3000.\n
            **env_file (str, optional)**: the path of the .env file. Defaults to "".\n
        """

        self.chunk_size = chunk_size
        self.verbose = verbose
        self.threshold = threshold
        self.language = handle_language(language)
        self.search_type = handle_search_type(search_type, self.verbose)
        self.record_exp = record_exp
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.keep_logs = keep_logs
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens
        self.env_file = env_file
        self.timestamp_list = []
        self.logs = {}
        self.db = None
        self.ignored_files = []
        ### set model and embeddings ###
        self.model_obj = handle_model(model, self.verbose, self.temperature,
                                      self.max_output_tokens, self.env_file)
        self.model = handle_search_type(model)

        self.embeddings_obj = handle_embeddings(embeddings, self.verbose,
                                                self.env_file)
        self.embeddings = handle_search_type(embeddings)

    def _set_model(self, **kwargs):
        """change model, embeddings, search_type, temperature if user use **kwargs to change them."""
        ## check if we need to change db, model_obj or embeddings_obj ##
        if "search_type" in kwargs:
            self.search_type_str = handle_search_type(kwargs["search_type"],
                                                      self.verbose)

        if ("embeddings" in kwargs) or ("env_file" in kwargs):
            new_embeddings = self.embeddings
            new_env_file = self.env_file
            if "embeddings" in kwargs:
                new_embeddings = kwargs["embeddings"]
            if "env_file" in kwargs:
                new_env_file = kwargs["env_file"]

            if new_embeddings != self.embeddings or new_env_file != self.env_file:
                self.embeddings_obj = handle_embeddings(
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
                self.model_obj = handle_model(new_model, self.verbose,
                                              new_temp, new_tokens,
                                              new_env_file)

    def _change_variables(self, **kwargs):
        """change other arguments if user use **kwargs to change them."""
        ### check input argument is valid or not ###
        for key, value in kwargs.items():

            if (key == "model"
                    or key == "embeddings") and key in self.__dict__:
                self.__dict__[key] = handle_search_type(value)

            elif key == "language":
                self.language = handle_language(value)

            elif key in self.__dict__:  # check if variable exist
                if (getattr(self, key, None)
                        != value):  # check if variable value is different
                    self.__dict__[key] = value
            else:
                if key != "dbs":
                    logging.warning(f"argument {key} not exist")

    def _check_doc_path(doc_path: Union[List[str], str, dbs]):
        if isinstance(doc_path, dbs):
            return "use dbs object"

        return doc_path

    def _check_db(self) -> bool:
        """check if user input doc_path is exist or not

        Returns:
            _type_: _description_
        """
        if self.db is None:
            info = "document path not exist or don't have any file.\n"
            raise OSError(info)

        return True

    def _get_db(self, data: Union[List[str], str, dbs]):

        if isinstance(data, dbs):
            self.db = data
            self.ignored_files = []
        elif self.use_chroma:
            self.db, self.ignored_files = akasha.db.get_db_from_chromadb(
                self.data_source, self.embeddings)
        else:
            self.db, self.ignored_files = akasha.db.processMultiDB(
                self.data_source, self.verbose, self.embeddings_obj,
                self.chunk_size, self.ignore_check)
        return

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

        self.logs[timestamp]["language"] = language_dict[self.language]
        self.logs[timestamp]["temperature"] = self.temperature
        self.logs[timestamp]["max_input_tokens"] = self.max_input_tokens

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
        found_ref = "----------------\n" + splitter.join(
            [doc.page_content for doc in self.docs]) + splitter

        if self.verbose:
            print("Reference: \n\n" + found_ref)

        return found_ref