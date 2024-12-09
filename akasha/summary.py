from langchain.text_splitter import RecursiveCharacterTextSplitter
import akasha
from pathlib import Path
import time, datetime
import akasha.db as db
from typing import Union, List
import akasha.format as afr
from tqdm import tqdm
import math
import logging


def calculate_approx_sum_times(chunks: int, per_sum_chunks: int) -> int:
    """approximate the total times of summarizing we need to do in reduce_summary method

    Args:
        chunks (int): number of chunks
        per_sum_chunks (int): number of chunks we can summarize in one time

    Returns:
        int: the total times of summarizing we need to do
    """
    times = 0
    while chunks > 1:
        chunks = math.ceil(chunks / per_sum_chunks)
        times += chunks
    return times


def calculate_per_summary_chunks(language: str, max_input_tokens: int,
                                 summary_len: int, chunk_size: int) -> int:
    """calculate the estimation of chunks that can fit into llm each time

    Args:
        language (str): texts language
        max_input_tokens (int): the max tokens we want to fit into llm model at one time
        summary_len (int): the length of summary tokens
        chunk_size (int): the chunk size of texts we want to summarize

    Returns:
        int: the estimation of chunks that can fit into llm each time
    """

    ret = 2

    if "chinese" in afr.language_dict[language]:
        token_to_text = 2
    else:
        token_to_text = 1

    ret = max(ret,
              (token_to_text * max_input_tokens - summary_len) // chunk_size)

    return ret


class Summary(akasha.atman):
    """class for implement summary text file by llm model, include summarize_file method."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 40,
        model: str = akasha.akashas.DEFAULT_MODEL,
        verbose: bool = False,
        threshold: float = 0.0,
        language: str = "ch",
        record_exp: str = "",
        format_prompt: str = "",
        system_prompt: str = "",
        max_doc_len: int = 1500,
        temperature: float = 0.0,
        keep_logs: bool = False,
        auto_translate: bool = False,
        prompt_format_type: str = "auto",
        consecutive_merge_failures: int = 5,
        max_output_tokens: int = 1024,
        max_input_tokens: int = 3000,
        env_file: str = "",
    ):
        """initials of Summary class

        Args:
            **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.\n
            **chunk_overlap (int, optional)**: chunk overlap of texts from documents. Defaults to 40.\n
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
            **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
            **threshold (float, optional)**: (deprecated) the similarity threshold of searching. Defaults to 0.2.\n
            **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
                max token size of llm input.\n
            **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set
                record_exp as experiment name.  default "".\n
            **system_prompt (str, optional)**: the system prompt that you assign special instruction to llm model, so will not be used
                in searching relevant documents. Defaults to "".\n
            **max_doc_len (int, optional)**: max doc size of llm document input. Defaults to 3000.\n (will be deprecated and remove in the future 1.0.0 versin)
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n
            **keep_logs (bool, optional)**: record logs or not. Defaults to False.\n
            **auto_translate (bool, optional)**: auto translate the summary to target language since LLM may generate different language. 
            Defaults to False.\n
            **prompt_format_type (str, optional)**: the prompt and system prompt format for the language model, including auto, gpt, llama, chat_gpt, chat_mistral, chat_gemini . Defaults to "auto".
            **consecutive_merge_failures (int, optional)**: the number of consecutive merge failures before returning the current response list as the summary. Defaults to 5.
            **max_output_tokens (int, optional)**: max output tokens of llm model. Defaults to 1024.\n
            **max_input_tokens (int, optional)**: max input tokens of llm model. Defaults to 3000.\n      
            **env_file (str, optional)**: the path of env file. Defaults to "".\n  
        """

        ### set argruments ###
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        self.threshold = threshold
        self.language = akasha.format.handle_language(language)
        self.record_exp = record_exp
        self.format_prompt = format_prompt
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.auto_translate = auto_translate
        self.prompt_format_type = prompt_format_type
        self.keep_logs = keep_logs
        self.consecutive_merge_failures = consecutive_merge_failures
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens
        self.env_file = env_file
        ### set variables ###
        self.file_name = ""
        self.summary_type = ""
        self.summary_len = 500
        self.logs = {}
        self.model_obj = akasha.helper.handle_model(model, self.verbose,
                                                    self.temperature,
                                                    self.max_output_tokens,
                                                    env_file)
        self.model = akasha.helper.handle_search_type(model)
        self.doc_tokens = 0
        self.doc_length = 0
        self.summary = ""
        self.timestamp_list = []

    def _add_log(self, fn_type: str, timestamp: str, time: float,
                 response_list: list):
        """call this method to add log to logs dictionary

        Args:
            fn_type (str): the method current running
            timestamp (str): the method current running timestamp
            time (float): the spent time of the method
            response_list (list): the response list of the method
        """
        if not self.keep_logs:
            return

        if timestamp not in self.logs:
            self.logs[timestamp] = {}
        self.logs[timestamp]["fn_type"] = fn_type
        self.logs[timestamp]["model"] = self.model
        self.logs[timestamp]["chunk_size"] = self.chunk_size

        self.logs[timestamp]["language"] = akasha.format.language_dict[
            self.language]
        self.logs[timestamp]["temperature"] = self.temperature
        self.logs[timestamp]["file_name"] = self.file_name

        self.logs[timestamp]["time"] = time
        self.logs[timestamp]["doc_length"] = self.doc_length
        self.logs[timestamp]["doc_tokens"] = self.doc_tokens
        self.logs[timestamp]["system_prompt"] = self.system_prompt
        self.logs[timestamp]["format_prompt"] = self.format_prompt
        self.logs[timestamp]["summary_type"] = self.summary_type
        self.logs[timestamp]["summary_len"] = self.summary_len
        self.logs[timestamp]["summaries_list"] = response_list
        self.logs[timestamp]["summary"] = self.summary
        self.logs[timestamp]["auto_translate"] = self.auto_translate

    def _set_model(self, **kwargs):
        """change model_obj if "model" or "temperature" changed"""

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
                self.model_obj = akasha.helper.handle_model(
                    new_model, self.verbose, new_temp, new_tokens,
                    new_env_file)

    def _save_file(self, default_file_name: str, output_file_path: str):
        ### write summary to file ###
        if output_file_path == "":
            sum_path = Path("summarization/")
            if not sum_path.exists():
                sum_path.mkdir()

            output_file_path = ("summarization/" + default_file_name)
        elif output_file_path[-4:] != ".txt":
            output_file_path = output_file_path + ".txt"

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(self.summary)
        print("summarization saved in ", output_file_path, "\n\n")

    def _reduce_summary(self, texts: list, tokens: int, total_list: list,
                        progress: tqdm, pre_response_list_len: int,
                        consecutive_merge_fail: int):
        """Summarize each chunk and merge them until the combined chunks are smaller than the maximum token limit.
        Then, generate the final summary. This method is faster and requires fewer tokens than the refine method.

        Args:
            **texts (list)**: list of texts from documents\n
            **tokens (int)**: used to save total tokens in recursive call.\n
            **total_list (list)**: used to save total response in recursive call.\n

        Returns:
            (list,int): llm response list and total tokens
        """
        response_list = []
        i = 0
        while i < len(texts):
            token, cur_text, newi = akasha.helper._get_text(
                texts, "", i, self.max_input_tokens, self.model)
            tokens += token

            progress.update(1)

            ### do the final summary if all chunks can be fits into llm model ###
            if i == 0 and newi == len(texts):
                prompt = akasha.prompts.format_reduce_summary_prompt(
                    cur_text, self.summary_len)
                input_text = akasha.prompts.format_sys_prompt(
                    self.system_prompt, "\n" + prompt, self.prompt_format_type,
                    self.model)
                response = akasha.helper.call_model(
                    self.model_obj,
                    input_text,
                )

                total_list.append(response)

                if self.verbose:
                    print("prompt: \n", self.system_prompt + prompt)
                    print("\n\n")
                    print("response: \n", response)
                    print("\n\n\n\n\n\n")

                return total_list, tokens

            prompt = akasha.prompts.format_reduce_summary_prompt(cur_text, 0)
            input_text = akasha.prompts.format_sys_prompt(
                self.system_prompt, "\n" + prompt, self.prompt_format_type,
                self.model)
            response = akasha.helper.call_model(self.model_obj, input_text)

            i = newi

            if self.verbose:
                print("prompt: \n", self.system_prompt + prompt)
                print("\n\n")
                print("response: \n", response)
                print("\n\n\n\n\n\n")
            response_list.append(response)
            total_list.append(response)

        ### handle merge fail ###
        if pre_response_list_len == len(response_list):
            consecutive_merge_fail += 1
            # if consecutive_merge_fail >= self.consecutive_merge_failures: return current response_list as summary #
            if consecutive_merge_fail >= self.consecutive_merge_failures:
                logging.warning(
                    "Cannot summarize due to texts too long, return current response_list as summary."
                )
                total_list.append('\n\n'.join(response_list))
                return total_list, tokens
        else:
            consecutive_merge_fail = 0

        pre_response_list_len = len(response_list)

        return self._reduce_summary(response_list, tokens, total_list,
                                    progress, pre_response_list_len,
                                    consecutive_merge_fail)

    def _refine_summary(self, texts: list) -> Union[list, int]:
        """refine summary summarizing a chunk at a time and using the previous summary as a prompt for
        summarizing the next chunk. This approach may be slower and require more tokens, but it results in a higher level of summary consistency.

        Args:
            **texts (list)**: list of texts from documents\n

        Returns:
            (list,int): llm response list and total tokens
        """
        ### setting variables ###
        previous_summary = ""
        i = 0
        tokens = 0
        response_list = []
        prod_sys_prompt = akasha.prompts.format_sys_prompt(
            self.system_prompt, "", self.prompt_format_type, self.model)
        ###

        ### get tqdm progress bar and handle merge fail###
        progress = tqdm(total=len(texts), desc="Refine Summary")
        consecutive_merge_fail = 0
        fnsh_sum_list = []

        while i < len(texts):
            prei = i
            token, cur_text, i = akasha.helper._get_text(
                texts, previous_summary, i, self.max_input_tokens, self.model)
            tokens += token
            if previous_summary == "":
                prompt = akasha.prompts.format_reduce_summary_prompt(
                    cur_text, self.summary_len)
            else:
                prompt = akasha.prompts.format_refine_summary_prompt(
                    cur_text, previous_summary, self.summary_len)
            text_input = akasha.prompts.format_sys_prompt(
                self.system_prompt, "\n" + prompt, self.prompt_format_type,
                self.model)
            response = akasha.helper.call_model(
                self.model_obj,
                text_input,
            )

            if self.verbose:
                print("prompt: \n", self.system_prompt + prompt)
                print("\n\n")
                print("resposne: \n", response)
                print("\n\n\n\n\n\n")
            response_list.append(response)
            previous_summary = response
            progress.update(i - prei)

            ### handle merge fail ###
            if prei == i:
                consecutive_merge_fail += 1
                if consecutive_merge_fail >= self.consecutive_merge_failures:
                    logging.warning(
                        "Cannot summarize current chunk due to texts too long, skip summarize current chunk."
                    )
                    fnsh_sum_list.append(response)
                    consecutive_merge_fail = 0
                    previous_summary = ""

            else:
                consecutive_merge_fail = 0

        progress.close()
        ## merge the failed summaries ##
        if len(fnsh_sum_list) > 0:
            response_list[-1] = '\n\n'.join(
                fnsh_sum_list) + '\n\n' + response_list[-1]

        return response_list, tokens

    def _handle_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return texts

    def summarize_file(self,
                       file_path: str,
                       summary_type: str = "map_reduce",
                       summary_len: int = 500,
                       output_file_path: Union[str, None] = None,
                       **kwargs) -> str:
        """input a file path and return a summary of the file

        Args:
            **file_path (str)**:  the path of file you want to summarize, can be '.txt', '.docx', '.pdf' file.\n
            **summary_type (str, optional)**: summary method, "map_reduce" or "refine". Defaults to "map_reduce".\n
            **summary_len (int, optional)**: expected output length. Defaults to 500.\n
            **output_file_path (str, optional)**: the path of output file. Defaults to "".\n
            **kwargs: the arguments you set in the initial of the class, you can change it here. Include:\n
                chunk_size, chunk_overlap, model, verbose, language , record_exp,
                system_prompt, max_input_tokens, temperature.
        Returns:
            str: the summary of the file
        """

        ## set variables ##
        self.file_name = file_path
        self.summary_type = summary_type.lower()
        self.summary_len = summary_len
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        start_time = time.time()
        table = {}
        if not akasha.helper.is_path_exist(file_path):
            print("file path not exist\n\n")
            return ""

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")

        # Split the documents into sentences
        documents = db._load_file(self.file_name,
                                  self.file_name.split(".")[-1])
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", " ", ",", ".", "。", "!"],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        docs = text_splitter.split_documents(documents)
        self.doc_length = akasha.helper.get_docs_length(self.language, docs)
        texts = [doc.page_content for doc in docs]

        if summary_type == "refine":
            response_list, self.doc_tokens = self._refine_summary(texts)

        else:
            per_sum_chunks = calculate_per_summary_chunks(
                self.language, self.max_input_tokens, self.summary_len,
                self.chunk_size)
            approx_sum_times = calculate_approx_sum_times(
                len(texts), per_sum_chunks)

            response_list_len = len(texts)
            consecutive_merge_fail = 0

            progress = tqdm(total=approx_sum_times, desc="Reduce_map Summary")
            response_list, self.doc_tokens = self._reduce_summary(
                texts, 0, [], progress, response_list_len,
                consecutive_merge_fail)
            progress.close()

        self.summary = response_list[-1]
        p = akasha.prompts.format_refine_summary_prompt(
            "", "", self.summary_len)

        ### write summary to file, and if auto_translate is True , translate it ###
        if self.format_prompt != "":

            input_text = akasha.prompts.format_sys_prompt(
                self.format_prompt, "\n\n" + self.summary,
                self.prompt_format_type, self.model)
            self.summary = akasha.helper.call_model(
                self.model_obj,
                input_text,
            )

        if self.auto_translate:

            self.summary = akasha.helper.call_translator(
                self.model_obj, self.summary, self.prompt_format_type,
                self.language)

        ## change sim to trad if target language is traditional chinese ##
        if afr.language_dict[self.language] == "traditional chinese":
            self.summary = akasha.helper.sim_to_trad(self.summary)
            #self.summary = self.summary.replace("。", "。\n\n")

        end_time = time.time()
        if self.keep_logs == True:
            self.timestamp_list.append(timestamp)
            self._add_log("summarize_file", timestamp, end_time - start_time,
                          response_list)
        print(self.summary, "\n\n\n\n")

        if self.record_exp != "":
            params = akasha.format.handle_params(self.model, "",
                                                 self.chunk_size, "", -1, -1.0,
                                                 self.language, False)
            params["chunk_overlap"] = self.chunk_overlap
            params["summary_type"] = ("refine" if summary_type == "refine" else
                                      "map_reduce")
            metrics = akasha.format.handle_metrics(self.doc_length,
                                                   end_time - start_time,
                                                   self.doc_tokens)
            table = akasha.format.handle_table(p, response_list, self.summary)
            akasha.aiido_upload(self.record_exp, params, metrics, table,
                                output_file_path)

        if output_file_path is not None:
            self._save_file(
                file_path.split("/")[-1].split(".")[-2] + ".txt",
                output_file_path)

        return self.summary

    def summarize_articles(self,
                           articles: Union[str, List[str]],
                           summary_type: str = "map_reduce",
                           summary_len: int = 500,
                           output_file_path: Union[str, None] = None,
                           **kwargs) -> str:
        """input a file path and return a summary of the file

        Args:
            **articles (str)**:  the texts you want to summarize. Can be list of str or str.\n
            **summary_type (str, optional)**: summary method, "map_reduce" or "refine". Defaults to "map_reduce".\n
            **summary_len (int, optional)**: _description_. Defaults to 500.\n
            **output_file_path (str, optional)**: the path of output file. Defaults to "".\n
            **kwargs: the arguments you set in the initial of the class, you can change it here. Include:\n
                chunk_size, chunk_overlap, model, verbose, language , record_exp,
                system_prompt, max_input_tokens, temperature.
        Returns:
            str: the summary of the file
        """

        ## set variables ##
        self.articles = self._handle_texts(articles)
        self.articles_docs = db.change_text_to_doc(self.articles)
        self.summary_type = summary_type.lower()
        self.summary_len = summary_len
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        start_time = time.time()
        table = {}
        if ''.join(self.articles).replace(" ", "") == "":
            print("Error! texts are empty.\n\n")
            return ""

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")

        # Split the documents into sentences

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", " ", ",", ".", "。", "!"],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        docs = text_splitter.split_documents(self.articles_docs)
        self.doc_length = akasha.helper.get_docs_length(self.language, docs)
        texts = [doc.page_content for doc in docs]

        if summary_type == "refine":
            response_list, self.doc_tokens = self._refine_summary(texts)

        else:
            per_sum_chunks = calculate_per_summary_chunks(
                self.language, self.max_input_tokens, self.summary_len,
                self.chunk_size)
            approx_sum_times = calculate_approx_sum_times(
                len(texts), per_sum_chunks)

            response_list_len = len(texts)
            consecutive_merge_fail = 0

            progress = tqdm(total=approx_sum_times, desc="Reduce_map Summary")
            response_list, self.doc_tokens = self._reduce_summary(
                texts, 0, [], progress, response_list_len,
                consecutive_merge_fail)
            progress.close()

        self.summary = response_list[-1]
        p = akasha.prompts.format_refine_summary_prompt(
            "", "", self.summary_len)

        ### write summary to file, and if auto_translate is True , translate it ###
        if self.format_prompt != "":
            text_input = akasha.prompts.format_sys_prompt(
                self.format_prompt, "\n\n" + self.summary,
                self.prompt_format_type, self.model)

            self.summary = akasha.helper.call_model(
                self.model_obj,
                text_input,
            )

        if self.auto_translate:

            self.summary = akasha.helper.call_translator(
                self.model_obj, self.summary, self.prompt_format_type,
                self.language)

        ## change sim to trad if target language is traditional chinese ##
        if afr.language_dict[self.language] == "traditional chinese":
            self.summary = akasha.helper.sim_to_trad(self.summary)
            # self.summary = self.summary.replace("。", "。\n\n")

        print(self.summary, "\n\n\n\n")

        end_time = time.time()
        if self.keep_logs == True:
            self.timestamp_list.append(timestamp)
            self._add_log("summarize_file", timestamp, end_time - start_time,
                          response_list)

        if self.record_exp != "":
            params = akasha.format.handle_params(self.model, "",
                                                 self.chunk_size, "", -1, -1.0,
                                                 self.language, False)
            params["chunk_overlap"] = self.chunk_overlap
            params["summary_type"] = ("refine" if summary_type == "refine" else
                                      "map_reduce")
            metrics = akasha.format.handle_metrics(self.doc_length,
                                                   end_time - start_time,
                                                   self.doc_tokens)
            table = akasha.format.handle_table(p, response_list, self.summary)
            akasha.aiido_upload(self.record_exp, params, metrics, table,
                                output_file_path)

        if output_file_path is not None:
            self._save_file(
                f"summary_{timestamp.replace('/','-').replace(':','-')}",
                output_file_path)

        return self.summary
