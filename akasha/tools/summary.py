from akasha.utils.atman import basic_llm
from akasha.utils.base import (
    DEFAULT_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_INPUT_TOKENS,
)
from akasha.utils.prompts.gen_prompt import (
    format_refine_summary_prompt,
    format_reduce_summary_prompt,
    format_sys_prompt,
)
from akasha.utils.prompts.format import (
    handle_params,
    handle_metrics,
    handle_table,
    language_dict,
)
from akasha.utils.db.load_docs import load_docs_from_info
from akasha.helper.base import get_doc_length, get_docs_length
from akasha.helper.run_llm import call_model
from typing import Union, Tuple
from pathlib import Path
from langchain.schema import Document
import time
import datetime
import math
import logging
from akasha.helper.token_counter import myTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


class summary(basic_llm):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        sum_type: str = "map_reduce",
        sum_len: int = 500,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        consecutive_merge_failures: int = 5,
        temperature: float = 0.0,
        prompt_format_type: str = "auto",
        language: str = "ch",
        record_exp: str = "",
        system_prompt: str = "",
        keep_logs: bool = False,
        verbose: bool = False,
        env_file: str = "",
    ):
        """ "initials of Summary class

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
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n
            **keep_logs (bool, optional)**: record logs or not. Defaults to False.\n
            **prompt_format_type (str, optional)**: the prompt and system prompt format for the language model, including auto, gpt, llama, chat_gpt, chat_mistral, chat_gemini . Defaults to "auto".
            **consecutive_merge_failures (int, optional)**: the number of consecutive merge failures before returning the current response list as the summary. Defaults to 5.
            **max_output_tokens (int, optional)**: max output tokens of llm model. Defaults to 1024.\n
            **max_input_tokens (int, optional)**: max input tokens of llm model. Defaults to 3000.\n
            **env_file (str, optional)**: the path of env file. Defaults to "".\n
        """
        super().__init__(
            model=model,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            language=language,
            record_exp=record_exp,
            system_prompt=system_prompt,
            keep_logs=keep_logs,
            verbose=verbose,
            env_file=env_file,
        )
        self.prompt_format_type = prompt_format_type
        self.consecutive_merge_failures = consecutive_merge_failures
        self.prompt = ""
        self.summary = ""
        self.docs = []
        self.prompt_tokens, self.prompt_length = 0, 0
        self.doc_tokens, self.doc_length = 0, 0
        self.sum_len = sum_len
        self.sum_type = sum_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _display_info(self, batch: int = 1) -> bool:
        if self.verbose is False:
            return False
        print(
            f"Model: {self.model}, Temperature: {self.temperature}, Summary Type: {self.sum_type}"
        )
        print(
            f"Prompt format type: {self.prompt_format_type}, Max input tokens: {self.max_input_tokens}"
        )
        print(
            f"Prompt tokens: {self.prompt_tokens}, Prompt length: {self.prompt_length}"
        )
        print(f"Doc tokens: {self.doc_tokens}, Doc length: {self.doc_length}")

        return True

    def _add_basic_log(
        self, timestamp: str, fn_type: str, history_messages: list = []
    ) -> bool:
        if super()._add_basic_log(timestamp, fn_type) is False:
            return False

        self.logs[timestamp]["sum_type"] = self.sum_type
        self.logs[timestamp]["sum_len"] = self.sum_len
        self.logs[timestamp]["consecutive_merge_failures"] = (
            self.consecutive_merge_failures
        )
        self.logs[timestamp]["chunk_size"] = self.chunk_size
        self.logs[timestamp]["chunk_overlap"] = self.chunk_overlap

        return True

    def _add_result_log(self, timestamp: str, time: float, reponse_list: list) -> bool:
        if super()._add_result_log(timestamp, time) is False:
            return False

        ### add token information ###
        self.logs[timestamp]["summary"] = self.summary
        self.logs[timestamp]["prompt_tokens"] = self.prompt_tokens
        self.logs[timestamp]["prompt_length"] = self.prompt_length
        self.logs[timestamp]["doc_tokens"] = self.doc_tokens
        self.logs[timestamp]["doc_length"] = self.doc_length
        self.logs[timestamp]["summary_list"] = reponse_list

        return True

    def _upload_logs(
        self, tot_time: float, doc_len: int, doc_tokens: int, response_list: list
    ) -> str:
        """_summary_

        Args:
            tot_time (float): _description_
            doc_len (int): _description_
            doc_tokens (int): _description_
        """
        if self.record_exp == "":
            return "no record_exp assigned, so no logs uploaded"

        params = handle_params(
            self.model,
            self.language,
            "",
            chunk_size=self.chunk_size,
        )
        params["chunk_overlap"] = self.chunk_overlap
        params["summary_type"] = "refine" if self.sum_type == "refine" else "map_reduce"
        metrics = handle_metrics(doc_len, tot_time, doc_tokens)
        table = handle_table(self.system_prompt, response_list, self.summary)
        from akasha.utils.upload import aiido_upload

        aiido_upload(self.record_exp, params, metrics, table)

        return "logs uploaded"

    def _reduce_summary(
        self,
        texts: list,
        tokens: int,
        total_list: list,
        progress: tqdm,
        pre_response_list_len: int,
        consecutive_merge_fail: int,
    ):
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
            token, cur_text, newi = _get_text(
                texts, "", i, self.max_input_tokens, self.model
            )
            tokens += token

            progress.update(1)

            ### do the final summary if all chunks can be fits into llm model ###
            if i == 0 and newi == len(texts):
                prompt = format_reduce_summary_prompt(cur_text, self.sum_len)
                input_text = format_sys_prompt(
                    self.system_prompt,
                    "\n" + prompt,
                    self.prompt_format_type,
                    self.model,
                )
                response = call_model(self.model_obj, input_text, self.verbose)

                total_list.append(response)

                return total_list, tokens

            prompt = format_reduce_summary_prompt(cur_text, 0)
            input_text = format_sys_prompt(
                self.system_prompt, "\n" + prompt, self.prompt_format_type, self.model
            )
            response = call_model(self.model_obj, input_text, self.verbose)

            i = newi

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
                print(
                    "Cannot summarize due to texts too long, return current response_list as summary."
                )
                total_list.append("\n\n".join(response_list))
                return total_list, tokens
        else:
            consecutive_merge_fail = 0

        pre_response_list_len = len(response_list)

        return self._reduce_summary(
            response_list,
            tokens,
            total_list,
            progress,
            pre_response_list_len,
            consecutive_merge_fail,
        )

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
        ###

        ### get tqdm progress bar and handle merge fail###
        progress = tqdm(total=len(texts), desc="Refine Summary")
        consecutive_merge_fail = 0
        fnsh_sum_list = []

        while i < len(texts):
            prei = i
            token, cur_text, i = _get_text(
                texts, previous_summary, i, self.max_input_tokens, self.model
            )
            tokens += token
            if previous_summary == "":
                prompt = format_reduce_summary_prompt(cur_text, self.sum_len)
            else:
                prompt = format_refine_summary_prompt(
                    cur_text, previous_summary, self.sum_len
                )
            text_input = format_sys_prompt(
                self.system_prompt, "\n" + prompt, self.prompt_format_type, self.model
            )
            response = call_model(self.model_obj, text_input, self.verbose)

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
            response_list[-1] = "\n\n".join(fnsh_sum_list) + "\n\n" + response_list[-1]

        return response_list, tokens

    def __call__(self, content: Union[str, list, Path, Document], **kwargs) -> str:
        """input one or multiple content/source and return a summary of the content
        content can be a string, a list of strings, a path of file, a list of paths of files, a Document object, or a list of Document objects.
        Args:
            **content (Union[str, list, Path, Document])**:  the content you want to summarize, can be '.txt', '.docx', '.pdf' file.\n
            **sum_type (str, optional)**: summary method, "map_reduce" or "refine". Defaults to "map_reduce".\n
            **sum_len (int, optional)**: expected output length. Defaults to 500.\n
            **kwargs: the arguments you set in the initial of the class, you can change it here. Include:\n
                chunk_size, chunk_overlap, model, verbose, language , record_exp,
                system_prompt, max_input_tokens, temperature.
        Returns:
            str: the summary of the content.
        """

        self._set_model(**kwargs)
        self._change_variables(**kwargs)

        start_time = time.time()
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        self.docs = load_docs_from_info(content)
        self.doc_length = get_docs_length(self.language, self.docs)
        self.doc_tokens = myTokenizer.compute_tokens(
            "".join([d.page_content for d in self.docs]), self.model
        )
        self.prompt_length = get_doc_length(self.language, self.system_prompt)
        self.prompt_tokens = myTokenizer.compute_tokens(self.system_prompt, self.model)

        ### check if docs do not has any content ###
        if self.doc_tokens == 0:
            raise ValueError("Can not get any text in the content")
        ### check if prompt <= max_input_tokens ###
        if self.prompt_tokens > self.max_input_tokens:
            raise ValueError("Prompt tokens exceed max_input_tokens")

        self._add_basic_log(timestamp, "summary")
        self._display_info()  # display the information of the parameters
        self._display_docs()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", " ", ",", ".", "。", "!"],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_docs = text_splitter.split_documents(self.docs)
        split_texts = [doc.page_content for doc in split_docs]

        ### start to ask llm ###
        if self.sum_type == "refine":
            response_list, self.doc_tokens = self._refine_summary(split_texts)

        else:
            per_sum_chunks = _calculate_per_summary_chunks(
                self.language, self.max_input_tokens, self.sum_len, self.chunk_size
            )
            approx_sum_times = _calculate_approx_sum_times(
                len(split_texts), per_sum_chunks
            )

            response_list_len = len(split_texts)
            consecutive_merge_fail = 0

            progress = tqdm(total=approx_sum_times, desc="Reduce_map Summary")
            response_list, self.doc_tokens = self._reduce_summary(
                split_texts, 0, [], progress, response_list_len, consecutive_merge_fail
            )
            progress.close()

        self.summary = response_list[-1]

        end_time = time.time()

        self._add_result_log(timestamp, end_time - start_time, response_list)

        self._upload_logs(
            end_time - start_time, self.doc_length, self.doc_tokens, response_list
        )

        return self.summary


def _calculate_approx_sum_times(chunks: int, per_sum_chunks: int) -> int:
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


def _calculate_per_summary_chunks(
    language: str, max_input_tokens: int, summary_len: int, chunk_size: int
) -> int:
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

    if "chinese" in language_dict[language]:
        token_to_text = 2
    else:
        token_to_text = 1

    ret = max(ret, (token_to_text * max_input_tokens - summary_len) // chunk_size)

    return ret


def _get_text(
    texts: list,
    previous_summary: str,
    i: int,
    max_input_tokens: int,
    model_name: str = "openai:gpt-3.5-turbo",
) -> Tuple[int, str, int]:
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
