from akasha.utils.atman import basic_llm
from akasha.utils.base import DEFAULT_MODEL, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_INPUT_TOKENS
from akasha.utils.prompts.gen_prompt import default_ask_prompt, default_conclusion_prompt, format_sys_prompt
from akasha.utils.db.load_docs import load_docs_from_info
from akasha.helper.base import get_doc_length
from akasha.helper.preprocess_prompts import merge_history_and_prompt
from akasha.helper.run_llm import call_model, call_stream_model, call_batch_model, check_relevant_answer
from typing import Callable, Union, List, Tuple, Generator
from pathlib import Path
from langchain.schema import Document
import time, datetime
from akasha.helper.token_counter import myTokenizer


class ask(basic_llm):

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        prompt_format_type: str = "auto",
        language: str = "ch",
        record_exp: str = "",
        system_prompt: str = "",
        keep_logs: bool = False,
        verbose: bool = False,
        stream: bool = False,
        env_file: str = "",
    ):
        """_summary_

        Args:
            model (str, optional): _description_. Defaults to DEFAULT_MODEL.
            max_input_tokens (int, optional): _description_. Defaults to DEFAULT_MAX_INPUT_TOKENS.
            max_output_tokens (int, optional): _description_. Defaults to DEFAULT_MAX_OUTPUT_TOKENS.
            temperature (float, optional): _description_. Defaults to 0.0.
            language (str, optional): _description_. Defaults to "ch".
            record_exp (str, optional): _description_. Defaults to "".
            system_prompt (str, optional): _description_. Defaults to "".
            keep_logs (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to False.
            env_file (str, optional): _description_. Defaults to "".
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
        self.stream = stream
        self.prompt_format_type = prompt_format_type
        self.prompt = ""
        self.response = ""
        self.docs = []
        self.prompt_tokens, self.prompt_length = 0, 0
        self.doc_tokens, self.doc_length = 0, 0

        ## set default RAG prompt ##
        if self.system_prompt.replace(' ', '') == "":
            self.system_prompt = default_ask_prompt(self.language)

    def _display_info(self, batch: int = 1) -> bool:

        if self.verbose == False:
            return False
        print(f"Model: {self.model}, Temperature: {self.temperature}")
        print(
            f"Prompt format type: {self.prompt_format_type}, Max input tokens: {self.max_input_tokens}"
        )
        print(
            f"Prompt tokens: {self.prompt_tokens}, Prompt length: {self.prompt_length}"
        )
        print(f"Doc tokens: {self.doc_tokens}, Doc length: {self.doc_length}")
        print(f"Batch:  {batch}\n\n")

        return True

    def _add_basic_log(self,
                       timestamp: str,
                       fn_type: str,
                       history_messages: list = []) -> bool:

        if super()._add_basic_log(timestamp, fn_type) == False:
            return False

        self.logs[timestamp]["prompt"] = self.prompt
        self.logs[timestamp]["history_messages"] = history_messages
        return True

    def _add_result_log(self, timestamp, time) -> bool:

        if super()._add_result_log(timestamp, time) == False:
            return False

        ### add token information ###
        self.logs[timestamp]["response"] = self.response
        self.logs[timestamp]["prompt_tokens"] = self.prompt_tokens
        self.logs[timestamp]["prompt_length"] = self.prompt_length
        self.logs[timestamp]["doc_tokens"] = self.doc_tokens
        self.logs[timestamp]["doc_length"] = self.doc_length

        return True

    def __call__(self,
                 prompt: str,
                 info: Union[str, list, Path, Document] = "",
                 history_messages: List[str] = [],
                 **kwargs) -> str:
        """_summary_

        Args:
            prompt (str): _description_
            info (Union[str, list], optional): _description_. Defaults to "".
            history_messages (list, optional): _description_. Defaults to [].

        Returns:
            str: _description_
        """

        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.prompt = prompt
        self.response = ""

        start_time = time.time()
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        self.docs = load_docs_from_info(info)

        ### check if prompt <= max_input_tokens ###
        tot_prompts = self.prompt + self.system_prompt + '\n\n'.join(
            history_messages)
        self.prompt_length = get_doc_length(self.language, tot_prompts)
        self.prompt_tokens = myTokenizer.compute_tokens(
            tot_prompts, self.model) + 10

        if self.prompt_tokens > self.max_input_tokens:
            print(
                "\n\nThe tokens of prompt is larger than max_input_tokens.\n\n"
            )
            raise ValueError(
                "The tokens of prompt is larger than max_input_tokens.")

        self._add_basic_log(timestamp,
                            "ask",
                            history_messages=history_messages)

        ### separate documents and count tokens ###
        cur_documents, self.doc_tokens = self._separate_docs()
        self.doc_length = get_doc_length(self.language, ''.join(cur_documents))

        prod_sys_prompts = self._process_batch_prompts(cur_documents,
                                                       history_messages)

        self._display_info(
            len(cur_documents))  # display the information of the parameters
        self._display_docs()
        ### start to ask llm ###
        if len(cur_documents) > 1:
            ## call batch model ##
            batch_responses = call_batch_model(
                self.model_obj,
                prod_sys_prompts,
            )
            fnl_conclusion_prompt = default_conclusion_prompt(
                prompt, self.language)
            ## check relevant answer if batch_responses > 10 ##
            if len(batch_responses) > 10:
                batch_responses = check_relevant_answer(
                    self.model_obj, batch_responses, self.prompt,
                    self.prompt_format_type)

            batch_responses, cur_len = _retri_max_texts(
                batch_responses, self.max_input_tokens -
                myTokenizer.compute_tokens(fnl_conclusion_prompt, self.model),
                self.model)
            fnl_input = format_sys_prompt(fnl_conclusion_prompt,
                                          "\n\n".join(batch_responses),
                                          self.prompt_format_type, self.model)

            if self.stream:
                return call_stream_model(
                    self.model_obj,
                    fnl_input,
                )

            self.response = call_model(self.model_obj, fnl_input)

        else:

            if self.stream:
                return call_stream_model(
                    self.model_obj,
                    prod_sys_prompts[0],
                )

            self.response = call_model(self.model_obj, prod_sys_prompts[0])

        end_time = time.time()
        self._add_result_log(timestamp, end_time - start_time)

        # self._upload_logs(end_time - start_time, self.doc_length,
        #                   self.doc_tokens)
        return self.response

    def _separate_docs(self, ) -> Tuple[List[str], int]:
        """separate documents if the total length of documents exceed the max_input_tokens

        Returns:
            ret (List[str]): list of string of separated documents texts
            tot_len (int): the length of total documents
            
        """
        tot_token_len = 0
        cur_len = 0

        left_tokens = self.max_input_tokens - self.prompt_tokens
        ret = [""]
        for db_doc in self.docs:

            cur_token_len = myTokenizer.compute_tokens(db_doc.page_content,
                                                       self.model)

            if cur_len + cur_token_len > left_tokens:
                if cur_token_len <= left_tokens:
                    cur_len = cur_token_len
                    ret.append(db_doc.page_content)
                else:
                    new_docs = self._truncate_docs(db_doc.page_content)
                    ret.extend(new_docs)
                    ret.append("")
                    cur_len = 0

                tot_token_len += cur_token_len
                continue

            cur_len += cur_token_len
            ret[-1] += db_doc.page_content + "\n"
            tot_token_len += cur_token_len

        ## remove the last empty string ##
        if ret[-1] == "":
            ret = ret[:-1]

        return ret, tot_token_len

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
        truncated_token_len = myTokenizer.compute_tokens(
            truncate_content, self.model)
        while truncated_token_len > self.max_input_tokens:
            idx *= 2
            truncate_content = text[:(tot_len // idx)]
            truncated_token_len = myTokenizer.compute_tokens(
                truncate_content, self.model)

        rge = tot_len // idx
        st = 0
        ed = rge
        while st < tot_len:
            new_docs.append(text[st:ed])
            st += rge
            ed += rge

        return new_docs

    def _process_batch_prompts(self,
                               cur_documents: List[str],
                               history_messages: List[str] = []) -> List[str]:
        """_summary_

        Args:
            prompts (List[str]): _description_

        Returns:
            List[str]: _description_
        """
        prod_sys_prompts = []

        for d_count in range(len(cur_documents)):

            prod_sys_prompt = merge_history_and_prompt(
                history_messages,
                self.system_prompt,
                "Reference document: " + cur_documents[d_count] +
                "\n\nUser question: " + self.prompt,
                self.prompt_format_type,
                model=self.model)
            prod_sys_prompts.append(prod_sys_prompt)

        return prod_sys_prompts


def _retri_max_texts(
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
        txt_len = myTokenizer.compute_tokens(text, model_name)
        if cur_len + txt_len > left_token_len:
            break
        cur_len += txt_len
        ret.append(text)
    return ret, cur_len
