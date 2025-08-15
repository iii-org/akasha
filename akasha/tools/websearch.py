from akasha.utils.base import (
    DEFAULT_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_INPUT_TOKENS,
)
from .ask import ask, _retri_max_texts
from akasha.utils.prompts.gen_prompt import (
    default_ask_prompt,
    default_conclusion_prompt,
    format_sys_prompt,
)
import time
import datetime
from akasha.utils.prompts.format import handle_params, handle_metrics, handle_table
from akasha.helper.base import get_doc_length
from akasha.helper.run_llm import call_model, call_batch_model, check_relevant_answer
from akasha.helper.web_engine import load_docs_from_webengine

from akasha.helper.token_counter import myTokenizer


class websearch(ask):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        prompt_format_type: str = "auto",
        search_engine: str = "wiki",
        search_num: int = 5,
        language: str = "ch",
        record_exp: str = "",
        system_prompt: str = "",
        keep_logs: bool = False,
        verbose: bool = False,
        stream: bool = False,
        env_file: str = "",
    ):
        """websearch class will search the user prompt in the web and based on the results to answer the question.

        Args:
            model (str, optional): _description_. Defaults to DEFAULT_MODEL.
            max_input_tokens (int, optional): _description_. Defaults to DEFAULT_MAX_INPUT_TOKENS.
            max_output_tokens (int, optional): _description_. Defaults to DEFAULT_MAX_OUTPUT_TOKENS.
            temperature (float, optional): _description_. Defaults to 0.0.
            language (str, optional): _description_. Defaults to "ch".
            search_engine (str, optional): the search api methods, includes "serper", "brave","wiki" . Defaults to "wiki".
            search_num (int, optional): the number of search results. Defaults to 5.
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
            prompt_format_type=prompt_format_type,
            language=language,
            record_exp=record_exp,
            system_prompt=system_prompt,
            keep_logs=keep_logs,
            verbose=verbose,
            stream=stream,
            env_file=env_file,
        )
        self.prompt = ""
        self.response = ""
        self.search_engine = search_engine.lower()
        self.search_num = search_num
        self.docs = []
        self.prompt_tokens, self.prompt_length = 0, 0
        self.doc_tokens, self.doc_length = 0, 0

        ## set default RAG prompt ##
        if self.system_prompt.replace(" ", "") == "":
            self.system_prompt = default_ask_prompt(self.language)

    def _display_info(self, batch: int = 1) -> bool:
        if self.verbose is False:
            return False

        print(f"Model: {self.model}, Temperature: {self.temperature}")
        print(f"Search engine: {self.search_engine}, Search num: {self.search_num}")
        print(
            f"Prompt format type: {self.prompt_format_type}, Max input tokens: {self.max_input_tokens}"
        )
        print(
            f"Prompt tokens: {self.prompt_tokens}, Prompt length: {self.prompt_length}"
        )
        print(f"Doc tokens: {self.doc_tokens}, Doc length: {self.doc_length}\n\n")

        return True

    def _add_basic_log(
        self,
        timestamp: str,
        fn_type: str,
    ) -> bool:
        if super()._add_basic_log(timestamp, fn_type) is False:
            return False

        self.logs[timestamp]["prompt"] = self.prompt
        self.logs[timestamp]["search_engine"] = self.search_engine
        self.logs[timestamp]["search_num"] = self.search_num
        self.logs[timestamp].pop("history_messages", None)
        return True

    def _add_result_log(self, timestamp, time) -> bool:
        if super()._add_result_log(timestamp, time) is False:
            return False

        ### add token information ###
        self.logs[timestamp]["response"] = self.response
        self.logs[timestamp]["prompt_tokens"] = self.prompt_tokens
        self.logs[timestamp]["prompt_length"] = self.prompt_length
        self.logs[timestamp]["doc_tokens"] = self.doc_tokens
        self.logs[timestamp]["doc_length"] = self.doc_length
        return True

    def _upload_logs(self, tot_time: float, doc_len: int, doc_tokens: int) -> str:
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
        )
        params["search_engine"] = self.search_engine
        params["search_num"] = self.search_num
        metrics = handle_metrics(doc_len, tot_time, doc_tokens)
        table = handle_table(self.prompt, self.docs, self.response)
        from akasha.utils.upload import aiido_upload

        aiido_upload(self.record_exp, params, metrics, table)

        return "logs uploaded"

    def __call__(self, prompt: str, **kwargs):
        """search the prompt in the web and based on the results to answer the question.

        Args:
            prompt (str): the user input prompt

        Returns:
            str: the response of the prompt
        """
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.prompt = prompt
        self.response = ""

        start_time = time.time()
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        self._add_basic_log(timestamp, "websearch")
        self.docs = load_docs_from_webengine(
            self.prompt,
            self.search_engine,
            self.search_num,
            self.language,
            self.env_file,
        )

        ### check if prompt <= max_input_tokens ###
        tot_prompts = self.prompt + self.system_prompt
        self.prompt_length = get_doc_length(self.language, tot_prompts)
        self.prompt_tokens = myTokenizer.compute_tokens(tot_prompts, self.model)

        if self.prompt_tokens > self.max_input_tokens:
            print("\n\nThe tokens of prompt is larger than max_input_tokens.\n\n")
            raise ValueError("The tokens of prompt is larger than max_input_tokens.")

        ### separate documents and count tokens ###
        cur_documents, self.doc_tokens = self._separate_docs()
        self.doc_length = get_doc_length(self.language, "".join(cur_documents))

        prod_sys_prompts = self._process_batch_prompts(cur_documents)

        self._display_info(
            len(cur_documents)
        )  # display the information of the parameters
        self._display_docs()

        ### start to ask llm ###
        if len(cur_documents) > 1:
            ## call batch model ##
            batch_responses = call_batch_model(
                self.model_obj,
                prod_sys_prompts,
            )
            fnl_conclusion_prompt = default_conclusion_prompt(prompt, self.language)
            ## check relevant answer if batch_responses > 10 ##
            if len(batch_responses) > 10:
                batch_responses = check_relevant_answer(
                    self.model_obj,
                    batch_responses,
                    self.prompt,
                    self.prompt_format_type,
                )

            batch_responses, cur_len = _retri_max_texts(
                batch_responses,
                self.max_input_tokens
                - myTokenizer.compute_tokens(fnl_conclusion_prompt, self.model),
                self.model,
            )

            fnl_input = format_sys_prompt(
                fnl_conclusion_prompt,
                "\n\n".join(batch_responses),
                self.prompt_format_type,
                self.model,
            )

            if self.stream:
                return self._display_stream(
                    fnl_input,
                )

            self.response = call_model(self.model_obj, fnl_input, self.verbose)

        else:
            if self.stream:
                return self._display_stream(prod_sys_prompts[0])

            self.response = call_model(
                self.model_obj, prod_sys_prompts[0], self.verbose
            )

        end_time = time.time()
        self._add_result_log(timestamp, end_time - start_time)

        self._upload_logs(end_time - start_time, self.doc_length, self.doc_tokens)
        return self.response
