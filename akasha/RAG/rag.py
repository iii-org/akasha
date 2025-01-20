from pathlib import Path
import time, datetime
from typing import Callable, Union, List, Tuple, Generator
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from akasha.utils.base import DEFAULT_CHUNK_SIZE, DEFAULT_EMBED, DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MODEL, DEFAULT_SEARCH_TYPE

from akasha.utils import atman
from akasha.utils import dbs
from akasha.helper.token_counter import myTokenizer
from akasha.helper.run_llm import call_model, call_stream_model
from akasha.helper.preprocess_prompts import merge_history_and_prompt
from akasha.utils.prompts.gen_prompt import default_doc_ask_prompt
from akasha.utils.search.retrievers.base import get_retrivers
from akasha.utils.search.search_doc import search_docs


class RAG(atman):
    """class for implement search db based on user prompt and generate response from llm model, include get_response and chain_of_thoughts."""

    def __init__(
        self,
        model: Union[str, BaseLanguageModel] = DEFAULT_MODEL,
        embeddings: Union[str, Embeddings] = DEFAULT_EMBED,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        search_type: Union[str, Callable] = DEFAULT_SEARCH_TYPE,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        threshold: float = 0.0,
        language: str = "ch",
        record_exp: str = "",
        system_prompt: str = "",
        prompt_format_type: str = "auto",
        keep_logs: bool = False,
        use_chroma: bool = False,
        ignore_check: bool = False,
        stream: bool = False,
        verbose: bool = False,
        env_file: str = "",
    ):
        """initials of Doc_QA class

        Args:
            embeddings (_type_, optional): embedding model, including two types(openai and huggingface). Defaults to "openai:text-embedding-ada-002".
            chunk_size (int, optional): the max length of each text segments. Defaults to 1000.
            model (_type_, optional): language model. Defaults to "openai:gpt-3.5-turbo".
            verbose (bool, optional): print the processing text or not. Defaults to False.
            threshold (float, optional): (deprecated) threshold of similarity for searching relavant documents. Defaults to 0.2.
            language (str, optional): "ch" chinese or "en" english. Defaults to "ch".
            search_type (Union[str, Callable], optional): _description_. Defaults to "auto".
            record_exp (str, optional): experiment name of aiido. Defaults to "".
            system_prompt (str, optional): the prompt you want llm to output in certain format. Defaults to "".
            prompt_format_type (str, optional): the prompt and system prompt format for the language model, including auto, gpt, llama, chat_gpt, chat_mistral, chat_gemini . Defaults to "auto".
            temperature (float, optional): temperature for language model. Defaults to 0.0.
            keep_logs (bool, optional): record logs or not. Defaults to False.
            use_chroma (bool, optional): use chroma db name instead of documents path to load data or not. Defaults to False.
            ignore_check (bool, optional): speed up loading data if the chroma db is already existed. Defaults to False.
            max_output_tokens (int, optional): max output tokens of llm model. Defaults to 1024.\n
            max_input_tokens (int, optional): max input tokens of llm model. Defaults to 3000.\n
            env_file (str, optional): the path of the .env file. Defaults to "".\n
        """

        super().__init__(model, embeddings, chunk_size, search_type,
                         max_input_tokens, max_output_tokens, temperature,
                         threshold, language, record_exp, system_prompt,
                         keep_logs, verbose, env_file)
        ### set argruments ###
        self.data_source = ""
        self.use_chroma = use_chroma
        self.ignore_check = ignore_check
        self.prompt_format_type = prompt_format_type
        self.stream = stream
        ### set variables ###

        self.docs = []
        self.doc_tokens = 0
        self.doc_length = 0
        self.response = ""
        self.prompt = ""

    def _add_basic_log(self,
                       timestamp: str,
                       fn_type: str,
                       history_messages: list = []):

        if self.keep_logs == False:
            return

        super()._add_basic_log(timestamp, fn_type)

        self.logs[timestamp]["prompt"] = self.prompt
        self.logs[timestamp]["embeddings"] = self.embeddings
        self.logs[timestamp]["history_messages"] = history_messages

        cur_source = self.data_source
        if not isinstance(cur_source, list):
            cur_source = [cur_source]

        self.logs[timestamp]["data_source"] = []

        for data_path in cur_source:
            if isinstance(data_path, Path):
                self.logs[timestamp]["data_source"].append(data_path.__str__())
            else:
                self.logs[timestamp]["data_source"].append(data_path)

        return

    def _add_result_log(self, timestamp, time):

        if self.keep_logs == False:
            return

        super()._add_result_log(timestamp, time)
        self.logs[timestamp]["response"] = self.response
        return

    def __call__(self,
                 data_source: Union[List[Union[str, Path]], Path, str, dbs],
                 prompt: str,
                 history_messages: list = [],
                 **kwargs):
        """input the documents directory path and question, will first store the documents
        into vectors db (chromadb), then search similar documents based on the prompt question.
        llm model will use these documents to generate the response of the question.

            Args:
                **data_source (Union[List[Union[str, Path]], Path, str, dbs])**: documents directory path\n
                **prompt (str)**:question you want to ask.\n
                **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
                embeddings, chunk_size, model, verbose, topK, language , search_type, record_exp,
                system_prompt, max_doc_len, temperature.

            Returns:
                response (str): the response from llm model.
        """
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.data_source = self._check_doc_path(data_source)
        self._get_db(data_source)  # create self.db and self.ignore_files
        self.prompt = prompt

        start_time = time.time()
        self._check_db()
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        self._add_basic_log(timestamp, "RAG", history_messages)

        ### start to get response ###
        retrivers_list = get_retrivers(
            self.db,
            self.embeddings_obj,
            self.threshold,
            self.search_type,
            self.env_file,
        )
        left_tokens = self.max_input_tokens - \
            myTokenizer.compute_tokens(self.prompt, self.model) - \
            myTokenizer.compute_tokens('\n\n'.join(history_messages),
                                       self.model)
        self.docs, self.doc_length, self.doc_tokens = search_docs(
            retrivers_list,
            self.prompt,
            self.model,
            left_tokens,
            self.search_type,
            self.language,
        )
        if self.docs is None:

            print("\n\nNo Relevant Documents.\n\n")
            self.docs = []

        ## format prompt ##
        if self.system_prompt.replace(' ', '') == "":
            self.system_prompt = default_doc_ask_prompt(self.language)

        text_input = merge_history_and_prompt(history_messages,
                                              self.system_prompt,
                                              self._display_docs() +
                                              "User question: " + self.prompt,
                                              self.prompt_format_type,
                                              model=self.model)

        end_time = time.time()
        if self.stream:
            return call_stream_model(
                self.model_obj,
                text_input,
            )

        self.response = call_model(
            self.model_obj,
            text_input,
        )

        self._add_result_log(timestamp, end_time - start_time)

        self._upload_logs(end_time - start_time, self.doc_length,
                          self.doc_tokens)

        return self.response
