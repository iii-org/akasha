import pathlib
import time
from typing import Callable, Union, List, Tuple, Generator
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from akasha.utils.base import DEFAULT_CHUNK_SIZE, DEFAULT_EMBED, DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MODEL, DEFAULT_SEARCH_TYPE

from akasha.utils import atman
from akasha.helper import handle_embeddings, handle_model, handle_search_type, myTokenizer
from akasha.utils import dbs
from akasha.RAG.ask import get_response


class RAG(atman):
    """class for implement search db based on user prompt and generate response from llm model, include get_response and chain_of_thoughts."""

    def __init__(
        self,
        embeddings: Union[str, Embeddings] = DEFAULT_EMBED,
        chunk_size: int = DEFAULT_SEARCH_TYPE,
        model: Union[str, BaseLanguageModel] = DEFAULT_MODEL,
        verbose: bool = False,
        threshold: float = 0.0,
        language: str = "ch",
        search_type: Union[str, Callable] = DEFAULT_SEARCH_TYPE,
        record_exp: str = "",
        system_prompt: str = "",
        prompt_format_type: str = "auto",
        temperature: float = 0.0,
        keep_logs: bool = False,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        compression: bool = False,
        use_chroma: bool = False,
        ignore_check: bool = False,
        stream: bool = False,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
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
            compression (bool, optional): compress the selected documents or not. Defaults to False.
            use_chroma (bool, optional): use chroma db name instead of documents path to load data or not. Defaults to False.
            ignore_check (bool, optional): speed up loading data if the chroma db is already existed. Defaults to False.
            max_output_tokens (int, optional): max output tokens of llm model. Defaults to 1024.\n
            max_input_tokens (int, optional): max input tokens of llm model. Defaults to 3000.\n
            env_file (str, optional): the path of the .env file. Defaults to "".\n
        """

        super().__init__(chunk_size, model, verbose, threshold, language,
                         search_type, record_exp, system_prompt, temperature,
                         keep_logs, max_output_tokens, max_input_tokens,
                         env_file)
        ### set argruments ###
        self.doc_path = ""
        self.compression = compression
        self.use_chroma = use_chroma
        self.ignore_check = ignore_check
        self.prompt_format_type = prompt_format_type
        ### set variables ###
        self.logs = {}
        self.model_obj = handle_model(model, self.verbose, self.temperature,
                                      self.max_output_tokens, self.env_file)
        self.model = handle_search_type(model)

        self.embeddings_obj = handle_embeddings(embeddings, self.verbose,
                                                self.env_file)
        self.embeddings = handle_search_type(embeddings)

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

    def _separate_docs(self,
                       history_messages: list = []) -> Tuple[List[str], int]:
        """separate documents if the total length of documents exceed the max_input_tokens

        Returns:
            ret (List[str]): list of string of separated documents texts
            tot_len (int): the length of total documents
            
        """
        tot_len = 0
        cur_len = 0

        left_tokens = self.max_input_tokens - myTokenizer.compute_tokens(
            self.prompt, self.model) - myTokenizer.compute_tokens(
                '\n\n'.join(history_messages), self.model)
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

                tot_len += cur_token_len
                continue

            cur_len += cur_token_len
            ret[-1] += db_doc.page_content + "\n"
            tot_len += cur_token_len

        ## remove the last empty string ##
        if ret[-1] == "":
            ret = ret[:-1]

        return ret, tot_len

    def ask(self,
            doc_path: Union[List[str], str, dbs],
            prompt: str,
            history_messages: list = [],
            **kwargs):

        return get_response(self, doc_path, prompt, history_messages, **kwargs)
