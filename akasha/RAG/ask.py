from akasha.utils import dbs

from typing import Callable, Union, List, Tuple, Generator
from akasha.utils import atman, aiido_upload
from akasha.helper import myTokenizer
import datetime, traceback
import time


def get_response(self: atman,
                 doc_path: Union[List[str], str, dbs],
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
                embeddings, chunk_size, model, verbose, topK, language , search_type, record_exp,
                system_prompt, max_doc_len, temperature.

            Returns:
                response (str): the response from llm model.
        """
    self._set_model(**kwargs)
    self._change_variables(**kwargs)
    if isinstance(doc_path, dbs):
        self.doc_path = "use dbs object"
    else:
        self.doc_path = doc_path
    self.prompt = prompt
    search_dict = {}

    return "miaoa"
