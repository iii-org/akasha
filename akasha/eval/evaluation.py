from .model_eval import Model_Eval
from .base import (
    check_sum_type,
    get_non_repeat_rand_int,
    get_source_files,
    check_essay_system_prompt,
    get_question_from_file,
)
from akasha.utils.db.db_structure import dbs
from akasha.utils.base import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBED,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_TYPE,
)
from akasha.utils.prompts.gen_prompt import format_create_question_prompt
from akasha.helper.base import get_docs_length, get_doc_length
from akasha.helper.run_llm import call_model
from akasha.utils.search.search_doc import search_docs
from akasha.utils.search.retrievers.base import get_retrivers
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
import datetime
import time
from typing import List, Union, Tuple, Callable
from pathlib import Path
from langchain.schema import Document
from tqdm import tqdm
import logging


class eval(Model_Eval):
    def __init__(
        self,
        model: Union[str, BaseLanguageModel] = DEFAULT_MODEL,
        embeddings: Union[str, Embeddings] = DEFAULT_EMBED,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        search_type: Union[str, Callable] = DEFAULT_SEARCH_TYPE,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        question_type: str = "fact",
        question_style: str = "essay",
        temperature: float = 0.0,
        threshold: float = 0.0,
        language: str = "ch",
        record_exp: str = "",
        system_prompt: str = "",
        prompt_format_type: str = "auto",
        keep_logs: bool = False,
        use_chroma: bool = False,
        verbose: bool = False,
        env_file: str = "",
    ):
        """initials of Model_Eval class

        Args:
            **embeddings (str, optional)**: the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002".\n
            **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.\n
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
            **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
            **threshold (float, optional)**: (deprecated) the similarity threshold of searching. Defaults to 0.2.\n
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
            **prompt_format_type (str, optional)**: the prompt and system prompt format for the language model, including auto, gpt, llama, chat_gpt, chat_mistral, chat_gemini . Defaults to "auto".\n
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n
            **keep_logs (bool, optional)**: record logs or not. Defaults to False.\n
            **question_style (str, optional)**: the style of question you want to generate, "essay" or "single_choice". Defaults to "essay".\n
            **question_type (str, optional)**: the type of question you want to generate, "fact", "summary", "irrelevant", "compared". Defaults to "fact".\n
            **max_output_tokens (int, optional)**: max output tokens of llm model. Defaults to 1024.\n
            **max_input_tokens (int, optional)**: max input tokens of llm model. Defaults to 3000.\n
        """
        super().__init__(
            model,
            embeddings,
            chunk_size,
            search_type,
            max_input_tokens,
            max_output_tokens,
            question_type,
            question_style,
            temperature,
            threshold,
            language,
            record_exp,
            system_prompt,
            prompt_format_type,
            keep_logs,
            use_chroma,
            verbose,
            env_file,
        )

    def create_questionset(
        self,
        data_source: Union[List[Union[str, Path]], Path, str, dbs],
        question_num: int = 10,
        choice_num: int = 4,
        output_file_path: str = "",
        **kwargs,
    ) -> Tuple[list, list]:
        """auto create question set by llm model, each time it will randomly select a range of documents from the documents directory,
        then use llm model to generate a question and answer pair, and save it into a txt file.
        1.The format of "single_choice" questionset should be one line one question, and the possibles answers and questions are separate by tab(\t),
        the last one is which options is the correct answer, for example, the file should look like: \n
            "What is the capital of Taiwan?" Taipei  Kaohsiung  Taichung  Tainan     1
            何者是台灣的首都?   台北    高雄    台中    台南    1
        2. The format of "essay" questionset should be one line one question, and the reference answer is next line, every questions are separate by
        two newline(\n\n). For example, the file should look like: \n
            問題：根據文件中的訊息，智慧製造的複雜性已超越系統整合商的負荷程度，未來產業鏈中的角色將傾向朝共和共榮共創智慧製造商機，而非過往的單打獨鬥模式發展。請問為什麼供應商、電信商、軟體開發商、平台商、雲端服務供應商、系統整合商等角色會傾向朝共和共榮共創智慧製造商機的方向發展？
            答案：因為智慧製造的複雜性已超越系統整合商的負荷程度，單一角色難以完成整個智慧製造的需求，而共和共榮共創的模式可以整合各方的優勢，共同創造智慧製造的商機。

            問題：根據文件中提到的資訊技術商（IT）和營運技術商（OT），請列舉至少兩個邊緣運算產品或解決方案。
            答案：根據文件中的資訊，NVIDIA的邊緣運算產品包括Jetson系列和EGX系列，而IBM的邊緣運算產品包括IBM Edge Application Manager和IBM Watson Anywhere。

            Args:
                **doc_path (str)**: documents directory path\n
                **question_num (int, optional)**: number of questions you want to create. Defaults to 10.\n
                **choice_num (int, optional)**: the number of choices for each single choice question, only use it if question_type is "single_choice".
            Defaults to 4.\n
                **output_file_path (str, optional)**: the path of output question set txt file, if not assign, use doc_path+datetime as the file name.
                **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
                question_style, question_type, embeddings, chunk_size, model, verbose, language , search_type, record_exp,
                system_prompt, max_input_tokens, temperature.
            Raises:
                Exception: _description_

            Returns:
                (question_list:list, answer_list:list): the question and answer list that generated by llm model
        """

        ## set class variables ##
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.data_source = self._check_doc_path(data_source)
        self.question_num = question_num
        vis_doc_range = set()
        doc_range = (
            (1999 + self.chunk_size) // self.chunk_size
        )  # doc_range is determine by the chunk size, so the select documents won't be too short to having trouble genereating a question

        if check_sum_type(self.question_type, self.question_style):
            return [], []

        self._display_info()
        self._get_db(data_source)
        self._check_db()

        ## process of creating compare question is different from other, so we separate it ##
        if self.question_type in [
            "compare",
            "comparison",
            "comparisons",
            "比較",
            "compared",
        ]:
            return self._create_compare_questionset(choice_num, output_file_path)

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        start_time = time.time()
        self._add_basic_log(timestamp, "create_questionset", doc_range, choice_num)
        self.prompt_tokens, self.prompt_length = [], []
        self.doc_tokens, self.doc_length = [], []
        self.question, self.answer, self.docs = [], [], []
        table = {}

        texts = [pg_content for pg_content in self.db.docs]
        metadata = [metadata for metadata in self.db.metadatas]

        ## prevent doc_range - len of texts <= 0 ##
        doc_range = min(doc_range, len(texts) - 1)

        progress = tqdm(total=question_num, desc=f"Create Q({self.question_type})")
        regenerate_limit = question_num
        ### random select a range of documents from the documents , and use llm model to generate a question and answer pair ###
        for i in range(question_num):
            print(" ")
            progress.update(1)
            print("\n")
            random_index = get_non_repeat_rand_int(
                vis_doc_range, len(texts) - doc_range, doc_range
            )

            doc_text = "\n".join(texts[random_index : random_index + doc_range])
            docs = [
                Document(page_content=texts[k], metadata=metadata[k])
                for k in range(random_index, random_index + doc_range)
            ]
            source_files_name = get_source_files(
                metadata, random_index, doc_range, self.language
            )
            try:
                q_prompt = format_create_question_prompt(
                    doc_text, self.question_type, self.question_style
                )
                response = call_model(
                    self.model_obj,
                    q_prompt,
                    self.verbose,
                )

                if not self._process_response(
                    response, doc_text, choice_num, source_files_name
                ):
                    raise Exception(f"Question Format Error, got {response}")

                self.doc_length.append(get_docs_length(self.language, docs))
                self.doc_tokens.append(self.model_obj.get_num_tokens(doc_text))
                self.prompt_length.append(
                    get_doc_length(self.language, q_prompt) - self.doc_length[-1]
                )
                self.prompt_tokens.append(
                    self.model_obj.get_num_tokens(q_prompt) - self.doc_tokens[-1]
                )
                self.docs.extend(docs)

            except Exception as e:
                print(e)
                if regenerate_limit > 0:
                    regenerate_limit -= 1
                    i -= 1
                    progress.update(-1)
                    logging.warning(f"{e}.\n\n Regenerate\n")
                    continue
                else:
                    logging.error(f"{e}.\n\n Stop\n")
                    raise e

            table = self._update_table(table, docs)

        progress.close()  # end running llm progress bar

        end_time = time.time()
        self._display_info_fnl()
        self._add_result_log(timestamp, end_time - start_time)
        self._upload_logs(
            end_time - start_time, table, sum(self.doc_length), sum(self.doc_tokens)
        )
        self._save_questionset(timestamp, output_file_path)

        return self.question, self.answer

    def create_topic_questionset(
        self,
        data_source: Union[List[str], str],
        topic: str,
        question_num: int = 10,
        choice_num: int = 4,
        output_file_path: str = "",
        **kwargs,
    ) -> Tuple[list, list]:
        """similar to auto_create_questionset, but it will use the topic to find the related documents and create questionset.
        Args:
            **doc_path (str)**: documents directory path\n
            **topic (str)**: the topic of the questionset\n
            **question_num (int, optional)**: number of questions you want to create. Defaults to 10.\n
            **choice_num (int, optional)**: the number of choices for each single choice question, only use it if question_type is "single_choice".
        Defaults to 4.\n
            **output_file_path (str, optional)**: the path of output question set txt file, if not assign, use doc_path+datetime as the file name.
            **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
            question_style, question_type, embeddings, chunk_size, model, verbose, language , search_type, record_exp,
            system_prompt, max_input_tokens, temperature.
        Raises:
            Exception: _description_

        Returns:
            (question_list:list, answer_list:list): the question and answer list that generated by llm model
        """
        ## set class variables ##
        ## set class variables ##
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.data_source = self._check_doc_path(data_source)
        self.question_num = question_num
        vis_doc_range = set()
        doc_range = (
            (1999 + self.chunk_size) // self.chunk_size
        )  # doc_range is determine by the chunk size, so the select documents won't be too short to having trouble genereating a question

        if check_sum_type(self.question_type, self.question_style):
            return [], []

        self._display_info()
        self._get_db(data_source)
        self._check_db()

        ## set local variables ##
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        start_time = time.time()
        self._add_basic_log(timestamp, "create_questionset", doc_range, choice_num)
        self.prompt_tokens, self.prompt_length = [], []
        self.doc_tokens, self.doc_length = [], []
        self.question, self.answer, self.docs = [], [], []
        table = {}

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        start_time = time.time()

        self._add_basic_log(timestamp, "create_topic_questionset")

        ## search related documents ##
        retrivers_list = get_retrivers(
            self.db,
            self.embeddings_obj,
            self.threshold,
            self.search_type
            if self.custom_search_func is None
            else self.custom_search_func,
            self.env_file,
        )
        self.docs, doc_length, doc_tokens = search_docs(
            retrivers_list,
            topic,
            self.model,
            self.max_input_tokens - 100,
            self.search_type,
            self.language,
        )

        texts = [doc.page_content for doc in self.docs]
        metadata = [doc.metadata for doc in self.docs]

        doc_range = min(doc_range, len(texts) - 1)

        progress = tqdm(total=question_num, desc=f"Create Q({self.question_type})")
        regenerate_limit = question_num
        ### random select a range of documents from the documents , and use llm model to generate a question and answer pair ###
        for i in range(question_num):
            print(" ")
            progress.update(1)
            print("\n")
            random_index = get_non_repeat_rand_int(
                vis_doc_range, len(texts) - doc_range, doc_range
            )

            doc_text = "\n".join(texts[random_index : random_index + doc_range])
            docs = [
                Document(page_content=texts[k], metadata=metadata[k])
                for k in range(random_index, random_index + doc_range)
            ]
            source_files_name = get_source_files(
                metadata, random_index, doc_range, self.language
            )

            try:
                q_prompt = format_create_question_prompt(
                    doc_text, self.question_type, self.question_style, topic
                )

                response = call_model(
                    self.model_obj,
                    q_prompt,
                    self.verbose,
                )
                if not self._process_response(
                    response, doc_text, choice_num, source_files_name
                ):
                    raise Exception(f"Question Format Error, got {response}")

                self.doc_length.append(get_docs_length(self.language, docs))
                self.doc_tokens.append(self.model_obj.get_num_tokens(doc_text))
                self.prompt_length.append(
                    get_doc_length(self.language, q_prompt) - self.doc_length[-1]
                )
                self.prompt_tokens.append(
                    self.model_obj.get_num_tokens(q_prompt) - self.doc_tokens[-1]
                )
                self.docs.extend(docs)

            except Exception as e:
                if regenerate_limit > 0:
                    regenerate_limit -= 1
                    i -= 1
                    progress.update(-1)
                    logging.warning(f"{e}.\n\n Regenerate\n")
                    continue
                else:
                    logging.error(f"{e}.\n\n Stop\n")
                    raise e

            table = self._update_table(table, docs)

        progress.close()  # end running llm progress bar

        end_time = time.time()

        ### record logs ###
        self._display_info_fnl()
        self._upload_logs(
            end_time - start_time, table, sum(self.doc_length), sum(self.doc_tokens)
        )

        self._add_result_log(timestamp, end_time - start_time)

        self._save_questionset(timestamp, output_file_path)

        return self.question, self.answer

    def evaluation(
        self,
        questionset_file: str,
        data_source: Union[List[str], str],
        eval_model: Union[BaseLanguageModel, str] = "",
        **kwargs,
    ) -> Union[Tuple[float, float, list], Tuple[float, list]]:
        """parse the question set txt file generated from "create_questionset" function and then use llm model to generate response,
        evaluate the performance of the given paramters based on similarity between responses and the default answers, use
        and rouge_l to evaluate the response if you use essay type to generate questionset.  And use correct_count to evaluate
        the response if you use single_choice type to generate questionset.  **Noted that the question_type must match the questionset_file's type**.


            Args:
            **questionset_flie (str)**: the path of question set txt file, accept .txt, .docx and .pdf.\n
            **question_type (str, optional)**: the type of question you want to generate, "essay" or "single_choice". Defaults to "essay".\n
            **eval_model (Union[BaseLanguageModel, str], optional)**: llm model use to score the response. Defaults to "gpt-3.5-turbo".\n
            **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
                embeddings, chunk_size, model, verbose, language , search_type, record_exp,
                system_prompt, max_input_tokens, temperature.\n
        """
        ## set class variables ##
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.data_source = self._check_doc_path(data_source)
        self._decide_eval_model(eval_model)

        ### get question and answer from questionset file ###
        question, answer, self.question_type, self.question_style = (
            get_question_from_file(
                questionset_file, self.question_type, self.question_style
            )
        )

        if check_sum_type(self.question_type, self.question_style):
            if self.question_style.lower() == "essay":
                return 0.0, 0.0, 0.0, []
            else:
                return 0.0, []

        self.system_prompt = (
            check_essay_system_prompt(
                self.question_style, self.language, self.system_prompt
            )
            + self.system_prompt
        )

        ## db ##
        self._display_info()
        self._get_db(data_source)  # create self.db and self.ignore_files
        self._check_db()

        ## set local variables ##
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        start_time = time.time()
        self.doc_tokens, self.doc_length = [], []
        self.question, self.answer, self.docs = [], [], []
        if self.question_style.lower() == "essay":
            self.score = {"rouge": [], "llm_score": []}
        else:
            self.score = {"correct_count": 0}
        table = {}
        total_docs = []

        self.question_num = len(question)
        progress = tqdm(total=self.question, desc=f"Run Eval({self.question_style})")
        ## add logs ##
        self._add_basic_log(timestamp, "evaluation", -1, -1, questionset_file)

        ### for each question and answer, use llm model to generate response, and evaluate the response by rouge_l and llm###
        retrivers_list = get_retrivers(
            self.db,
            self.embeddings_obj,
            self.threshold,
            self.search_type
            if self.custom_search_func is None
            else self.custom_search_func,
            self.env_file,
        )

        for i in range(self.question_num):
            print(" ")
            progress.update(1)
            print("\n")

            new_table = self._eval_get_res(
                question[i], answer[i], timestamp, retrivers_list
            )
            # ---- #
            if self.record_exp != "":
                for key in new_table:
                    if key not in table:
                        table[key] = []
                    table[key].append(new_table[key])

        progress.close()  # end running llm progress bar
        self.docs = total_docs
        ### record logs ###
        end_time = time.time()
        self.question = question
        self.answer = answer
        self._display_info_fnl()
        self._add_result_log(timestamp, end_time - start_time)

        if self.question_style.lower() == "essay":
            avg_rouge = round(sum(self.score["rouge"]) / len(self.score["rouge"]), 3)
            avg_llm_score = round(
                sum(self.score["llm_score"]) / len(self.score["llm_score"]), 3
            )

            self._upload_logs(
                end_time - start_time,
                table,
                sum(self.doc_length),
                sum(self.doc_tokens),
                True,
            )

            return avg_rouge, avg_llm_score, self.doc_tokens

        else:
            correct_rate = self.score["correct_count"] / self.question_num
            self._upload_logs(
                end_time - start_time,
                table,
                sum(self.doc_length),
                sum(self.doc_tokens),
                True,
            )

            return correct_rate, self.doc_tokens

    def _decide_eval_model(self, eval_model: Union[BaseLanguageModel, str]):
        if isinstance(eval_model, str):
            if eval_model == "":
                pass
            elif eval_model == self.model:
                self.eval_model = self.model_obj
            else:
                self.eval_model = eval_model
        else:
            self.eval_model = eval_model
