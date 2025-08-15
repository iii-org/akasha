from akasha.utils.prompts.gen_prompt import (
    format_wrong_answer,
    format_sys_prompt,
    format_category_prompt,
    compare_question_prompt,
    default_doc_ask_prompt,
    format_question_query,
    format_llama_json,
)
from akasha.helper import call_model
from akasha.helper.base import extract_json, get_docs_length, get_doc_length
from akasha.helper.scores import get_bert_score, get_llm_score, get_rouge_score
from akasha.utils.prompts.format import (
    handle_score_table,
    handle_metrics,
    handle_params,
    handle_table,
)
from akasha.utils.atman import atman
from akasha.utils.search.search_doc import search_docs
from akasha.utils.base import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBED,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_TYPE,
)
from .base import get_non_repeat_rand_int, find_same_category

import numpy as np
from tqdm import tqdm
import json
import importlib
from collections import defaultdict

from typing import Union, Callable
import os
import datetime
import time
import traceback
import logging
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain.schema import Document


def get_torch():
    ttorch = importlib.import_module("torch")
    return ttorch


def _generate_single_choice_question(
    doc_text: str,
    question: str,
    cor_ans: str,
    model,
    system_prompt: str,
    choice_num: int,
) -> list:
    """Based on gernerated question and answer, generate wrong answers for single choice question

    Args:
        **doc_text (str)**: the document text that used to generate question and answer\n
        **question (str)**: question generated from previous step\n
        **cor_ans (str)**: correct answer genereated from previous step\n
        **model (var)**: llm model\n
        **system_prompt (str)**: the system prompt that you assign special instruction to llm model, currently not be used in the function\n
        **choice_num (int)**: the number of options for each single choice question\n

    Raises:
        Exception: if the format of the response is not correct, raise exception

    Returns:
        list: choice_num of wrong answers and a correct answer, and the index of correct answer
    """
    res = []
    count = 0
    random_index = np.random.randint(choice_num)
    q_prompt = format_wrong_answer(choice_num - 1, doc_text, question, cor_ans)

    input_text = format_sys_prompt(system_prompt, q_prompt)
    response = call_model(model, input_text, False)

    ### separate the response into wrong answers ###
    try:
        process = response.split("錯誤答案：")
        process = process[1:]
        if len(process) != choice_num - 1:
            raise Exception("Answer Format Error")
    except Exception:
        try:
            process = response.split("：")[1:]
            if len(process) != choice_num - 1:
                raise Exception("Answer Format Error")
        except Exception:
            process = response.split(":")[1:]
    ### combine the wrong answers and correct answer into a single choice question ###
    for wrong_ans in process:
        if wrong_ans == "":
            continue
        elif count == random_index:
            res.append(str(count + 1) + "." + cor_ans.replace("\n", ""))
            count += 1

        wrong_ans = (
            str(count + 1) + "." + wrong_ans.replace("\n", "").replace("錯誤答案", "")
        )
        res.append(wrong_ans)
        count += 1

    if count < choice_num:
        res.append(str(count + 1) + "." + cor_ans.replace("\n", ""))

    res.append(str(random_index + 1))

    return res


class Model_Eval(atman):
    """class for implement evaluation of llm model, include auto_create_questionset and auto_evaluation."""

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
        super().__init__(
            model,
            embeddings,
            chunk_size,
            search_type,
            max_input_tokens,
            max_output_tokens,
            temperature,
            threshold,
            language,
            record_exp,
            system_prompt,
            keep_logs,
            verbose,
            use_chroma,
            env_file,
        )

        ### set argruments ###
        self.data_source = ""
        self.prompt_format_type = prompt_format_type
        self.question_type = question_type
        self.question_style = question_style
        self.question_num = 10
        self.eval_model = self.model_obj
        ### set variables ###

        self.docs = []
        self.prompt_tokens, self.prompt_length = [], []
        self.doc_tokens, self.doc_length = [], []
        self.question = []
        self.answer = []
        self.response = []
        self.score = {}

    def _add_basic_log(
        self,
        timestamp: str,
        fn_type: str,
        doc_range: int = -1,
        choice_num: int = -1,
        questionset_file: str = "",
    ) -> bool:
        if super()._add_basic_log(timestamp, fn_type) is False:
            return False

        self.logs[timestamp]["question_num"] = self.question_num
        self.logs[timestamp]["question_type"] = self.question_type
        self.logs[timestamp]["question_style"] = self.question_style

        if doc_range != -1 and choice_num != -1:
            self.logs[timestamp]["doc_range"] = doc_range
            self.logs[timestamp]["choice_num"] = choice_num
        else:
            self.logs[timestamp]["search_type"] = self.search_type
            self.logs[timestamp]["questionset_path"] = questionset_file
        return True

    def _add_result_log(self, timestamp: str, time: float) -> bool:
        """add post-process log to self.logs

        Args:
            timestamp (str): timestamp of this run
            time (float): spent time of this run
        """

        if super()._add_result_log(timestamp, time) is False:
            return False

        self.logs[timestamp]["prompt_length"] = self.prompt_length
        self.logs[timestamp]["prompt_tokens"] = self.prompt_tokens
        self.logs[timestamp]["doc_length"] = self.doc_length
        self.logs[timestamp]["doc_tokens"] = self.doc_tokens
        self.logs[timestamp]["question"] = self.question
        self.logs[timestamp]["answer"] = self.answer

        if "bert" in self.score:
            self.logs[timestamp]["bert"] = self.score["bert"]
            self.logs[timestamp]["rouge"] = self.score["rouge"]
            self.logs[timestamp]["llm_score"] = self.score["llm_score"]
        elif "correct_count" in self.score:
            self.logs[timestamp]["correct_count"] = self.score["correct_count"]

        return True

    def _display_info(self) -> bool:
        if self.verbose is False:
            return False
        print(f"Model: {self.model}, Embeddings: {self.embeddings}")
        print(f"Chunk size: {self.chunk_size}, Search type: {self.search_type}")

        return True

    def _display_info_fnl(self) -> bool:
        if self.verbose is False:
            return False
        print(
            f"Prompt tokens: {self.prompt_tokens}, Prompt length: {self.prompt_length}"
        )
        print(f"Doc tokens: {self.doc_tokens}, Doc length: {self.doc_length}\n\n")

        return True

    def _save_questionset(self, timestamp: str, output_file_path: str):
        """save questions and ref answers into txt file, and save the path of question set into logs

        Args:
            timestamp (str): the timestamp of the question set created function
            output_file_path (str): file name of the question set txt file, if not assign, use doc_path+datetime as the file name.
        """
        ### write question and answer into txt file, but first check if "questionset" directory exist or not, it not, first create it.
        ### for filename, if not assign, use doc_path+datetime as the file name.
        if not os.path.exists("questionset"):
            os.makedirs("questionset")

        if output_file_path == "":
            now = datetime.datetime.now()
            date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
            output_file_path = "questionset/" + str(date_time_string) + ".json"
        elif not output_file_path.endswith(".json"):
            output_file_path = output_file_path + ".json"

        data = {
            "question_style": self.question_style,
            "question_type": self.question_type,
            "question": self.question,
            "answer": self.answer,
        }

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print("\nquestion set saved in ", output_file_path, "\n\n")
        if self.keep_logs is True:
            self.logs[timestamp]["question"] = self.question
            self.logs[timestamp]["answer"] = self.answer
            self.logs[timestamp]["questionset_path"] = output_file_path

        return

    def _process_fact(
        self, response: str, doc_text: str, choice_num: int, source_file_name: str = ""
    ) -> bool:
        """parse the question and answer from the llm response, and save it into question and answer list.
        if can not parse the response, return False

        Args:
            response (str): llm generated response
            doc_text (str): the document text that used to generate question and answer
            choice_num (int): the number of options for each single choice question

        Returns:
            bool: if can not parse the response, return False
        """
        try:
            process = "".join(response.split("問題：")).split("答案：")
            if len(process) < 2:
                raise SyntaxError("Question Format Error")
        except Exception:
            process = "".join(response.split("問題:")).split("答案:")
            if len(process) < 2:
                return False

        process[0] = process[0].replace("根據文件", "")
        self.question.append(source_file_name + process[0])
        if self.question_style == "essay":
            self.answer.append(process[1])

        else:
            anss = _generate_single_choice_question(
                doc_text,
                process[0],
                process[1],
                self.model_obj,
                self.system_prompt,
                choice_num,
            )
            self.answer.append(anss)

        return True

    def _process_summary(self, response: str, doc_text: str) -> bool:
        """parse the question and answer from the llm response, and save it into question and answer list.
        if can not parse the response, return False

        Args:
            response (str): llm generated response
            doc_text (str): the document text that used to generate question and answer

        Returns:
            bool: if can not parse the response, return False
        """
        try:
            process = response.split("答案：")
            if len(process) < 2:
                raise SyntaxError("Question Format Error")
        except Exception:
            process = response.split("答案:")
            if len(process) < 2:
                return False

        self.question.append(doc_text.replace("\n", "") + "\n")
        self.answer.append(process[-1].replace("\n", ""))

        return True

    def _process_response(
        self, response: str, doc_text: str, choice_num: int, source_file_name: str = ""
    ):
        """process the response from llm model, and generate question and answer pair
        based on the question type and question style
        """
        if self.question_type.lower() in [
            "summary",
            "sum",
            "summarization",
            "summarize",
            "summaries",
            "摘要",
        ]:
            return self._process_summary(response, doc_text)

        return self._process_fact(response, doc_text, choice_num, source_file_name)

    def _create_compare_questionset(self, choice_num: int, output_file_path: str):
        """create compare question set, first randomly select documents and label the category of some proper nouns in the documents,
        can use the documents and proper nouns that have same category to generate compare question and answer pair.

        Args:
            choice_num (int): the number of options for each single choice question
            output_file_path (str): file name of the question set txt file, if not assign, use doc_path+datetime as the file name.

        Returns:
            _type_: _description_
        """
        ## set local variables ##
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        start_time = time.time()
        doc_range = 1
        cate_threshold = 3
        vis_doc_range = set()
        self.doc_tokens, self.doc_length = [], []
        self.question, self.answer, self.docs = [], [], []
        table = {}

        ## add logs ##

        self._add_basic_log(timestamp, "create_questionset", doc_range, choice_num)

        texts = [pg_content for pg_content in self.db.docs]
        metadata = [metadata for metadata in self.db.metadatas]

        progress = tqdm(total=self.question_num, desc=f"Create Q({self.question_type})")
        print("\n")
        regenerate_limit = self.question_num
        category = defaultdict(list)
        set_category = defaultdict(set)
        for i in range(self.question_num):
            docs = []
            print(" ")
            progress.update(1)
            print("\n")

            ### start to random choose text, and use text to add different category things into dictionary,###
            ### if the value of any category large than category threshold, then we can use the category to create compare question ###
            compare_resource = find_same_category(category, cate_threshold)
            while not isinstance(compare_resource, list):
                random_index = get_non_repeat_rand_int(
                    vis_doc_range, len(texts) - doc_range, doc_range
                )
                doc_text = texts[random_index]
                docs.append(
                    Document(
                        page_content=texts[random_index],
                        metadata=metadata[random_index],
                    )
                )
                count = 3
                try:
                    ## ask model to get category & nouns, add to category ##
                    category_prompt = format_category_prompt(doc_text, self.language)
                    response = call_model(self.model_obj, category_prompt, False)

                    json_response = extract_json(response)
                    if json_response is None:
                        raise Exception("Response Format Error")

                    for k, v in json_response.items():
                        if k not in set_category[v]:
                            category[v].append([k, doc_text])
                            set_category[v].add(k)
                    compare_resource = find_same_category(category, cate_threshold)

                except Exception as e:
                    print("error during generate categories\n", e)
                    count -= 1
                    if count == 0:
                        break
            topic, nouns, used_texts = compare_resource
            used_texts = "\n".join(set(used_texts))
            nouns = ", ".join(nouns)
            try:
                q_prompt = compare_question_prompt(
                    self.question_style, topic, nouns, used_texts
                )
                response = call_model(self.model_obj, q_prompt, False)

                if not self._process_response(response, used_texts, choice_num, ""):
                    raise Exception(f"Question Format Error, got {response}")

                self.doc_length.append(get_docs_length(self.language, docs))
                self.doc_tokens.append(self.model_obj.get_num_tokens(used_texts))
                self.prompt_length.append(get_doc_length(self.language, q_prompt))
                self.prompt_tokens.append(self.model_obj.get_num_tokens(q_prompt))
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

            # remove the category from category dictionary
            del category[topic]
            # remove the category from set_category dictionary
            del set_category[topic]

            # aiido upload
            table = self._update_table(table, docs)

        progress.close()  # end running llm progress bar

        end_time = time.time()

        ### record logs ###
        self._upload_logs(
            end_time - start_time, table, sum(self.doc_length), sum(self.doc_tokens)
        )
        self._add_result_log(timestamp, end_time - start_time)

        self._save_questionset(timestamp, output_file_path)

        return self.question, self.answer

    def _eval_get_res_fact(
        self,
        question: Union[str, list],
        answer: str,
        timestamp: str,
        retrivers_list: list,
    ) -> dict:
        """generate fact resposne from the question, can evaluate with reference answer

        Args:
            question (Union[str, list]): if it's single_choice, it should be a list(include options), else it should be a string of question
            answer (str): the reference answer of the question
            timestamp (str): the timestamp of the auto evaluation function

        Returns:
            dict: evaluation result
        """

        ### format question ###
        if self.question_style.lower() == "essay":
            query = question
            prod_sys = self.system_prompt + default_doc_ask_prompt(self.language)
            query_with_prompt = question

        else:
            prod_sys = self.system_prompt
            query, ans = format_question_query(question, answer)
            query_with_prompt = format_llama_json(query)

        ### get docs ###

        query_len = get_doc_length(self.language, query_with_prompt)
        query_tokens = self.model_obj.get_num_tokens(query_with_prompt)

        self.docs, doc_length, doc_tokens = search_docs(
            retrivers_list,
            query,
            self.model,
            self.max_input_tokens - query_tokens,
            self.search_type,
            self.language,
        )
        if self.doc_tokens == 0:
            print(
                "Warning: Unable to retrieve any documents, possibly due to insufficient remaining tokens.\n\n"
            )
            self.docs = []

        ### ask llm ###

        try:
            self._display_docs()
            intput_text = format_sys_prompt(
                prod_sys + self._format_docs(),
                query_with_prompt,
                self.prompt_format_type,
                self.model,
            )
            response = call_model(self.model_obj, intput_text, False)
            self.response.append(response)
            self.doc_length.append(doc_length)
            self.doc_tokens.append(doc_tokens)
            self.prompt_length.append(query_len)
            self.prompt_tokens.append(query_tokens)
        except Exception as e:
            traceback.print_exc()
            # response = ["running model error"]
            torch = get_torch()
            torch.cuda.empty_cache()
            logging.error(f"running model error\n {e}")
            raise e

        if self.question_style.lower() == "essay":
            if self.verbose:
                print("Question: ", question, "\n\n")
                print("Reference Answer: ", answer, "\n\n")
                print("Generated Response: ", response, "\n\n")

            self.score["bert"].append(get_bert_score(response, answer, self.language))
            self.score["rouge"].append(get_rouge_score(response, answer, self.language))
            self.score["llm_score"].append(
                get_llm_score(response, answer, self.eval_model, "auto")
            )

            new_table = handle_table(
                question + "\nAnswer:  " + answer, self.docs, response
            )
            new_table = handle_score_table(
                new_table,
                self.score["bert"][-1],
                self.score["rouge"][-1],
                self.score["llm_score"][-1],
            )

        else:
            if self.verbose:
                print("Question: ", question, "\n\n")
                print("Reference Answer: ", ans, "\n\n")
                print("Generated Response: ", response, "\n\n")

            new_table = handle_table(query + "\nAnswer:  " + ans, self.docs, response)
            new_table = {}
            result = extract_result(response)

            if str(ans).replace(" ", "") in str(result):
                self.score["correct_count"] += 1

        return new_table

    def _eval_get_res_summary(self, sum_doc: str, answer: str, timestamp: str) -> dict:
        """generate summary resposne from the question, can evaluate with reference answer

        Args:
            sum_doc (str): text that need to be summarized
            answer (str): the reference answer of the summary
            timestamp (str): the timestamp of the auto evaluation function

        Returns:
            dict: evaluation result
        """

        prompt = "請對以下文件進行摘要: "
        intput_text = format_sys_prompt(
            self.system_prompt,
            prompt + "\n\n" + sum_doc,
            self.prompt_format_type,
            self.model,
        )

        self.docs = [Document(page_content=sum_doc, metadata={"source": "", "page": 0})]

        try:
            response = call_model(self.model_obj, intput_text, False)
            self.response.append(response)
            self.doc_length.append(get_doc_length(self.language, sum_doc))
            self.doc_tokens.append(self.model_obj.get_num_tokens(sum_doc))
            self.prompt_length.append(8)
            self.prompt_tokens.append(15)

        except Exception as e:
            traceback.print_exc()
            # response = ["running model error"]
            torch = get_torch()
            torch.cuda.empty_cache()
            logging.error(f"running model error\n {e}")
            raise e

        if self.verbose:
            print("Question: ", prompt + "\n" + sum_doc, "\n\n")
            print("Reference Answer: ", answer, "\n\n")
            print("Generated Response: ", response, "\n\n")

        self.score["bert"].append(get_bert_score(response, answer, self.language))
        self.score["rouge"].append(get_rouge_score(response, answer, self.language))
        self.score["llm_score"].append(
            get_llm_score(response, answer, self.eval_model, "auto")
        )

        # new_table = akasha.format.handle_table(prompt + "\nAnswer:  " + answer,
        #                                        self.docs, response)
        # new_table = akasha.format.handle_score_table(
        #     new_table,
        #     self.score["bert"][-1],
        #     self.score["rouge"][-1],
        #     self.score["llm_score"][-1],
        # )
        new_table = {}

        return new_table

    def _eval_get_res(
        self,
        question: Union[list, str],
        answer: str,
        timestamp: str,
        retrivers_list: list,
    ) -> dict:
        """separate the question type and call different function to generate response

        Args:
            question (str): the question that need to be evaluated
            answer (str): the reference answer of the question
            timestamp (str): the timestamp of the auto evaluation function

        Returns:
            dict: _description_
        """
        if (
            self.question_type.lower()
            in ["fact", "facts", "factoid", "factoids", "事實"]
            or self.question_type.lower()
            in ["irre", "irrelevant", "irrelevance", "無關"]
            or self.question_type.lower()
            in ["compared", "compare", "comparison", "comparisons", "比較"]
        ):
            return self._eval_get_res_fact(question, answer, timestamp, retrivers_list)

        elif self.question_type.lower() in [
            "summary",
            "sum",
            "summarization",
            "summarize",
            "summaries",
            "摘要",
        ]:
            return self._eval_get_res_summary(question, answer, timestamp)

    def _update_table(self, table: dict, docs: list) -> dict:
        """add data into table dictionary for aiido

        Args:
            table (dict): _description_
            docs (list): _description_

        Returns:
            dict: _description_
        """
        if self.record_exp == "":
            return {}

        new_table = handle_table(self.question[-1], docs, self.answer[-1])
        for key in new_table:
            if key not in table:
                table[key] = []
            table[key].append(new_table[key])
        return table

    def _upload_logs(
        self,
        tot_time: float,
        table: dict,
        doc_length: int = 0,
        doc_tokens: int = 0,
        from_eval: bool = False,
    ) -> str:
        if self.record_exp == "":
            return "no record_exp assigned, so no logs uploaded"

        params = handle_params(
            self.model,
            self.language,
            self.search_type,
            self.threshold,
            self.embeddings,
            self.chunk_size,
        )
        metrics = handle_metrics(doc_length, tot_time, doc_tokens)
        if from_eval:
            if self.question_style.lower() == "essay":
                avg_bert = round(sum(self.score["bert"]) / len(self.score["bert"]), 3)
                avg_rouge = round(
                    sum(self.score["rouge"]) / len(self.score["rouge"]), 3
                )
                avg_llm_score = round(
                    sum(self.score["llm_score"]) / len(self.score["llm_score"]), 3
                )

                metrics["avg_bert"] = avg_bert
                metrics["avg_rouge"] = avg_rouge
                metrics["avg_llm_score"] = avg_llm_score
            else:
                correct_rate = self.score["correct_count"] / self.question_num
                metrics["correct_rate"] = correct_rate

        from akasha.utils.upload import aiido_upload

        aiido_upload(self.record_exp, params, metrics, table)

        return "logs uploaded"


def extract_result(response: str):
    """to prevent the output of llm format is not what we want, try to extract the answer (digit) from the llm output

    Args:
        **response (str)**: llm output\n

    Returns:
        int: digit of answer
    """
    try:
        res = extract_json(response)
        # res = str(json.loads(response)["ans"]).replace(" ", "")
        if res is None:
            raise Exception("can not find the json format in the response")
        res = res["ans"]
    except Exception:
        res = -1
        for c in response:
            if c.isdigit():
                res = c

                break
    return res
