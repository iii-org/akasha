import datetime
import time
from tqdm import tqdm
import akasha
import akasha.eval as eval
import akasha.db
import os, traceback, logging
import numpy as np
import torch, gc
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from typing import Callable, Union, Tuple, List
from collections import defaultdict


def _generate_single_choice_question(
    doc_text: str,
    question: str,
    cor_ans: str,
    model,
    system_prompt: str,
    choice_num: int,
) -> str:
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
        str: choice_num of wrong answers and a correct answer, and the index of correct answer, separated by "\t"
    """
    res = ""
    count = 0
    random_index = np.random.randint(choice_num)
    q_prompt = akasha.prompts.format_wrong_answer(choice_num - 1, doc_text,
                                                  question, cor_ans)

    input_text = akasha.prompts.format_sys_prompt(system_prompt, q_prompt)
    response = akasha.helper.call_model(model, input_text)

    ### separate the response into wrong answers ###
    try:
        process = response.split("錯誤答案：")
        process = process[1:]
        if len(process) != choice_num - 1:
            raise Exception("Answer Format Error")
    except:
        try:
            process = response.split("：")[1:]
            if len(process) != choice_num - 1:
                raise Exception("Answer Format Error")
        except:
            process = response.split(":")[1:]
    ### combine the wrong answers and correct answer into a single choice question ###
    for wrong_ans in process:
        if wrong_ans == "":
            continue
        elif count == random_index:
            res += "\t" + str(count + 1) + '.' + cor_ans.replace("\n", "")
            count += 1

        wrong_ans = str(count + 1) + '.' + wrong_ans.replace("\n", "").replace(
            "錯誤答案", "")
        res += "\t" + wrong_ans
        count += 1

    if count < choice_num:
        res += "\t" + str(count + 1) + '.' + cor_ans.replace("\n", "")

    res += "\t" + str(random_index + 1)

    return res


class Model_Eval(akasha.atman):
    """class for implement evaluation of llm model, include auto_create_questionset and auto_evaluation."""

    def __init__(
        self,
        embeddings: str = "openai:text-embedding-ada-002",
        chunk_size: int = 1000,
        model: str = "openai:gpt-3.5-turbo",
        verbose: bool = False,
        topK: int = -1,
        threshold: float = 0.0,
        language: str = "ch",
        search_type: Union[str, Callable] = "svm",
        record_exp: str = "",
        system_prompt: str = "",
        prompt_format_type: str = "gpt",
        max_doc_len: int = 1500,
        temperature: float = 0.0,
        keep_logs: bool = False,
        max_output_tokens: int = 1024,
        question_type: str = "fact",
        question_style: str = "essay",
        use_chroma: bool = False,
        use_rerank: bool = False,
        ignore_check: bool = False,
        max_input_tokens: int = 3000,
        env_file: str = "",
    ):
        """initials of Model_Eval class

        Args:
            **embeddings (str, optional)**: the embeddings used in query and vector storage. Defaults to "text-embedding-ada-002".\n
            **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.\n
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
            **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
            **topK (int, optional)**: search top k number of similar documents. Defaults to 2.\n
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
            **max_doc_len (int, optional)**: max document size of llm input. Defaults to 1500.\n (deprecated in 1.0.0)
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n
            **keep_logs (bool, optional)**: record logs or not. Defaults to False.\n
            **question_style (str, optional)**: the style of question you want to generate, "essay" or "single_choice". Defaults to "essay".\n
            **question_type (str, optional)**: the type of question you want to generate, "fact", "summary", "irrelevant", "compared". Defaults to "fact".\n
            **use_rerank (bool, optional)**: use rerank model to re-rank the selected documents or not. Defaults to False.
            **max_output_tokens (int, optional)**: max output tokens of llm model. Defaults to 1024.\n
            **max_input_tokens (int, optional)**: max input tokens of llm model. Defaults to 3000.\n
        """

        super().__init__(chunk_size, model, verbose, topK, threshold, language,
                         search_type, record_exp, system_prompt, max_doc_len,
                         temperature, keep_logs, max_output_tokens,
                         max_input_tokens, env_file)
        ### set argruments ###
        self.doc_path = ""
        self.question_type = question_type
        self.question_style = question_style
        self.question_num = 0
        self.prompt_format_type = prompt_format_type
        ### set variables ###
        self.logs = {}
        self.model_obj = akasha.helper.handle_model(model, self.verbose,
                                                    self.temperature,
                                                    self.max_output_tokens,
                                                    self.env_file)
        self.embeddings_obj = akasha.helper.handle_embeddings(
            embeddings, self.verbose, self.env_file)
        self.embeddings = akasha.helper.handle_search_type(embeddings)
        self.model = akasha.helper.handle_search_type(model)
        self.search_type = search_type
        self.db = None
        self.docs = []
        self.doc_tokens = 0
        self.doc_length = 0
        self.question = []
        self.answer = []
        self.response = []
        self.score = {}
        self.ignored_files = []
        self.use_chroma = use_chroma
        self.ignore_check = ignore_check
        self.use_rerank = use_rerank

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
        self.logs[timestamp]["system_prompt"] = self.system_prompt

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
        if isinstance(self.doc_path, list):
            suf_path = self.doc_path[0].split("/")[-2]
        else:
            suf_path = self.doc_path.split("/")[-2]
        if output_file_path == "":
            now = datetime.datetime.now()
            date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
            output_file_path = ("questionset/" + suf_path + "_" +
                                str(date_time_string) + ".txt")
        elif output_file_path[-4:] != ".txt":
            output_file_path = output_file_path + ".txt"
        with open(output_file_path, "w", encoding="utf-8") as f:
            for w in range(len(self.question)):
                if self.question_style == "essay":
                    f.write(self.question[w].replace("\n", "") + "\n" +
                            self.answer[w].replace("\n", "") + "\n\n")
                else:
                    if w == len(self.question) - 1:
                        f.write(self.question[w].replace("\n", "") +
                                self.answer[w].replace("\n", ""))
                    else:
                        f.write(self.question[w].replace("\n", "") +
                                self.answer[w].replace("\n", "") + "\n")

        print("question set saved in ", output_file_path, "\n\n")
        if self.keep_logs == True:
            self.logs[timestamp]["question"] = self.question
            self.logs[timestamp]["answer"] = self.answer
            self.logs[timestamp]["questionset_path"] = output_file_path

        return

    def _process_fact(self,
                      response: str,
                      doc_text: str,
                      choice_num: int,
                      source_file_name: str = "") -> bool:
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
        except:
            process = "".join(response.split("問題:")).split("答案:")
            if len(process) < 2:
                return False

        process[0] = process[0].replace("根據文件", "")
        self.question.append("問題： " + source_file_name + process[0])
        if self.question_style == "essay":
            self.answer.append("答案： " + process[1])

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
            response = process[0] + "\n" + "選項:\n" + anss + "\n\n"

        return True

    def _process_summary(self, response, doc_text: str) -> bool:
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
        except:
            process = response.split("答案:")
            if len(process) < 2:
                return False

        self.question.append("問題：  " + doc_text.replace("\n", "") + "\n")
        self.answer.append("答案： " + process[-1].replace("\n", ""))

        return True

    def _process_irrelevant(self, response, doc_text: str,
                            choice_num: int) -> bool:
        """current not used, since llm may generate questions that can be answered.

        Args:
            response (_type_): _description_
            doc_text (str): _description_
            choice_num (int): _description_

        Returns:
            bool: _description_
        """
        try:
            process = response.split("問題：")
            if len(process) < 2:
                raise Exception("Question Format Error")
        except:
            process = response.split("問題:")
            if len(process) < 2:
                return False

        default_ans = "根據文件中的訊息，無法回答此問題。"

        self.question.append("問題： " + process[-1])
        if self.question_style == "essay":
            self.answer.append("答案： " + default_ans)
            response += "\n答案： " + default_ans
        else:
            anss = _generate_single_choice_question(
                doc_text,
                process[-1],
                default_ans,
                self.model_obj,
                self.system_prompt,
                choice_num,
            )
            self.answer.append(anss)
            response = process[0] + "\n" + "選項:\n" + anss + "\n\n"

        if self.verbose:
            print(response)

        return True

    def _process_response(self,
                          response: str,
                          doc_text: str,
                          choice_num: int,
                          source_file_name: str = ""):
        """process the response from llm model, and generate question and answer pair
            based on the question type and question style
        """
        if self.question_type.lower() in ["fact", "facts", "factoid", "factoids", "事實"] or \
            self.question_type.lower() in ["irre", "irrelevant", "irrelevance", "無關"] or \
                self.question_type.lower() in ["compared", "compare", "comparison", "comparisons", "比較"]:
            return self._process_fact(response, doc_text, choice_num,
                                      source_file_name)

        elif self.question_type.lower() in [
                "summary", "sum", "summarization", "summarize", "summaries",
                "摘要"
        ]:
            return self._process_summary(response, doc_text)

        return self._process_fact(response, doc_text, choice_num,
                                  source_file_name)

    def _create_compare_questionset(self, choice_num: int,
                                    output_file_path: str):
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
        self.doc_tokens, self.doc_length = 0, 0
        self.question, self.answer, self.docs = [], [], []
        table = {}

        ## add logs ##
        if self.keep_logs == True:
            self.timestamp_list.append(timestamp)
            self._add_basic_log(timestamp, "auto_create_questionset")
            self.logs[timestamp]["doc_range"] = doc_range
            self.logs[timestamp]["question_num"] = self.question_num
            self.logs[timestamp]["question_type"] = self.question_type
            self.logs[timestamp]["question_style"] = self.question_style
            self.logs[timestamp]["choice_num"] = choice_num

        texts = [doc.page_content for doc in self.db]
        metadata = [doc.metadata for doc in self.db]

        progress = tqdm(total=self.question_num,
                        desc=f"Create Q({self.question_type})")
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
                random_index = akasha.helper.get_non_repeat_rand_int(
                    vis_doc_range,
                    len(texts) - doc_range, doc_range)
                doc_text = texts[random_index]
                docs.append(
                    Document(page_content=texts[random_index],
                             metadata=metadata[random_index]))
                count = 3
                try:
                    ## ask model to get category & nouns, add to category ##
                    category_prompt = akasha.prompts.format_category_prompt(
                        doc_text, self.language)
                    response = akasha.helper.call_model(
                        self.model_obj, category_prompt)

                    json_response = akasha.helper.extract_json(response)
                    if json_response is None:
                        raise Exception("Response Format Error")

                    for k, v in json_response.items():
                        if k not in set_category[v]:
                            category[v].append([k, doc_text])
                            set_category[v].add(k)
                    compare_resource = find_same_category(
                        category, cate_threshold)

                except Exception as e:
                    print("error during generate categories\n", e)
                    count -= 1
                    if count == 0:
                        break
            topic, nouns, used_texts = compare_resource
            used_texts = "\n".join(set(used_texts))
            nouns = ", ".join(nouns)
            try:
                q_prompt = akasha.prompts.compare_question_prompt(
                    self.question_style, topic, nouns, used_texts)
                response = akasha.helper.call_model(self.model_obj, q_prompt)

                if not self._process_response(response, used_texts, choice_num,
                                              ""):
                    raise Exception(f"Question Format Error, got {response}")

                self.doc_length += akasha.helper.get_docs_length(
                    self.language, docs)
                self.doc_tokens += self.model_obj.get_num_tokens(used_texts)
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

            new_table = akasha.format.handle_table(self.question[-1], docs,
                                                   self.answer[-1])
            for key in new_table:
                if key not in table:
                    table[key] = []
                table[key].append(new_table[key])

        progress.close()  # end running llm progress bar

        end_time = time.time()

        ### record logs ###
        if self.record_exp != "":
            params = akasha.format.handle_params(
                self.model,
                self.embeddings,
                self.chunk_size,
                self.search_type_str,
                self.topK,
                self.threshold,
                self.language,
            )
            metrics = akasha.format.handle_metrics(self.doc_length,
                                                   end_time - start_time,
                                                   self.doc_tokens)
            params["doc_range"] = doc_range
            akasha.aiido_upload(self.record_exp, params, metrics, table)

        if self.keep_logs == True:
            self._add_result_log(timestamp, end_time - start_time)

        self._save_questionset(timestamp, output_file_path)

        return self.question, self.answer

    def _eval_get_res_fact(self, question: Union[str, list], answer: str,
                           timestamp: str, retrivers_list: list) -> dict:
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
            prod_sys = self.system_prompt + akasha.prompts.default_doc_ask_prompt(
                self.language)
            query_with_prompt = question

        else:
            prod_sys = self.system_prompt
            query, ans = akasha.prompts.format_question_query(question, answer)
            query_with_prompt = akasha.prompts.format_llama_json(query)

        ### get docs ###
        self.docs, docs_len, docs_token = akasha.search.get_docs(
            self.db,
            self.embeddings_obj,
            retrivers_list,
            query,
            self.use_rerank,
            self.language,
            self.search_type,
            self.verbose,
            self.model,
            self.max_input_tokens,
        )

        ### ask llm ###
        try:
            intput_text = akasha.prompts.format_sys_prompt(
                prod_sys + self._display_docs(), query_with_prompt,
                self.prompt_format_type)
            response = akasha.helper.call_model(self.model_obj, intput_text)
            self.response.append(response)
            self.doc_length += docs_len
            self.doc_tokens += docs_token
        except Exception as e:
            traceback.print_exc()
            #response = ["running model error"]
            torch.cuda.empty_cache()
            logging.error(f"running model error\n {e}")
            raise e

        if self.question_style.lower() == "essay":

            if self.verbose:
                print("Question: ", question, "\n\n")
                print("Reference Answer: ", answer, "\n\n")
                print("Generated Response: ", response, "\n\n")

            self.score["bert"].append(
                eval.scores.get_bert_score(response, answer, self.language))
            self.score["rouge"].append(
                eval.scores.get_rouge_score(response, answer, self.language))
            self.score["llm_score"].append(
                eval.scores.get_llm_score(response, answer, self.eval_model,
                                          self.prompt_format_type))

            new_table = akasha.format.handle_table(
                question + "\nAnswer:  " + answer, self.docs, response)
            new_table = akasha.format.handle_score_table(
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

            new_table = akasha.format.handle_table(query + "\nAnswer:  " + ans,
                                                   self.docs, response)
            result = akasha.helper.extract_result(response)

            if str(ans).replace(' ', '') in str(result):
                self.score["correct_count"] += 1

        return new_table

    def _eval_get_res_summary(self, sum_doc: str, answer: str,
                              timestamp: str) -> dict:
        """generate summary resposne from the question, can evaluate with reference answer

        Args:
            sum_doc (str): text that need to be summarized
            answer (str): the reference answer of the summary
            timestamp (str): the timestamp of the auto evaluation function

        Returns:
            dict: evaluation result
        """

        prompt = "請對以下文件進行摘要: "
        intput_text = akasha.prompts.format_sys_prompt(
            self.system_prompt, prompt + "\n\n" + sum_doc,
            self.prompt_format_type)

        self.docs = [
            Document(page_content=sum_doc, metadata={
                "source": "",
                "page": 0
            })
        ]

        try:

            response = akasha.helper.call_model(self.model_obj, intput_text)
            self.response.append(response)
            self.doc_length += akasha.helper.get_doc_length(
                self.language, sum_doc)
            self.doc_tokens += self.model_obj.get_num_tokens(sum_doc)
        except Exception as e:
            traceback.print_exc()
            #response = ["running model error"]
            torch.cuda.empty_cache()
            logging.error(f"running model error\n {e}")
            raise e

        if self.verbose:
            print("Question: ", prompt + "\n" + sum_doc, "\n\n")
            print("Reference Answer: ", answer, "\n\n")
            print("Generated Response: ", response, "\n\n")

        self.score["bert"].append(
            eval.scores.get_bert_score(response, answer, self.language))
        self.score["rouge"].append(
            eval.scores.get_rouge_score(response, answer, self.language))
        self.score["llm_score"].append(
            eval.scores.get_llm_score(response, answer, self.eval_model,
                                      self.prompt_format_type))

        new_table = akasha.format.handle_table(prompt + "\nAnswer:  " + answer,
                                               self.docs, response)
        new_table = akasha.format.handle_score_table(
            new_table,
            self.score["bert"][-1],
            self.score["rouge"][-1],
            self.score["llm_score"][-1],
        )

        return new_table

    def _eval_get_res(self, question: Union[list, str], answer: str,
                      timestamp: str, retrivers_list: list) -> dict:
        """separate the question type and call different function to generate response

        Args:
            question (str): the question that need to be evaluated
            answer (str): the reference answer of the question
            timestamp (str): the timestamp of the auto evaluation function

        Returns:
            dict: _description_
        """
        if self.question_type.lower() in ["fact", "facts", "factoid", "factoids", "事實"] or \
            self.question_type.lower() in ["irre", "irrelevant", "irrelevance", "無關"] or\
            self.question_type.lower() in ["compared", "compare", "comparison", "comparisons", "比較"]:
            return self._eval_get_res_fact(question, answer, timestamp,
                                           retrivers_list)

        elif self.question_type.lower() in [
                "summary", "sum", "summarization", "summarize", "summaries",
                "摘要"
        ]:
            return self._eval_get_res_summary(question, answer, timestamp)

    def auto_create_questionset(
        self,
        doc_path: Union[List[str], str],
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
        self.doc_path = doc_path
        self.question_num = question_num

        if check_sum_type(self.question_type, self.question_style):
            return [], []

        ## check db ##

        if self.use_chroma:
            self.db, self.ignored_files = akasha.db.get_db_from_chromadb(
                self.doc_path, "use rerank to get docs")
        else:
            self.db, self.ignored_files = akasha.db.processMultiDB(
                self.doc_path, self.verbose, "eval_get_doc", self.embeddings,
                self.chunk_size, self.ignore_check)
        if not self._check_db():
            return [], []

        ## process of creating compare question is different from other, so we separate it ##
        if self.question_type in [
                "compare", "comparison", "comparisons", "比較", "compared"
        ]:
            return self._create_compare_questionset(choice_num,
                                                    output_file_path)

        ## set local variables ##
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")

        start_time = time.time()
        doc_range = (
            1999 + self.chunk_size
        ) // self.chunk_size  # doc_range is determine by the chunk size, so the select documents won't be too short to having trouble genereating a question

        vis_doc_range = set()
        self.doc_tokens, self.doc_length = 0, 0
        self.question, self.answer, self.docs = [], [], []
        table = {}
        ## add logs ##
        if self.keep_logs == True:
            self.timestamp_list.append(timestamp)
            self._add_basic_log(timestamp, "auto_create_questionset")
            self.logs[timestamp]["doc_range"] = doc_range
            self.logs[timestamp]["question_num"] = question_num
            self.logs[timestamp]["question_type"] = self.question_type
            self.logs[timestamp]["question_style"] = self.question_style
            self.logs[timestamp]["choice_num"] = choice_num

        texts = [doc.page_content for doc in self.db]
        metadata = [doc.metadata for doc in self.db]

        ## prevent doc_range - len of texts <= 0 ##
        doc_range = min(doc_range, len(texts) - 1)

        progress = tqdm(total=question_num,
                        desc=f"Create Q({self.question_type})")
        regenerate_limit = question_num
        ### random select a range of documents from the documents , and use llm model to generate a question and answer pair ###
        for i in range(question_num):
            print(" ")
            progress.update(1)
            print("\n")
            random_index = akasha.helper.get_non_repeat_rand_int(
                vis_doc_range,
                len(texts) - doc_range, doc_range)

            doc_text = "\n".join(texts[random_index:random_index + doc_range])
            docs = [
                Document(page_content=texts[k], metadata=metadata[k])
                for k in range(random_index, random_index + doc_range)
            ]
            source_files_name = get_source_files(metadata, random_index,
                                                 doc_range, self.language)
            try:
                q_prompt = akasha.prompts.format_create_question_prompt(
                    doc_text, self.question_type, self.question_style)
                response = akasha.helper.call_model(self.model_obj, q_prompt)

                if not self._process_response(response, doc_text, choice_num,
                                              source_files_name):
                    raise Exception(f"Question Format Error, got {response}")

                self.doc_length += akasha.helper.get_docs_length(
                    self.language, docs)
                self.doc_tokens += self.model_obj.get_num_tokens(doc_text)
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

            new_table = akasha.format.handle_table(self.question[-1], docs,
                                                   self.answer[-1])
            for key in new_table:
                if key not in table:
                    table[key] = []
                table[key].append(new_table[key])

        progress.close()  # end running llm progress bar

        end_time = time.time()

        ### record logs ###
        if self.record_exp != "":
            params = akasha.format.handle_params(
                self.model,
                self.embeddings,
                self.chunk_size,
                self.search_type_str,
                self.topK,
                self.threshold,
                self.language,
            )
            metrics = akasha.format.handle_metrics(self.doc_length,
                                                   end_time - start_time,
                                                   self.doc_tokens)
            params["doc_range"] = doc_range
            akasha.aiido_upload(self.record_exp, params, metrics, table)

        self._add_result_log(timestamp, end_time - start_time)

        self._save_questionset(timestamp, output_file_path)
        return self.question, self.answer

    def auto_evaluation(
        self,
        questionset_file: str,
        doc_path: Union[List[str], str],
        eval_model: str = "openai:gpt-3.5-turbo",
        **kwargs,
    ) -> Union[Tuple[float, float, float, int], Tuple[float, int]]:
        """parse the question set txt file generated from "auto_create_questionset" function and then use llm model to generate response,
        evaluate the performance of the given paramters based on similarity between responses and the default answers, use bert_score
        and rouge_l to evaluate the response if you use essay type to generate questionset.  And use correct_count to evaluate
        the response if you use single_choice type to generate questionset.  **Noted that the question_type must match the questionset_file's type**.


            Args:
            **questionset_flie (str)**: the path of question set txt file, accept .txt, .docx and .pdf.\n
            **question_type (str, optional)**: the type of question you want to generate, "essay" or "single_choice". Defaults to "essay".\n
            **eval_model (str, optional)**: llm model use to score the response. Defaults to "gpt-3.5-turbo".\n
            **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
                embeddings, chunk_size, model, verbose, language , search_type, record_exp,
                system_prompt, max_input_tokens, temperature.\n
        """

        ## set class variables ##
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        if isinstance(doc_path, akasha.db.dbs):
            self.doc_path = "use dbs object"
        else:
            self.doc_path = doc_path
        self.eval_model = eval_model
        if check_sum_type(self.question_type, self.question_style):
            return 0.0, 0
        self.system_prompt = check_essay_system_prompt(
            self.question_style, self.language,
            self.system_prompt) + self.system_prompt

        ## check db ##
        if isinstance(doc_path, akasha.db.dbs):
            self.db = kwargs['dbs']
            self.ignored_files = []
        elif self.use_chroma:
            self.db, self.ignored_files = akasha.db.get_db_from_chromadb(
                doc_path, self.embeddings)
        else:
            self.db, self.ignored_files = akasha.db.processMultiDB(
                doc_path, self.verbose, self.embeddings_obj, self.embeddings,
                self.chunk_size, self.ignore_check)
        if not self._check_db():
            return ""

        ## set local variables ##
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        start_time = time.time()
        self.doc_tokens, self.doc_length = 0, 0
        self.question, self.answer, self.docs = [], [], []
        search_dict = {}
        if self.question_style.lower() == "essay":
            self.score = {"bert": [], "rouge": [], "llm_score": []}
        else:
            self.score = {"correct_count": 0}
        table = {}
        total_docs = []
        question, answer = akasha.helper.get_question_from_file(
            questionset_file, self.question_style)
        self.question_num = len(question)
        progress = tqdm(total=self.question,
                        desc=f"Run Eval({self.question_style})")
        ## add logs ##
        if self.keep_logs == True:
            self.timestamp_list.append(timestamp)
            self._add_basic_log(timestamp, "auto_evaluation")
            self.logs[timestamp]["questionset_path"] = questionset_file
            self.logs[timestamp]["question_num"] = self.question_num
            self.logs[timestamp]["question_type"] = self.question_type
            self.logs[timestamp]["question_style"] = self.question_style
            self.logs[timestamp]["search_type"] = self.search_type_str

        ### for each question and answer, use llm model to generate response, and evaluate the response by bert_score and rouge_l ###
        retrivers_list = akasha.search.get_retrivers(
            self.db, self.embeddings_obj, self.use_rerank, self.threshold,
            self.search_type, search_dict)

        for i in range(self.question_num):
            print(" ")
            progress.update(1)
            print("\n")

            new_table = self._eval_get_res(question[i], answer[i], timestamp,
                                           retrivers_list)
            total_docs.extend(self.docs)
            # ---- #

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
        if self.keep_logs == True:
            self._add_result_log(timestamp, end_time - start_time)
            self.logs[timestamp]["response"] = self.response
            for k, v in search_dict.items():
                self.logs[timestamp][k] = v

        if self.question_style.lower() == "essay":
            avg_bert = round(
                sum(self.score["bert"]) / len(self.score["bert"]), 3)
            avg_rouge = round(
                sum(self.score["rouge"]) / len(self.score["rouge"]), 3)
            avg_llm_score = round(
                sum(self.score["llm_score"]) / len(self.score["llm_score"]), 3)
            if self.keep_logs == True:
                self.logs[timestamp]["bert"] = self.score["bert"]
                self.logs[timestamp]["rouge"] = self.score["rouge"]
                self.logs[timestamp]["llm_score"] = self.score["llm_score"]
            if self.record_exp != "":
                params = akasha.format.handle_params(
                    self.model,
                    self.embeddings,
                    self.chunk_size,
                    self.search_type_str,
                    self.topK,
                    self.threshold,
                    self.language,
                )
                metrics = akasha.format.handle_metrics(self.doc_length,
                                                       end_time - start_time,
                                                       self.doc_tokens)
                metrics["avg_bert"] = avg_bert
                metrics["avg_rouge"] = avg_rouge
                metrics["avg_llm_score"] = avg_llm_score
                akasha.aiido_upload(self.record_exp, params, metrics, table)

            return avg_bert, avg_rouge, avg_llm_score, self.doc_tokens

        else:
            correct_rate = (self.score["correct_count"] / self.question_num)
            if self.keep_logs == True:
                self.logs[timestamp]["correct_rate"] = correct_rate

            if self.record_exp != "":
                params = akasha.format.handle_params(
                    self.model,
                    self.embeddings,
                    self.chunk_size,
                    self.search_type_str,
                    self.topK,
                    self.threshold,
                    self.language,
                )
                metrics = akasha.format.handle_metrics(self.doc_length,
                                                       end_time - start_time,
                                                       self.doc_tokens)
                metrics["correct_rate"] = (self.score["correct_count"] /
                                           self.question_num)
                akasha.aiido_upload(self.record_exp, params, metrics, table)

            return correct_rate, self.doc_tokens

    def optimum_combination(
        self,
        questionset_flie: str,
        doc_path: Union[List[str], str],
        embeddings_list: list = ["openai:text-embedding-ada-002"],
        chunk_size_list: list = [500],
        model_list: list = ["openai:gpt-3.5-turbo"],
        search_type_list: list = ["svm", "tfidf", "mmr"],
        **kwargs,
    ) -> Tuple[list, list]:
        """test all combinations of giving lists, and run auto_evaluation to find parameters of the best result.

        Args:
            **questionset_flie (str)**: the path of question set txt file, accept .txt, .docx and .pdf.\n
            **doc_path (str)**: documents directory path\n
            **question_type (str, optional)**: the type of question you want to generate, "essay" or "single_choice". Defaults to "essay".\n
            **embeddings_list (_type_, optional)**: list of embeddings models. Defaults to ["openai:text-embedding-ada-002"].\n
            **chunk_size_list (list, optional)**: list of chunk sizes. Defaults to [500].\n
            **model_list (_type_, optional)**: list of models. Defaults to ["openai:gpt-3.5-turbo"].\n
            **threshold (float, optional)**: (deprecated) the similarity threshold of searching. Defaults to 0.2.\n
            **search_type_list (list, optional)**: list of search types, currently have "merge", "svm", "knn", "tfidf", "mmr". Defaults to ['svm','tfidf','mmr'].
        Returns:
            (list,list): return best score combination and best cost-effective combination
        """

        self._change_variables(**kwargs)
        start_time = time.time()
        combinations = akasha.helper.get_all_combine(embeddings_list,
                                                     chunk_size_list,
                                                     model_list,
                                                     search_type_list)
        progress = tqdm(len(combinations),
                        total=len(combinations),
                        desc="RUN LLM COMBINATION")
        print("\n\ntotal combinations: ", len(combinations))
        result_list = []
        if self.question_type.lower() == "essay":
            bcb = 0.0
            bcr = 0.0
            bcl = 0.0
        else:
            bcr = 0.0

        for embed, chk, mod, st in combinations:
            progress.update(1)

            if self.question_type.lower() == "essay":
                cur_bert, cur_rouge, cur_llm, tokens = self.auto_evaluation(
                    questionset_flie,
                    doc_path,
                    embeddings=embed,
                    chunk_size=chk,
                    model=mod,
                    search_type=st,
                )

                bcb = max(bcb, cur_bert)
                bcr = max(bcr, cur_rouge)
                bcl = max(bcl, cur_llm)
                cur_tup = (
                    cur_bert,
                    cur_rouge,
                    cur_llm,
                    embed,
                    chk,
                    mod,
                    self.search_type_str,
                )
            else:
                cur_correct_rate, tokens = self.auto_evaluation(
                    questionset_flie,
                    doc_path,
                    embeddings=embed,
                    chunk_size=chk,
                    model=mod,
                    search_type=st,
                )
                bcr = max(bcr, cur_correct_rate)
                cur_tup = (
                    cur_correct_rate,
                    cur_correct_rate / tokens,
                    embed,
                    chk,
                    mod,
                    self.search_type_str,
                )
            result_list.append(cur_tup)

        progress.close()

        if self.question_type.lower() == "essay":
            ### record bert score logs ###
            print("Best Bert Score: ", "{:.3f}".format(bcb))

            bs_combination = akasha.helper.get_best_combination(result_list, 0)
            print("\n\n")

            ### record rouge score logs ###
            print("Best Rouge Score: ", "{:.3f}".format(bcr))

            rs_combination = akasha.helper.get_best_combination(result_list, 1)
            print("\n\n")

            ### record llm_score logs ###
            print("Best llm score: ", "{:.3f}".format(bcl))
            # score_comb = "Best score combination: \n"
            # print(score_comb)
            # logs.append(score_comb)
            ls_combination = akasha.helper.get_best_combination(result_list, 2)
            print("\n\n")

        else:
            ### record logs ###
            print("Best correct rate: ", "{:.3f}".format(bcr))
            score_comb = "Best score combination: \n"
            print(score_comb)

            bs_combination = akasha.helper.get_best_combination(result_list, 0)

            print("\n\n")
            cost_comb = "Best cost-effective: \n"
            print(cost_comb)

            bc_combination = akasha.helper.get_best_combination(result_list, 1)

        end_time = time.time()
        format_time = "time spend: " + "{:.3f}".format(end_time - start_time)
        print(format_time)

        if self.question_type.lower() == "essay":
            return bs_combination, rs_combination, ls_combination
        return bs_combination, bc_combination

    def create_topic_questionset(
        self,
        doc_path: Union[List[str], str],
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
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        if isinstance(doc_path, akasha.db.dbs):
            self.doc_path = "use dbs object"
        else:
            self.doc_path = doc_path
        self.question_num = question_num

        if check_sum_type(self.question_type, self.question_style, "related"):
            return [], []

        ## check db ##
        if isinstance(doc_path, akasha.db.dbs):
            self.db = kwargs['dbs']
            self.ignored_files = []
        elif self.use_chroma:
            self.db, self.ignored_files = akasha.db.get_db_from_chromadb(
                self.doc_path, self.embeddings)
        else:
            self.db, self.ignored_files = akasha.db.processMultiDB(
                self.doc_path, self.verbose, self.embeddings_obj,
                self.embeddings, self.chunk_size, self.ignore_check)
        if not self._check_db():
            return [], []

        ## set local variables ##
        doc_range = (
            1999 + self.chunk_size
        ) // self.chunk_size  # doc_range is determine by the chunk size, so the select documents won't be too short to having trouble genereating a question
        vis_doc_range = set()
        self.doc_tokens, self.doc_length = 0, 0
        self.question, self.answer, self.docs = [], [], []
        table = {}
        search_dict = {}
        retrivers_list = akasha.search.get_retrivers(
            self.db, self.embeddings_obj, self.use_rerank, self.threshold,
            self.search_type, search_dict)
        ## add logs ##
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        start_time = time.time()
        if self.keep_logs == True:
            self.timestamp_list.append(timestamp)
            self._add_basic_log(timestamp, "related_questionset")
            self.logs[timestamp]["doc_range"] = doc_range
            self.logs[timestamp]["question_num"] = question_num
            self.logs[timestamp]["question_type"] = self.question_type
            self.logs[timestamp]["question_style"] = self.question_style
            self.logs[timestamp]["choice_num"] = choice_num
            self.logs[timestamp]["topic"] = topic

        ## search related documents ##
        self.docs, docs_len, docs_token = akasha.search.get_docs(
            self.db,
            self.embeddings_obj,
            retrivers_list,
            topic,
            self.use_rerank,
            self.language,
            self.search_type,
            self.verbose,
            self.model,
            50000,
            False,
        )

        texts = [doc.page_content for doc in self.docs]
        metadata = [doc.metadata for doc in self.docs]

        doc_range = min(doc_range, len(texts) - 1)

        progress = tqdm(total=question_num,
                        desc=f"Create Q({self.question_type})")
        regenerate_limit = question_num
        ### random select a range of documents from the documents , and use llm model to generate a question and answer pair ###
        for i in range(question_num):
            print(" ")
            progress.update(1)
            print("\n")
            random_index = akasha.helper.get_non_repeat_rand_int(
                vis_doc_range,
                len(texts) - doc_range, doc_range)

            doc_text = "\n".join(texts[random_index:random_index + doc_range])
            docs = [
                Document(page_content=texts[k], metadata=metadata[k])
                for k in range(random_index, random_index + doc_range)
            ]
            source_files_name = get_source_files(metadata, random_index,
                                                 doc_range, self.language)

            try:
                q_prompt = akasha.prompts.format_create_question_prompt(
                    doc_text, self.question_type, self.question_style, topic)

                response = akasha.helper.call_model(self.model_obj, q_prompt)
                if not self._process_response(response, doc_text, choice_num,
                                              source_files_name):
                    raise Exception(f"Question Format Error, got {response}")

                self.doc_length += akasha.helper.get_docs_length(
                    self.language, docs)
                self.doc_tokens += self.model_obj.get_num_tokens(doc_text)
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

            new_table = akasha.format.handle_table(self.question[-1], docs,
                                                   self.answer[-1])
            for key in new_table:
                if key not in table:
                    table[key] = []
                table[key].append(new_table[key])

        progress.close()  # end running llm progress bar

        end_time = time.time()

        ### record logs ###
        if self.record_exp != "":
            params = akasha.format.handle_params(
                self.model,
                self.embeddings,
                self.chunk_size,
                self.search_type_str,
                self.topK,
                self.threshold,
                self.language,
            )
            metrics = akasha.format.handle_metrics(self.doc_length,
                                                   end_time - start_time,
                                                   self.doc_tokens)
            params["doc_range"] = doc_range
            akasha.aiido_upload(self.record_exp, params, metrics, table)

        self._add_result_log(timestamp, end_time - start_time)

        self._save_questionset(timestamp, output_file_path)

        return self.question, self.answer


### sub func###
def check_sum_type(question_type: str,
                   question_style: str,
                   func: str = "auto") -> bool:
    ## check question_type and question_style, summary can not be used in single_choice question_type ##
    if func != "auto" and question_type.lower() in [
            'compare', 'comparison', 'comparisons', '比較', 'compared'
    ]:
        print("compare can not be used in related_questionset\n\n")
        return True
    if question_type.lower() in [
            "summary", "sum", "summarization", "summarize", "summaries", "摘要"
    ] and question_style.lower() == "single_choice":
        print("summary can not be used in single_choice question_type\n\n")

        return True

    return False


def check_essay_system_prompt(question_style: str, language: str,
                              system_prompt: str) -> str:
    ## check question_style and language, if question_style is essay and language is chinese, add prompt ##
    if question_style.lower() == "essay" and language.lower(
    ) == "ch" and "用中文回答" not in system_prompt:
        return " 用中文回答"

    return ""


def find_same_category(
    category: dict,
    cate_threshold: int,
) -> Union[list, bool]:
    """iterate the category dictionary and check if any category has more than cate_threshold items,
    if yes, return the category name. 

    Args:
        category (dict): {"category_name":[[item1,doc1], [item2,doc2],...], ...}
        cate_threshold (int): default 3

    Returns:
        _type_: _description_
    """
    for k, v in category.items():
        if len(v) >= cate_threshold:
            res = [
                k, [noun[0] for noun in category[k]],
                [doc[1] for doc in category[k]]
            ]
            return res
    return False


def get_source_files(metadata: list,
                     random_index: int,
                     doc_range: int,
                     language: str = "ch") -> str:
    """from metadata of selected document chunks, get the non repeated source file name

    Args:
        metadata (list): list of metadata dict of document chunks
        random_index (int): selected document chunks start index
        doc_range (int): selected document chunks range

    Returns:
        str: source file name
    """
    file_sources = set()

    for i in range(random_index, random_index + doc_range):
        try:
            if metadata[i]["source"] != "":
                file_name = ''.join(
                    (metadata[i]["source"].split('/')[-1]).split('.')[:-1])
                file_sources.add(file_name)
        except:
            continue

    if len(file_sources) == 0:
        return ""

    if "chinese" in akasha.format.language_dict[language]:
        return "根據文件\"" + "、".join(list(file_sources)) + "\"，"

    return "based on file \"" + "、".join(list(file_sources)) + "\","
