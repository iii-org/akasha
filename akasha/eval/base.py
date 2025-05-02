from typing import Union
import numpy as np
from akasha.utils.prompts.format import language_dict
from pathlib import Path
import json


### sub func###
def get_non_repeat_rand_int(vis: set, num: int, doc_range: int):
    temp = np.random.randint(num)
    if len(vis) >= num // 2:
        vis = set()
    if (temp not in vis) and (temp + doc_range - 1 not in vis):
        for i in range(doc_range):
            vis.add(temp + i)
        return temp
    return get_non_repeat_rand_int(vis, num, doc_range)


def check_sum_type(question_type: str, question_style: str, func: str = "auto") -> bool:
    ## check question_type and question_style, summary can not be used in single_choice question_type ##
    if func != "auto" and question_type.lower() in [
        "compare",
        "comparison",
        "comparisons",
        "比較",
        "compared",
    ]:
        print("compare can not be used in related_questionset\n\n")
        raise ValueError("compare can not be used in related_questionset")
        # return True
    if (
        question_type.lower()
        in ["summary", "sum", "summarization", "summarize", "summaries", "摘要"]
        and question_style.lower() == "single_choice"
    ):
        print("summary can not be used in single_choice question_type\n\n")
        raise ValueError("summary can not be used in single_choice question_type")
        # return True

    return False


def check_essay_system_prompt(
    question_style: str, language: str, system_prompt: str
) -> str:
    ## check question_style and language, if question_style is essay and language is chinese, add prompt ##
    if (
        question_style.lower() == "essay"
        and language.lower() == "ch"
        and "用中文回答" not in system_prompt
    ):
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
                k,
                [noun[0] for noun in category[k]],
                [doc[1] for doc in category[k]],
            ]
            return res
    return False


def get_source_files(
    metadata: list, random_index: int, doc_range: int, language: str = "ch"
) -> str:
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
                file_name = "".join(
                    (metadata[i]["source"].split("/")[-1]).split(".")[:-1]
                )
                file_sources.add(file_name)
        except Exception:
            continue

    if len(file_sources) == 0:
        return ""

    if "chinese" in language_dict[language]:
        return '根據文件"' + "、".join(list(file_sources)) + '"，'

    return 'based on file "' + "、".join(list(file_sources)) + '",'


def get_question_from_file(
    path: str, question_type: str, question_style: str
) -> Union[list, list, str, str]:
    """load questions from file and save the questions into lists.
    a question list include the question, mutiple options, and the answer (the number of the option),
      and they are all separate by space in the file.
       if the questionset file has question_type and question_style key, it will replace the question_type and question_style with the value.
    Args:
        **path (str)**: path of the question file\n
        **question_type (str)**: question type, default is "fact"\n
        **question_style (str)**: question style, default is "essay"\n


    Returns:
        list: list of question list
    """
    f_path = Path(path)

    # Ensure the file exists
    if not f_path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")

    # Read the JSON file
    with f_path.open(mode="r", encoding="utf-8") as file:
        content = json.load(file)

    # Validate the JSON structure
    if not all(key in content for key in ["question", "answer"]):
        raise ValueError("The JSON file does not contain 'question' and 'answer' keys.")

    if "question_type" in content:
        question_type = content["question_type"]
    if "question_style" in content:
        question_style = content["question_style"]

    questions = content["question"]
    answers = []

    if question_style.lower() == "essay":
        # content = content.split("\n\n")
        # for i in range(len(content)):
        #     if content[i] == "":
        #         continue

        #     try:
        #         process = "".join(content[i].split("問題：")).split("答案：")
        #         if len(process) < 2:
        #             raise SyntaxError("Question Format Error")
        #     except:
        #         process = "".join(content[i].split("問題:")).split("答案:")
        #         if len(process) < 2:
        #             continue

        #     questions.append(process[0])
        #     answers.append(process[1])
        # return questions, answers
        answers = content["answer"]
        return questions, answers, question_type, question_style
    else:
        for idx, words in enumerate(content["answer"]):
            questions[idx] = [questions[idx]]
            questions[idx].extend(words[:-1])
            answers.append(words[-1])
    # for con in content.split("\n"):
    #     if con == "":
    #         continue
    #     words = [word for word in con.split("\t") if word != ""]

    #     questions.append(words[:-1])
    #     answers.append(words[-1])

    # return questions, answers
    return questions, answers, question_type, question_style
