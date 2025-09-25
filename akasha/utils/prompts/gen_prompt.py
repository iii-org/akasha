from typing import List, Union, Tuple
from akasha.utils.prompts.format import language_dict
from urllib.parse import urlparse
import os
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

sys_s = "[INST] <<SYS>> "
sys_e = " <<SYS>> [/INST]\n\n"


def _separate_name(name: str):
    """separate type:name by ':'

    Args:
        **name (str)**: string with format "type:name" \n

    Returns:
        (str, str): res_type , res_name
    """
    sep = name.split(":")

    if len(sep) > 2:
        res_type = sep[0].lower()
        res_name = ":".join(sep[1:])
    elif len(sep) < 2:
        ### if the format type not equal to type:name ###
        res_type = sep[0].lower()
        res_name = ""
    else:
        res_type = sep[0].lower()
        res_name = sep[1]

    return res_type, res_name


def format_chat_gemini_prompt(system_prompt: str, prompt: str) -> List[dict]:
    if system_prompt == "" and prompt == "":
        return []

    if system_prompt == "":
        return [{"role": "user", "parts": [prompt]}]

    if prompt == "":
        return [{"role": "model", "parts": [system_prompt]}]

    return [
        {
            "role": "model",
            "parts": [system_prompt]
        },
        {
            "role": "user",
            "parts": [prompt]
        },
    ]


def format_chat_gemma_prompt(system_prompt: str, prompt: str) -> List[dict]:
    if system_prompt == "" and prompt == "":
        return []

    if system_prompt == "":
        return [{"role": "user", "content": prompt}]

    if prompt == "":
        return [{"role": "user", "content": system_prompt}]

    return [{"role": "user", "content": system_prompt + "\n\n" + prompt}]


def format_chat_gpt_prompt(system_prompt: str, prompt: str) -> List[dict]:
    if system_prompt == "" and prompt == "":
        return []

    if system_prompt == "":
        return [{"role": "user", "content": prompt}]

    if prompt == "":
        return [{"role": "system", "content": system_prompt}]

    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        },
    ]


def format_chat_mistral_prompt(system_prompt: str, prompt: str) -> List[dict]:
    if system_prompt == "" and prompt == "":
        return []

    if system_prompt == "":
        return [{"role": "user", "content": prompt}]

    if prompt == "":
        return [
            {
                "role": "user",
                "content": "start conversation."
            },
            {
                "role": "assistant",
                "content": system_prompt
            },
        ]

    return [
        {
            "role": "user",
            "content": "start conversation."
        },
        {
            "role": "assistant",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        },
    ]


def format_llama_sys_prompt(system_prompt: str, prompt: str) -> str:
    if system_prompt == "" and prompt == "":
        return ""

    if system_prompt == "":
        return "[INST] " + prompt + " [/INST]\n"
    if prompt == "":
        return "[INST] <<SYS>> " + system_prompt + " <<SYS>> \n"

    return "[INST] <<SYS>> " + system_prompt + " <<SYS>> \n\n" + prompt + " [/INST]\n"


def format_GPT_sys_prompt(system_prompt: str, prompt: str) -> str:
    if system_prompt == "" and prompt == "":
        return ""

    if system_prompt == "":
        return "Human: " + prompt + "\n\n"
    if prompt == "":
        return "System: " + system_prompt + "\n"

    return "System: " + system_prompt + "\n\n" + "Human: " + prompt + "\n"


def format_sys_prompt(
    system_prompt: str,
    prompt: str,
    prompt_format_type: str = "auto",
    model: str = "remote:xxx",
) -> Union[str, List[dict]]:
    if prompt_format_type.lower() == "auto":
        prompt_format_type = decide_auto_prompt_format_type(model)

    if prompt_format_type.lower() == "llama":
        ret_text = format_llama_sys_prompt(system_prompt, prompt)

    elif prompt_format_type.lower() == "chat_gpt":
        ret_text = format_chat_gpt_prompt(system_prompt, prompt)

    elif prompt_format_type.lower() == "chat_mistral":
        ret_text = format_chat_mistral_prompt(system_prompt, prompt)

    elif prompt_format_type.lower() == "chat_gemma":
        ret_text = format_chat_gemma_prompt(system_prompt, prompt)

    elif prompt_format_type.lower() == "chat_gemini":
        ret_text = format_chat_gemini_prompt(system_prompt, prompt)

    else:
        ret_text = format_GPT_sys_prompt(system_prompt, prompt)

    return ret_text


def decide_auto_prompt_format_type(model: str = "remote:xxx") -> str:
    """since prompt_format_type is "auto", so we based on the model name to decide which prompt format type to use

    Args:
        model (str, optional): model_type and model name. Defaults to "remote:xxx".

    Returns:
        str: _description_
    """
    model_type, model_name = _separate_name(model)

    if "openai" in model_type or "openai" in model_name:
        return "chat_gpt"
    elif "gemini" in model_type or model_type in ["google", "gemi"]:
        return "chat_gemini"
    elif ("anthropic" in model_type or "claude" in model_name
          or "mistral" in model_name or "mixtral" in model_name
          or "gemma" in model_name):
        return "chat_mistral"

    elif "llama" in model_type or "llama" in model_name:
        return "llama"

    return "gpt"


def format_history_prompt(
    history_messages: list,
    prompt_format_type: str = "chat_gpt",
    user_tag="user",
    assistant_tag="assistant",
) -> List[dict]:
    prod_history = []
    if isinstance(history_messages, str):
        history_messages = [history_messages]

    for idx, msg in enumerate(history_messages):
        if idx % 2 == 0:
            prod_history.append({"role": user_tag, "content": msg})
        else:
            prod_history.append({"role": assistant_tag, "content": msg})

    return prod_history


def format_question_query(question: list, answer: str) -> Tuple[str, str]:
    """generate a certain format of question to input to llm. Last element means which selection is the correct answer.
       return the question query string and the answer string.\n
      example:    ["what is 1+1 euqals to?", "2", "4", "8", "10", "1"]
      after :
        query = " what is 1+1 euqals to?
            1. 2
            2. 4
            3. 8
            4. 10
            "
        ans = "1"
    Args:
        **question (list)**: list of question and selections\n

    Returns:
        (str, str): return the question query string and the answer string
    """
    n = len(question)
    if n == 0:
        return "", ""
    elif n == 1:
        return "Question: " + question[0], ""
    query = "Question: " + question[0] + "\n"

    for i in range(1, n):
        query += question[i] + "\n"

    return query, answer


def format_llama_json(query: str):
    """insert system prompt for llm to generate JSON format of {"ans":selection number}

    Args:
        **query (str)**: a question string with selections that we want llm to answer\n

    Returns:
       prompt (str): the whole prompt includes system prompt and question
    """

    sys_prompt = (
        "human will give you a question with several possible answer, use the content of documents "
        +
        "to choose correct answer. and you need to return the answer of this question as"
        +
        ' JSON structure with key "ans", and only this JSON structure, please don\'t add any other word. for example, '
        +
        'User: what is 1+1 euqals to? 1.「2」  2.「4」 3.「10」 4.「15」 \n you: {"ans":1}'
    )

    prompt = sys_s + sys_prompt + sys_e
    return prompt + query


def format_chinese_json(query: str):
    """system prompt for single choice question in chinese

    Args:
        **query (str)**: question string\n

    Returns:
        str: combined prompt
    """
    sys_prompt = ("### 指令: 我会给出一个问题和几个可能的选项，请只根据提供的文件找到其中正确的一个答案，" +
                  "並回答答案為第幾個選項。若沒有提供，請照你的知識回答，并将答案以JSON的格式表示，如答案為第一個選項，" +
                  "回答的格式為{'ans':1}，不要添加其他字。  ### 问题和选项:\n")
    # ts_prompt = "### Instruction: I will provide a question and several possible options in the input. Please find the correct answer based solely on the provided texts, and respond with the number of the option that is the correct answer. If no texts is provided, please respond based on your knowledge, and format the answer in JSON format. For example, if the answer is the first option, the format of the response should be {'Answer': 1}. Please do not add any additional words. ### Input:"
    # sys_prompt = ts_prompt + query "  ### Response:"
    return sys_prompt + query


def format_wrong_answer(num: int, doc_text: str, question: str,
                        correct_ans: str) -> str:
    """prompt for generate wrong answers to create single choice question

    Args:
        **num (int)**: number of wrong answers that we want to generate\n
        **doc_text (str)**: document texts that used to generate question\n
        **question (str)**: question string\n
        **correct_ans (str)**: correct answer string\n

    Returns:
        str: combined prompt
    """

    q_prompt = (
        sys_s +
        f"根據以下的文件、問題和正確答案，請基於文件、問題和正確答案生成{num}個錯誤答案，錯誤答案應該與正確答案有相關性但數字、內容或定義錯誤，或者與正確答案不相同但有合理性。並注意各個錯誤答案必須都不相同。\n\n示例格式：\n<開始文件>\n...\n<結束文件>\n<開始問題>\n...\n<結束問題>\n<開始正確答案>\n...\n<結束正確答案>\n\n錯誤答案：錯誤答案1在這里\n\n錯誤答案：錯誤答案2在這里\n\n錯誤答案：錯誤答案3在這里\n\n。開始吧！"
        + sys_e + "<開始文件>\n")
    end_doc = "<結束文件>\n"
    st_q = "<開始問題>\n"
    end_q = "<結束問題>\n"
    st_cor = "<開始正確答案>\n"
    end_cor = "<結束正確答案>\n"
    q_prompt = (q_prompt + doc_text + end_doc + st_q + question + end_q +
                st_cor + correct_ans + end_cor + "\n\n")

    return q_prompt


def format_create_question_prompt(
    doc_text: str,
    question_type: str = "fact",
    question_style: str = "essay",
    topic: str = "",
    system_prompt: str = "",
) -> str:
    """prompts for auto generate question from document

    Args:
        **doc_text (str)**: texts from documents\n
        **question_type (str)**: question type, can be "single choice", "essay"\n

    Returns:
        str: combined prompt
    """
    qt = ""
    question_type = question_type.lower()
    question_style = question_style.lower()
    if question_style not in [
            "essay", "問答", "essay question", "essay_question"
    ]:
        qt = "少於100字的"

    if question_type in ["fact", "facts", "factoid", "factoids", "事實"]:
        if system_prompt == "":
            system_prompt = "人類：您是一位教師，正在為測驗準備問題。\n"
        q_prompt = fact_question_prompt(qt, topic)
    elif question_type in [
            "summary",
            "sum",
            "summarization",
            "summarize",
            "summaries",
            "摘要",
    ]:
        q_prompt = summary_question_prompt(qt)
    elif question_type in ["irre", "irrelevant", "irrelevance", "無關"]:
        if system_prompt == "":
            system_prompt = "人類：我想測試語言模型根據文件回答的可靠性。\n"
        q_prompt = irre_question_prompt(qt, topic)

    # end_prompt = "<End Document>\n"
    end_prompt = "<結束文件>\n"
    # generate question prompt = generate_question_prompt(Document)
    q_prompt = q_prompt + doc_text + end_prompt

    return system_prompt + q_prompt


def fact_question_prompt(qt: str = "", topic: str = "") -> str:
    """prompt for generate fact question"""

    if topic != "":
        topic = f"相關{topic}的"

    fact_prompt = f"請基於文件只生成一個{topic}問題和一個{qt}答案，問題應該詳細並且明確基於文件中的訊息，但不要使用'根據文件中的訊息'等詞語取代想詢問的問題內容。"
    format_prompt = "\n\n示例格式：\n<開始文件>\n...\n<結束文件>\n問題：問題在這里\n答案：答案在這里\n\n。開始吧！"
    return sys_s + fact_prompt + format_prompt + sys_e + "<開始文件>\n"


def summary_question_prompt(qt: str = "") -> str:
    """prompt for generate summary question"""

    sum_prompt = '請根據以下文件進行摘要，並將摘要放在"答案"後面.'
    format_prompt = ("\n\n示例格式：\n<開始文件>\n...\n<結束文件>\n答案：摘要在這裡\n\n。開始吧！")
    return sys_s + sum_prompt + format_prompt + sys_e + "<開始文件>\n"


def irre_question_prompt(qt: str = "", topic: str = "") -> str:
    """prompt for generate irrelevant question"""
    if topic != "":
        topic = f"和{topic}"

    irre_prompt = f"請只生成一個名詞或領域與文件內容{topic}相關，但文件內容中不存在的問題。這代表說，詢問任何人這個問題與文件，都無法回答該問題。問題應該詳細但不要使用'根據文件中的訊息'等詞語取代想詢問的問題內容。根據這個問題和文件，生成一個{qt}回答"
    format_prompt = "\n\n示例格式：\n<開始文件>\n...\n<結束文件>\n問題：問題在這里\n答案：答案在這里\n\n。開始吧！"
    return sys_s + irre_prompt + format_prompt + sys_e + "<開始文件>\n"


def compare_question_prompt(question_style: str, topic: str, nouns: str,
                            used_texts: str):
    """prompt for generate compare question"""

    qt = ""
    if question_style not in [
            "essay", "問答", "essay question", "essay_question"
    ]:
        qt = "少於100字的"

    com_prompt = f"人類：您是一位教師，正在為測驗準備問題。\n請基於文件和相關主題只生成一個相關主題之間的比較問題和一個{qt}答案，問題應該詳細並且明確基於文件中的訊息，但不要使用'根據文件中的訊息'等詞語取代想詢問的問題內容。"
    format_prompt = "\n\n示例格式：\n<開始文件>\n...\n<結束文件>\n<開始相關主題>\n...\n<結束相關主題>\n問題：問題在這里\n答案：答案在這里\n\n。開始吧！"

    return (sys_s + com_prompt + format_prompt + sys_e + "<開始文件>\n" +
            used_texts + "<結束文件>\n" + "<開始相關主題>\n" + nouns + "<結束相關主題>\n")


def format_llm_score(cand: str, ref: str):
    """the system prompt for llm to calculate the cnadidate is correct or not.

    Args:
        **cand (str)**: llm generated response that we want to test the performance\n
        **ref (str)**: reference answer\n
    """

    sys_prompt = (
        "human will give you a [candidate] sentence and a [reference] sentence, please score the [candidate] sentence "
        +
        "based on the [reference] sentence, the higher score means the [candidate] sentence has enough information and correct answer that [reference] sentence has."
        +
        "remember, you can only return the score and need to return the score of this [candidate] sentence as a float number range from 0 to 1.\n"
        +
        "Example Format:\n Human: [candidate]: ...\n\n [reference]: ...\n\n You: 0.8\n\n"
    )

    # prompt = sys_s + sys_prompt + sys_e
    return sys_prompt, "[candidate]: " + cand + "\n\n[reference]: " + ref + "\n\n"


def format_reduce_summary_prompt(cur_text: str,
                                 summary_len: int = 500,
                                 language: str = "zh"):
    """the prompt for llm to generate a summary of the given text

    Args:
        **cur_text (str)**: the text that we want llm to generate a summary\n
        **summary_len (int, optional)**: the summary word length we want llm to generate. Defaults to 500.\n

    Returns:
        str: summary prompt.
    """
    target_language = language_dict[language]
    underline = "------------"

    if summary_len > 0:
        if "chinese" in target_language:
            sys_prompt = (
                f"將以下內容總結成一個{summary_len}字的摘要，一步一步來，如果你做得好，我會給你100美元小費。\n\n" +
                underline + "\n" + cur_text + underline)
        else:
            sys_prompt = (
                f"Write a concise {summary_len} words summary of the following text. Do it step by step and I will tip you 100 bucks if you are doing well.\n\n"
                + underline + "\n" + cur_text + underline)

    else:
        if "chinese" in target_language:
            sys_prompt = ("將以下內容總結成一個摘要，一步一步來，如果你做得好，我會給你100美元小費。\n\n" +
                          underline + "\n" + cur_text + underline)
        else:
            sys_prompt = (
                "Write a concise summary of the following. Do it step by step and I will tip you 100 bucks if you are doing well.\n\n"
                + underline + "\n" + cur_text + underline)

    return sys_prompt


def format_refine_summary_prompt(cur_text: str,
                                 previous_summary: str,
                                 summary_len: int = 500,
                                 language: str = "zh"):
    """the prompt for llm to generate the summary of the given text and previous summary

     Args:
        **cur_text (str)**: the text that we want llm to generate a summary\n
        **previous_summary (str)**: the previous summary that we want llm to generate a summary\n
        **summary_len (int, optional)**: the summary word length we want llm to generate. Defaults to 500.\n

    Returns:
        str: summary prompt.

    """
    target_language = language_dict[language]
    if "chinese" in target_language:
        sys_prompt = f"""將以下內容總結成一個{summary_len}字的摘要。
        我們已經提供了一個現有的摘要，原始摘要如下：{previous_summary}
        ------------\n
        {cur_text}\n
        ------------\n
        根據新的內容，修改原始摘要。
        如果新的內容不夠有用，請返回原始摘要。
        一步一步來，如果你做得好，我會給你100美元小費。
        """
    else:
        sys_prompt = f"""Your job is to produce a final summary of {summary_len} words.
        We have provided an existing summary up to a certain point, original summary is:  {previous_summary}
        ------------\n
        {cur_text}\n
        ------------\n
        Given the new context, refine the original summary.
        If the context isn't useful, return the original summary.
        Do it step by step and I will tip you 100 bucks if you are doing well.
        """
    return sys_prompt


def format_compression_prompt(query: str, doc: str):
    return f"""Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return an empty string. 

    Remember, *DO NOT* edit the extracted parts of the context.
    \nQuestion: {query}
    \nContext:
    {doc}"""


def format_pic_summary_prompt(chunk_size: int = 500):
    # f"please use traditional chinese to describe this picture in {chunk_size} words.\n\n"
    return "please use traditional chinese to describe this picture in details.\n\n"


def default_doc_ask_prompt(language: str = "zh"):
    prompt = """Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"""
    if "chinese" in language_dict[language]:
        prompt += "用中文回答下列問題：\n"

    return prompt


def default_ask_prompt(language: str = "zh"):
    prompt = """你是擅長回答問題的助手，若有提供參考文件，只根據參考文件回答問題;若沒有參考文件，則根據你的理解回答使用者的問題。\n"""

    if "chinese" in language_dict[language]:
        prompt += "用中文回答下列問題：\n"

    return prompt


def default_conclusion_prompt(question: str, language: str = "zh"):
    prompt = f"""User will give you several pieces of answer about a question, use those context to answer the question: {question}. 
Remember to response non-repeated, detailed and coherent answer..\n"""
    if "chinese" in language_dict[language]:
        prompt += "用中文回答。\n"

    return prompt


def default_doc_grader_prompt():
    return """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test, so if it's possible relevant to the question, grade it as relevant. 
    The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""


def default_answer_grader_prompt():
    return """You are a grader assessing whether an answer addresses / resolves a question \n 
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""


DEFAULT_CHINESE_ANS_PROMPT = " A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:用中文回答問題。"


def format_category_prompt(doc_text, language: str):
    """prompt for generate category for proper nouns"""

    if "chinese" in language_dict[language]:
        prompt = "用中文回答"
    else:
        prompt = ""
    prompt += f"""The document discuss some proper nouns, please find most important two of them and give those "proper noun" a general "category", categories like company, system, flower, software. The format of response is
    {{
      "proper noun": "category",
      "proper noun": "category",
    }}\n
    For example, the document may discuss the 5G, since 5G is a kind of communication technology, you should output "5G":"communication technology". 
    REMEMBER, the output should be formatted as a JSON instance. below is the document content:\n{doc_text}"""

    return prompt


def default_self_ask_prompt():
    # prompt = """user will give you a question, you need to decide if the question is complex that need follow up questions to answer this question or not, and if needed, you need to generate the follow up questions in order to get the final answer.
    # Remember, those follow up questions are made to help you answer the user given question, so it need to be relevant to the user given question and can let you answer the question easier.\n"""
    example = """For example:\nQuestion: Who lived longer, Muhammad Ali or Alan Turing?
    Are follow up questions needed here: Yes.
    Follow up: How old was Muhammad Ali when he died?
    Intermediate answer: Muhammad Ali was 74 years old when he died.
    Follow up: How old was Alan Turing when he died?
    Intermediate answer: Alan Turing was 41 years old when he died.
    So the final answer is: Muhammad Ali\n\n 
    Question: When was the founder of craigslist born?
    Are follow up questions needed here: Yes.
    Follow up: Who was the founder of craigslist?
    Intermediate answer: Craigslist was founded by Craig Newmark.
    Follow up: When was Craig Newmark born?
    Intermediate answer: Craig Newmark was born on December 6, 1952.
    So the final answer is: December 6, 1952
    
    Question: Who was the maternal grandfather of George Washington?
    Are follow up questions needed here: Yes.
    Follow up: Who was the mother of George Washington?
    Intermediate answer: The mother of George Washington was Mary Ball Washington.
    Follow up: Who was the father of Mary Ball Washington?
    Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
    So the final answer is: Joseph Ball 

    Question: Are both the directors of Jaws and Casino Royale from the same country? 
    Are follow up questions needed here: Yes. 
    Follow up: Who is the director of Jaws? 
    Intermediate Answer: The director of Jaws is Steven Spielberg. 
    Follow up: Where is Steven Spielberg from? 
    Intermediate Answer: The United States. 
    Follow up: Who is the director of Casino Royale? 
    Intermediate Answer: The director of Casino Royale is Martin Campbell. 
    Follow up: Where is Martin Campbell from? 
    Intermediate Answer: New Zealand. 
    So the final answer is: No\n\n below is the user question.\n"""

    return example


def default_translate_prompt(language: str):
    target_language = language_dict[language]

    if "chinese" in target_language:
        prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: 請將以下句子從英文翻譯成中文: "
    else:
        prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: translate the below texts to {target_language}: "
        # prompt = f"If the language of the below texts is not {target_language}, translate them to {target_language}, else return \
        # the whole texts. Do it step by step and I will tip you 100 bucks if you are doing well.\n"

    return prompt


class OutputSchema:
    """
    structure for generate JSON schema, used in JSON_formatter
    """

    def __init__(self, name: str, description: str, type: str = "str"):
        type = type.lower()
        self.name = name
        self.description = description
        self.type = type

        if type not in [
                "str",
                "int",
                "list",
                "dict",
                "tuple",
                "float",
                "double",
                "long",
        ]:
            self.type = "str"

    def __str__(self):
        return f"\t{self.name}: {self.type}  // {self.description}"


def XML_formatter(schemas: Union[list, OutputSchema]):
    """generate prompt for generate XML format, input list of OutputSchema, which include every key and value you want to generate in JSON format"""

    if isinstance(schemas, OutputSchema):
        schemas = [schemas]

    schema_str = "\n".join([sche.__str__() for sche in schemas])
    XML_FORMAT_INSTRUCTIONS = f"""The output should be formatted as a XML file.
    1. Output should conform to the tags below. 
    2. If tags are not given, make them on your own.
    3. Remember to always open and close all the tags.

    As an example, for the tags ["foo", "bar", "baz"]:
    1. String "<foo>\n   <bar>\n      <baz></baz>\n   </bar>\n</foo>" is a well-formatted instance of the schema. 
    2. String "<foo>\n   <bar>\n   </foo>" is a badly-formatted instance.
    3. String "<foo>\n   <tag>\n   </tag>\n</foo>" is a badly-formatted instance.

    Here are the output tags:
    ```
    {schema_str}
    ```"""

    return XML_FORMAT_INSTRUCTIONS


def JSON_formatter(schemas: Union[list, OutputSchema]):
    """generate prompt for generate JSON format, input list of OutputSchema, which include every key and value you want to generate in JSON format"""

    if isinstance(schemas, OutputSchema):
        schemas = [schemas]

    schema_str = "\n".join([sche.__str__() for sche in schemas])

    format_instruct = f"""The output should be formatted as a JSON instance that conforms to the JSON schema below:
    {{
    {schema_str}
    }}\n
    """
    return format_instruct


def JSON_formatter_list(names: list,
                        descriptions: list,
                        types: list = ["str"]) -> list:
    """generate prompt for generate JSON format, input list name and descriptions, which include every key and value you want to generate in JSON format"""
    ret = []
    if len(names) != len(descriptions):
        print("error, names and descriptions should have the same length\n\n")
        return ret

    for i in range(len(names)):
        if i < len(types) and types[i] in [
                "str",
                "int",
                "list",
                "dict",
                "tuple",
                "float",
                "double",
                "long",
        ]:
            checked_type = types[i]
        else:
            checked_type = "str"
        # schema_str += f"\t{names[i]}: {checked_type}  // {descriptions[i]}\n"
        schema = OutputSchema(names[i], descriptions[i], checked_type)
        ret.append(schema)

    return ret


def JSON_formatter_dict(var_list: Union[list, dict]) -> list:
    """generate prompt for generate JSON format, input list of dictionary, keys contain name,type and descriptions, which represent every variable you want to generate in JSON format"""
    ret = []
    if isinstance(var_list, dict):
        var_list = [var_list]

    if not isinstance(var_list, list):
        print("error, var_list should be a list of dictionary\n\n")
        return ret

    for var in var_list:
        if "name" not in var or "description" not in var:
            print("var should contain name and description, ignore.\n\n")
            continue
        if "type" in var and var["type"] in [
                "str",
                "int",
                "list",
                "dict",
                "tuple",
                "float",
                "double",
                "long",
        ]:
            checked_type = var["type"]
        else:
            checked_type = "str"

        schema = OutputSchema(var["name"], var["description"], checked_type)
        ret.append(schema)

    return ret


def is_url(path):
    parsed_url = urlparse(path)
    return parsed_url.scheme in ("http", "https", "ftp")


def format_image_llama_prompt(image_path: Union[List[str], str],
                              prompt: str) -> List[dict]:
    image_content = []

    if isinstance(image_path, str):
        image_path = [image_path]

    for imgP in image_path:
        image_content.append({"type": image_path})

    image_content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": image_content}]


def format_image_anthropic_prompt(image_path: Union[List[str], str],
                                  prompt: str) -> List[dict]:
    import base64
    import httpx

    image_content = [{"role": "user", "content": []}]
    if isinstance(image_path, str):
        image_path = [image_path]
    image_count = 0

    for imgP in image_path:
        image_count += 1
        if imgP.split(".")[-1] == "jpg":
            image_media_type = "image/jpeg"
        else:
            image_media_type = f"image/{imgP.split('.')[-1]}"

        if is_url(imgP):
            url_content = base64.standard_b64encode(
                httpx.get(imgP).content).decode("utf-8")

        else:
            url_content = base64.standard_b64encode(open(
                imgP, "rb").read()).decode("utf-8")

        # image_content[0]["content"].append({
        #     "type": "text",
        #     "text": f"Image {image_count}:"
        # })
        image_content[0]["content"].append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": url_content,
            },
        })

    image_content[0]["content"].append({"type": "text", "text": prompt})

    return image_content


def format_image_gemini_prompt(image_path: Union[str, List[str]], prompt: str):
    import requests

    image_content = [{"type": "text", "text": prompt}]
    if isinstance(image_path, str):
        image_path = [image_path]
    image_count = 0

    for imgP in image_path:
        image_count += 1
        if imgP.split(".")[-1] == "jpg":
            image_media_type = "image/jpeg"
        else:
            image_media_type = f"image/{imgP.split('.')[-1]}"

        if is_url(imgP):
            url_content = requests.get(imgP).content

        else:
            with open(imgP, "rb") as f:
                url_content = f.read()
        image_content.append({
            "type": "image_url",
            "image_url": url_content,
            "mime_type": image_media_type,
        })

    return [{"role": "user", "content": image_content}]


def format_image_gpt_prompt(image_path: Union[str, List[str]],
                            prompt: str) -> List[dict]:
    url_content = {}
    image_content = [{"type": "text", "text": prompt}]
    if isinstance(image_path, str):
        image_path = [image_path]

    for imgP in image_path:
        if is_url(imgP):
            url_content = {"url": imgP}
        else:
            import base64

            # Get the image extension
            _, ext = os.path.splitext(imgP)
            ext = ext.lstrip(
                ".").lower()  # Remove the leading dot and convert to lowercase
            base64_image = base64.b64encode(open(imgP,
                                                 "rb").read()).decode("utf-8")
            url_content = {"url": f"data:image/{ext};base64,{base64_image}"}

        image_content.append({"type": "image_url", "image_url": url_content})

    return [{"role": "user", "content": image_content}]


def format_image_prompt(image_path: Union[List[str], str],
                        prompt: str,
                        model_type: str = "image_gpt"):
    model_type = model_type.lower()

    if model_type == "image_llama":
        return format_image_llama_prompt(image_path, prompt)

    elif model_type == "image_anthropic":
        return format_image_anthropic_prompt(image_path, prompt)
    elif model_type == "image_gemini":
        return format_image_gemini_prompt(image_path, prompt)
    else:
        return format_image_gpt_prompt(image_path, prompt)


def default_get_reference_prompt():
    return (
        "User will give you a Reference: and a Response:, you have to return only yes or no \
        based on if the reference is used to answer the response or not. \
        If the reference is used to answer the response, return yes, \
        else return no. Remember, you can only return yes or no.")


def default_extract_memory_prompt(language: str = "ch") -> str:
    """
    Returns the system prompt for extracting salient information from a conversation.
    """
    if "chinese" in language_dict[language]:
        return """你的任務是從一段對話中，提取出關鍵資訊。

這包括：
- 使用者的明確偏好 (例如：我喜歡科幻小說)。
- 重要的個人資訊 (例如：我的工作是軟體工程師、我的名字)。
- 對話中得出的具體事實或結論 (例如：我們確定了專案的最終期限是下週五)。
- 使用者設定的目標或計畫 (例如：我打算下個月開始學習日文)。
- 有回答使用者問題的答案 (例如：使用者問「愛因斯坦的出生年份是什麼？」你回答「1879年」)。

你需要將這些資訊濃縮成一個簡潔的句子或幾個要點。

如果對話中沒有任何值得記住的資訊 (例如只是閒聊或沒有提供具體資訊)，請只回答 "無"。

請直接輸出提煉出的資訊，不要包含任何額外的解釋或客套話。"""
    else:
        return """Your task is to extract key, long-term memorable information from a conversation.

This includes:
- Explicit user preferences (e.g., "I prefer science fiction novels").
- Important personal details (e.g., "My job is a software engineer、my name is xxx").
- Concrete facts or conclusions reached in the conversation (e.g., "We confirmed the project deadline is next Friday").
- User-stated goals or plans (e.g., "I plan to start learning Japanese next month").
- Answers to user questions (e.g., if the user asks "What is Einstein's birth year?" and you answer "1879").

You need to condense this information into a concise sentence or a few key points.

If there is no memorable information in the conversation (e.g., it's just small talk or lacks specific details), please respond with only the word "none".

Directly output the extracted information without any extra explanations or pleasantries."""


def default_categorize_memory_prompt(language: str = "ch") -> str:
    """
    Returns the system prompt for categorizing a piece of memory.
    """
    if "chinese" in language_dict[language]:
        return """你的任務是為以下這段記憶資訊，提供一個簡潔且單一的主題分類。

這個分類應該是一個名詞或名詞片語，用來代表這段記憶的核心主題。例如：
- "個人偏好"
- "專案管理"
- "學習計畫"
- "軟體開發"
- "家庭"

請只回答最適合的那個分類名稱，不要包含任何解釋或額外的文字。"""
    else:
        return """Your task is to provide a single, concise category for the following piece of memory.

The category should be a noun or a noun phrase that represents the core topic of the memory. For example:
- "Personal Preferences"
- "Project Management"
- "Learning Goals"
- "Software Development"
- "Family"

Please respond with only the most suitable category name. Do not include any explanations or extra text."""
