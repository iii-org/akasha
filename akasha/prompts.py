


def format_question_query(question:list)->(str, str):
    """generate a certain format of question to input to llm. Last element means which selection is the correct answer.
       return the question query string and the answer string.
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
        question (list): list of question and selections and answer

    Returns:
        (str, str): return the question query string and the answer string
    """
    n = len(question)
    if  n == 0 :
        return "", ""
    elif n == 1:
        return "Question: "+ question[0], ""
    query = "Question: "+ question[0] + "\n"

    for i in range(1, n-1):
        query += str(i) + ". " + question[i] + "\n"

    return query, question[-1]



def format_llama_json(query):
    """insert system prompt for llm to generate JSON format of {"ans":selection number}

    Args:
        query (str): a question string with selections that we want llm to answer

    Returns:
       prompt (str): the whole prompt includes system prompt and question  
    """
    sys_b, sys_e = "<<SYS>>\n", "\n<</SYS>>\n\n"
    sys_prompt =  "[INST]" + sys_b +\
    " i will give you a question with several possible answer, use the content of documents "+\
    "to choose correct answer. and you need to return the answer of this question as" +\
    " JSON structure with key \"ans\", and only this JSON structure, please don't add any other word. for example, "+\
    "User: what is 1+1 euqals to? 1.「2」  2.「4」 3.「10」 4.「15」 \n you: {\"ans\":1}" + sys_e

    
    prompt =  query + "[/INST]"
    return sys_prompt + prompt


def format_chinese_json(query):

    sys_prompt = "### 指令: 我会给出一个问题和几个可能的选项，请只根据提供的文件找到其中正确的一个答案，"+\
    "並回答答案為第幾個選項。若沒有提供，請照你的知識回答，并将答案以JSON的格式表示，如答案為第一個選項，"+\
    "回答的格式為{'ans':1}，不要添加其他字。  ### 问题和选项:\n"

    return sys_prompt + query