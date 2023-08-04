


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