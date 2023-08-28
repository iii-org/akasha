


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


def format_chinese_json(query:str):
    """system prompt for single choice question in chinese

    Args:
        query (str): question string

    Returns:
        str: combined prompt
    """
    sys_prompt = "### 指令: 我会给出一个问题和几个可能的选项，请只根据提供的文件找到其中正确的一个答案，"+\
    "並回答答案為第幾個選項。若沒有提供，請照你的知識回答，并将答案以JSON的格式表示，如答案為第一個選項，"+\
    "回答的格式為{'ans':1}，不要添加其他字。  ### 问题和选项:\n"
    #ts_prompt = "### Instruction: I will provide a question and several possible options in the input. Please find the correct answer based solely on the provided texts, and respond with the number of the option that is the correct answer. If no texts is provided, please respond based on your knowledge, and format the answer in JSON format. For example, if the answer is the first option, the format of the response should be {'Answer': 1}. Please do not add any additional words. ### Input:"
    #sys_prompt = ts_prompt + query "  ### Response:"
    return sys_prompt + query




def format_create_question_prompt(doc_text:str)->str:
    """prompts for auto generate question from document

    Args:
        doc_text (str): texts from documents

    Returns:
        str: _description_
    """
    #q_prompt = "Human: You are a teacher coming up with questions to ask on a quiz. \nGiven the following document, please generate a question and answer based on that document.\n\nExample Format:\n<Begin Document>\n...\n<End Document>\nQUESTION: question here\nANSWER: answer here\n\nThese questions should be detailed and be based explicitly on information in the document. Begin!\n\n<Begin Document>\n\n"
    q_prompt =  "[INST] <<SYS>>" + "人類：您是一位教師，正在為測驗準備問題。\n根據以下文件，請基於該文件只生成一個問題和一個答案，問題應該詳細並且明確基於文件中的訊息。\n\n示例格式：\n<開始文件>\n...\n<結束文件>\n問題：問題在這里\n答案：答案在這里\n\n。開始吧！"+ "<<SYS>> [/INST]\n\n<開始文件>\n"
    #end_prompt = "<End Document>\n"
    end_prompt = "<結束文件>\n"
    # generate question prompt = generate_question_prompt(Document)
    q_prompt = q_prompt + doc_text + end_prompt
    
    return q_prompt