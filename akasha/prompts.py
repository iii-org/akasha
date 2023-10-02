
sys_s = "[INST] <<SYS>> " 
sys_e = " <<SYS>> [/INST]\n\n"


def format_llama_sys_prompt(system_prompt:str, prompt:str):
    
    if system_prompt == "":
        return "[INST] " + prompt + " [/INST]\n"
    return "[INST] <<SYS>> " + system_prompt + " <<SYS>> \n\n " + prompt + " [/INST]\n"
    
def format_sys_prompt(system_prompt:str, prompt:str, model_type:str="llama"):
    
    if model_type == "llama":
        return format_llama_sys_prompt(system_prompt, prompt)
    
    return format_llama_sys_prompt(system_prompt, prompt)


def format_question_query(question:list)->(str, str):
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
        **question (list)**: list of question and selections and answer\n

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
        **query (str)**: a question string with selections that we want llm to answer\n

    Returns:
       prompt (str): the whole prompt includes system prompt and question  
    """
    
    sys_prompt =  "human will give you a question with several possible answer, use the content of documents "+\
    "to choose correct answer. and you need to return the answer of this question as" +\
    " JSON structure with key \"ans\", and only this JSON structure, please don't add any other word. for example, "+\
    "User: what is 1+1 euqals to? 1.「2」  2.「4」 3.「10」 4.「15」 \n you: {\"ans\":1}" 

    
    prompt =  sys_s + sys_prompt + sys_e
    return prompt + query


def format_chinese_json(query:str):
    """system prompt for single choice question in chinese

    Args:
        **query (str)**: question string\n

    Returns:
        str: combined prompt
    """
    sys_prompt = "### 指令: 我会给出一个问题和几个可能的选项，请只根据提供的文件找到其中正确的一个答案，"+\
    "並回答答案為第幾個選項。若沒有提供，請照你的知識回答，并将答案以JSON的格式表示，如答案為第一個選項，"+\
    "回答的格式為{'ans':1}，不要添加其他字。  ### 问题和选项:\n"
    #ts_prompt = "### Instruction: I will provide a question and several possible options in the input. Please find the correct answer based solely on the provided texts, and respond with the number of the option that is the correct answer. If no texts is provided, please respond based on your knowledge, and format the answer in JSON format. For example, if the answer is the first option, the format of the response should be {'Answer': 1}. Please do not add any additional words. ### Input:"
    #sys_prompt = ts_prompt + query "  ### Response:"
    return sys_prompt + query



def format_wrong_answer(num:int, doc_text:str,question:str, correct_ans:str)->str:
    """prompt for generate wrong answers to create single choice question

    Args:
        **num (int)**: number of wrong answers that we want to generate\n
        **doc_text (str)**: document texts that used to generate question\n
        **question (str)**: question string\n
        **correct_ans (str)**: correct answer string\n

    Returns:
        str: combined prompt
    """
    
    q_prompt =  sys_s + f"根據以下的文件、問題和正確答案，請基於文件、問題和正確答案生成{num}個錯誤答案，錯誤答案應該與正確答案有相關性但數字、內容或定義錯誤，或者與正確答案不相同但有合理性。並注意各個錯誤答案必須都不相同。\n\n示例格式：\n<開始文件>\n...\n<結束文件>\n<開始問題>\n...\n<結束問題>\n<開始正確答案>\n...\n<結束正確答案>\n\n錯誤答案：錯誤答案1在這里\n\n錯誤答案：錯誤答案2在這里\n\n錯誤答案：錯誤答案3在這里\n\n。開始吧！"+ sys_e +"<開始文件>\n"
    end_doc = "<結束文件>\n"
    st_q = "<開始問題>\n"
    end_q = "<結束問題>\n"
    st_cor = "<開始正確答案>\n"
    end_cor = "<結束正確答案>\n"
    q_prompt = q_prompt + doc_text + end_doc + st_q + question + end_q + st_cor + correct_ans + end_cor + "\n\n"
    
    return q_prompt
def format_create_question_prompt(doc_text:str,question_type:str)->str:
    """prompts for auto generate question from document

    Args:
        **doc_text (str)**: texts from documents\n
        **question_type (str)**: question type, can be "single choice", "essay"\n

    Returns:
        str: combined prompt
    """
    qt = ""
    if question_type != "essay":
        qt = "少於100字的"
    #q_prompt = "Human: You are a teacher coming up with questions to ask on a quiz. \nGiven the following document, please generate a question and answer based on that document.\n\nExample Format:\n<Begin Document>\n...\n<End Document>\nQUESTION: question here\nANSWER: answer here\n\nThese questions should be detailed and be based explicitly on information in the document. Begin!\n\n<Begin Document>\n\n"
    q_prompt =  sys_s + f"人類：您是一位教師，正在為測驗準備問題。\n請基於文件只生成一個問題和一個{qt}答案，問題應該詳細並且明確基於文件中的訊息。\n\n示例格式：\n<開始文件>\n...\n<結束文件>\n問題：問題在這里\n答案：答案在這里\n\n。開始吧！"+ sys_e +"<開始文件>\n"
    #end_prompt = "<End Document>\n"
    end_prompt = "<結束文件>\n"
    # generate question prompt = generate_question_prompt(Document)
    q_prompt = q_prompt + doc_text + end_prompt
    
    return q_prompt





def format_llm_score(cand:str,ref:str):
    """the system prompt for llm to calculate the cnadidate is correct or not.

    Args:
        **cand (str)**: llm generated response that we want to test the performance\n
        **ref (str)**: reference answer\n
    """
    
    sys_prompt = "human will give you a [candidate] sentence and a [reference] sentence, please score the [candidate] sentence "+\
        "based on the [reference] sentence, the higher score means the [candidate] sentence has enough information and correct answer that [reference] sentence has." +\
        "remember, you can only return the score and need to return the score of this [candidate] sentence as a float number range from 0 to 1.\n" +\
        "Example Format:\n Human: [candidate]: ...\n [reference]: ...\n\n You: 0.8\n\n"
    
    prompt =  sys_s + sys_prompt + sys_e
    return prompt + "[candidate]: " + cand + "\n[reference]: " + ref + "\n"




def format_reduce_summary_prompt(cur_text:str, summary_len:int = 500):
    """the prompt for llm to generate a summary of the given text

    Args:
        **cur_text (str)**: the text that we want llm to generate a summary\n
        **summary_len (int, optional)**: the summary word length we want llm to generate. Defaults to 500.\n

    Returns:
        str: summary prompt.
    """
 
    underline = "------------"
    if summary_len > 0:
        sys_prompt =f"Write a concise {summary_len} words summary of the following:\n" + underline + "\n" + cur_text + underline

    else:
        sys_prompt = f"Write a concise summary of the following:\n" + underline + "\n" + cur_text + underline
    
    return sys_prompt




def format_refine_summary_prompt(cur_text:str, previous_summary:str, summary_len:int = 500):
    """the prompt for llm to generate the summary of the given text and previous summary
    
     Args:
        **cur_text (str)**: the text that we want llm to generate a summary\n
        **previous_summary (str)**: the previous summary that we want llm to generate a summary\n
        **summary_len (int, optional)**: the summary word length we want llm to generate. Defaults to 500.\n

    Returns:
        str: summary prompt.
        
    """
    
    sys_prompt =f"""Your job is to produce a final summary of {summary_len} words.
    We have provided an existing summary up to a certain point, original summary is:  {previous_summary}
    ------------\n
    {cur_text}\n
    ------------\n
    Given the new context, refine the original summary.
    If the context isn't useful, return the original summary.
    """
    return sys_prompt



def format_compression_prompt(query:str, doc:str):
    return  f"""Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return an empty string. 

Remember, *DO NOT* edit the extracted parts of the context.
\nQuestion: {query}
\nContext:
{doc}"""