from langchain.text_splitter import RecursiveCharacterTextSplitter
import akasha
from pathlib import Path 
import time

def summarize_file(file_path:str, model:str = "openai:gpt-3.5-turbo", chunk_size:int = 1000, chunk_overlap:int = 40, verbose:bool = False\
    ,language:str='ch',summary_type = "map_reduce", max_token:int = 3000, summary_len:int = 500, record_exp:str="")->str:
    """input a file path and return a summary of the file

    Args:
        **file_path (str)**:  the path of file you want to summarize, can be '.txt', '.docx', '.pdf' file.\n
        **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
        **chunk_size (int, optional)**: the text length to split document to multiple chunks. Defaults to 1000.\n
        **chunk_overlap (int, optional)**: the overlap texts of neighboring chunks . Defaults to 40.\n
        **verbose (bool)**: show log texts or not. Defaults to False.\n
        **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
            max token size of llm input.\n
        **summary_type (str, optional)**: summary method, "map_reduce" or "refine". Defaults to "map_reduce".\n
        **max_token (int, optional)**: the max tokens that input to llm model each time. Defaults to 3000.\n
        **summary_len (int, optional)**: _description_. Defaults to 500.\n
        **record_exp (str, optional)**: _description_. Defaults to "".\n

    Returns:
        str: _description_
    """
    ### setting variables ###
    start_time = time.time()
    summary_type = summary_type.lower()
    logs = ["\n\n---------------summarize_file------------------------\n"]
    params = akasha.format.handle_params(model, "", chunk_size, "", -1, -1.0, language, False)
    params['chunk_overlap'] = chunk_overlap
    params['summary_type'] = "refine" if summary_type == "refine" else "map_reduce"
    

    
    ### check if doc path exist ###
    if not akasha.helper.is_path_exist(file_path, logs):
        print("file path not exist\n\n")
        return ""
    
    
    # Split the documents into sentences
    documents = akasha.helper._load_file(file_path, file_path.split('.')[-1])
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n'," ", ",",".","。","!" ], chunk_size=chunk_size, chunk_overlap = chunk_overlap)
    docs = text_splitter.split_documents(documents)
    doc_length = akasha.helper.get_docs_length(language, docs)
    texts = [doc.page_content for doc in docs]
    model = akasha.helper.handle_model(model, logs, verbose)


    if summary_type == "refine":
        response_list, tokens = _refine_summary(model, verbose, texts, max_token, summary_len)
        
    else:
        response_list, tokens =  _reduce_summary(texts, model, max_token, summary_len, verbose, 0, [])
    
    summaries = response_list[-1]
    p = akasha.prompts.format_refine_summary_prompt("","", summary_len)

    logs.extend(response_list)
    ### write summary to file, and if language is chinese , translate it ###
    if language == 'ch':
        try:    ### try call openai llm model 
            response = model.predict("translate the following text into chinese: \n\n" + summaries)
            
        except:
            response = model._call("translate the following text into chinese: \n\n" + summaries)
        summaries = akasha.helper.sim_to_trad(response)
    logs.append(summaries)
    
    ### write summary to file ###
    sum_path = Path("summarization/")
    if not sum_path.exists():
        sum_path.mkdir()
    
    file_name = "summarization/" + file_path.split('/')[-1].split('.')[-2] +".txt"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(summaries)
    
    print(summaries,"\n\n\n\n")
    
    
    
    
    
    end_time = time.time()    
    if record_exp != "":    
        metrics = akasha.format.handle_metrics(doc_length, end_time - start_time, tokens)
        table = akasha.format.handle_table(p, response_list, summaries)
        akasha.aiido_upload(record_exp, params, metrics, table, file_path)
    print("summarization saved in ", file_name,"\n\n")
    akasha.helper.save_logs(logs)
    return summaries







def _reduce_summary(texts:list, model, max_token:int ,summary_len:int ,verbose:bool, tokens:int , total_list:list):
    """Summarize each chunk and merge them until the combined chunks are smaller than the maximum token limit. 
    Then, generate the final summary. This method is faster and requires fewer tokens than the refine method.

    Args:
        **texts (list)**: list of texts from documents\n
        **model (var)**: llm model\n
        **max_token (int, optional)**: the max tokens that input to llm model each time. Defaults to 3000.\n
        **summary_len (int, optional)**: the desired word length for the final summary you want llm to generate. Defaults to 500.\n
        **verbose (bool)**: show log texts or not. Defaults to False.\n
        **tokens (int)**: used to save total tokens in recursive call.\n
        **total_list (list)**: used to save total response in recursive call.\n

    Returns:
         (list,int): llm response list and total tokens
    """
    response_list = []
    i  = 0
    while i < len(texts):
        
        token, cur_text, newi = akasha.helper._get_text(texts, "", i, max_token, model)
        tokens += token
        
        ### do the final summary if all chunks can be fits into llm model ###
        if i==0 and newi==len(texts):
            prompt = akasha.prompts.format_reduce_summary_prompt(cur_text, summary_len)
            
            try:    ### try call openai llm model
                response = model.predict(prompt)
            
            except:
                response = model._call(prompt)
            
            total_list.append(response)
            
            if verbose:
                print("prompt: \n", prompt)
                print("\n\n\n\n\n\n")
                print("response: \n", response)
                print("\n\n\n\n\n\n")
            
            return total_list, tokens
        
        prompt = akasha.prompts.format_reduce_summary_prompt(cur_text, 0)
        
        try:    ### try call openai llm model
            response = model.predict(prompt)
        
        except:
            response = model._call(prompt)
            
        
        i = newi  
        if verbose:  
            print("prompt: \n", prompt)
            print("\n\n\n\n\n\n")
            print("response: \n", response)
            print("\n\n\n\n\n\n")
        response_list.append(response)       
        total_list.append(response)
    return _reduce_summary(response_list, model, max_token, summary_len, verbose, tokens, total_list)

        
        
        
        
        
def _refine_summary(model, verbose, texts:list, max_token:int = 3000, summary_len:int = 500)->(list,int):
    """refine summary summarizing a chunk at a time and using the previous summary as a prompt for 
    summarizing the next chunk. This approach may be slower and require more tokens, but it results in a higher level of summary consistency.

    Args:
        **model (var)**: llm model\n
        **verbose (bool)**: show log texts or not. Defaults to False.\n
        **texts (list)**: list of texts from documents\n
        **max_token (int, optional)**: the max tokens that input to llm model each time. Defaults to 3000.\n
        **summary_len (int, optional)**: the desired word length for the final summary you want llm to generate. Defaults to 500.\n

    Returns:
        (list,int): llm response list and total tokens
    """
    ### setting variables ###
    previous_summary = ""
    i = 0
    tokens = 0
    response_list = []
    ###
    
    
    
    while i < len(texts):
        token, cur_text, i = akasha.helper._get_text(texts, previous_summary, i, max_token, model)
        
        tokens += token
        if previous_summary== "":
            prompt = akasha.prompts.format_reduce_summary_prompt(cur_text, summary_len)
        else:
            prompt = akasha.prompts.format_refine_summary_prompt(cur_text, previous_summary, summary_len)
        
        
        try:    ### try call openai llm model 
            response = model.predict(prompt)
            
        except:
            response = model._call(prompt)
            
        if verbose:
            print(prompt)
            print("\n\n\n\n\n\n")
            print(response)
            print("\n\n\n\n\n\n")
        response_list.append(response)       
        previous_summary = response
        
    return response_list, tokens