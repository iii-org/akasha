import time
import datetime
import numpy as np
import jieba
import json
from typing import Union, List
from tqdm import tqdm
from pathlib import Path
import opencc
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter,  RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from akasha.models.hf import chatGLM, get_hf_model
from akasha.models.llama2 import peft_Llama2, get_llama_cpp_model
import os
jieba.setLogLevel(jieba.logging.INFO)  ## ignore logging jieba model information




def is_path_exist(path:str)->bool:
    
    try:
        des_path = Path(path)
        if not des_path.exists():
            raise FileNotFoundError
    except FileNotFoundError as err:
        
        print(path, err)
        return False
    return True



def _load_file(file_path, extension):
    try:
        if extension == "pdf" or  extension == "PDF":
            docs = PyPDFLoader(file_path).load()
        elif extension == "docx" or extension == "DOCX":
            docs = Docx2txtLoader(file_path).load()
            for i in range(len(docs)):
                docs[i].metadata['page'] = i
           
        elif extension == "csv":
            docs = CSVLoader(file_path).load()
            for i in range(len(docs)):
                docs[i].metadata['page'] = docs[i].metadata['row']
                del docs[i].metadata['row']
            
        else:
            docs = TextLoader(file_path,encoding='utf-8').load()
            for i in range(len(docs)):
                docs[i].metadata['page'] = i
        return docs
    except:
        print("Load",file_path,"failed, ignored.\n")
        return ""

def _load_files(doc_path:str, extension:str="pdf", vis:set = set())->list:
    """load text files of select extension into list of Documents

    Args:
        **doc_path (str)**: text files directory\n 
        **extension (str, optional):** the extension type. Defaults to "pdf".\n 

    Returns:
        list: list of Documents
    """
    res = []
    dir = Path(doc_path)
    pdf_files = dir.glob("*."+extension)
    loaders = [file.name for file in pdf_files]


    for loader in loaders:
    
        temp = _load_file(doc_path+loader, extension)
        if temp != "":
            if temp[0].metadata['source'] not in vis:
                res.extend(temp)
                
    return res
       



def _check_dir_exists(doc_path:str, embeddings_name:str, chunk_size:int)->bool:
    """create 'chromadb' directory if not exist, and check if the doc db storage exist
    or not. If exist, exit create_chromadb function. 

    Args:
        **doc_path (str)**: the path of documents directory, also used to check if it's in chromabd storage. \n 

    Returns:
        bool: return True of False if the doc db storage exist
    """

    chroma_path = Path('chromadb')
    if not chroma_path.exists():
        chroma_path.mkdir()
    
    

    suffix_path = doc_path.split('/')[-2]
    embed_type,embed_name = embeddings_name.split(':')
    chroma_docs_path = Path('chromadb/'+ suffix_path + '_' + embed_type + '_' + embed_name.replace('/','-') +'_' + str(chunk_size) ) 

    if chroma_docs_path.exists():
        return True




def _separate_name(name:str):
    """ separate type:name by ':'

    Args:
        **name (str)**: string with format "type:name" \n 

    Returns:
        (str, str): res_type , res_name
    """
    sep = name.split(':')
    if len(sep) != 2:
        ### if the format type not equal to type:name ###
        res_type = sep[0].lower()
        res_name = ''
    else:
        res_type = sep[0].lower()
        res_name = sep[1]

    return res_type, res_name



def processMultiDB(doc_path_list: Union[List[str], str], verbose:bool, embeddings:vars, embeddings_name:str, chunk_size:int):
    
    ## if doc_path_list is a str, juest call create_chromadb function ##
    if isinstance(doc_path_list, str):
        return create_chromadb(doc_path_list, verbose, embeddings, embeddings_name, chunk_size)
    
    if len(doc_path_list) == 0:
        return None
    
    ## if using rerank, extend all the documents into one list ## 
    if isinstance(embeddings, str):
        texts = []
        for doc_path in doc_path_list:
            temp = create_chromadb(doc_path, verbose, embeddings, embeddings_name, chunk_size)
            if temp is not None:
                texts.extend(temp)
        if len(texts) == 0:
            return None
        return texts

    
    ## if not using rerank, create chromadb for each doc_path and merge them ##
    dbs = Chroma(embedding_function=embeddings)
    for doc_path in doc_path_list:
    
       
        db2 = create_chromadb(doc_path, verbose, embeddings, embeddings_name, chunk_size)

        
        if db2 is None:
            continue
        else:
            db2_data=db2.get(include=['documents','metadatas','embeddings'])
            dbs._collection.add(
                embeddings=db2_data['embeddings'],
                metadatas=db2_data['metadatas'],
                documents=db2_data['documents'],
                ids=db2_data['ids']
            )
        
    ## check if dbs has any document ##
    temp_docs = dbs.get(include=['documents'])['documents'] 
    if len(temp_docs) == 0:

        return None
    
    
    #print("create dbs",dbs.get(include=['metadatas'])['metadatas'])
    return dbs


def create_chromadb(doc_path:str, verbose:bool, embeddings:vars, embeddings_name:str,chunk_size:int,\
                    sleep_time:int = 60) -> vars:
    """If the documents vector storage not exist, create chromadb based on all .pdf files in doc_path.
        It will create a directory chromadb/ and save documents db in chromadb/{doc directory name}

    Args:
        **doc_path (str)**: the path of directory that store all .pdf documents\n 
        **logs (list)**: list that store logs\n 
        **verbose (bool)**: print logs or not\n 
        **embeddings (vars)**: the embeddings used in transfer documents into vector storage, could be openai 
        tensorflow or huggingface embeddings. \n 
        **sleep_time (int, optional)**: sleep time to transfer documents into vector storage, this is for 
            preventing rate limit exceed when request too much tokens at a time. Defaults to 60.\n 

    Raises:
        FileNotFoundError: if can not found the doc_path directory, will raise error and return None.

    Returns:
        vars: return the created chroma client. If documents vector storage already exist, load the vector storage
        and return.
    """
    

    ### check if doc path exist ###
    if not is_path_exist(doc_path):
        
        return None
    
    
    
    if doc_path[-1] != '/':
        doc_path += '/'
    embed_type, embed_name = _separate_name(embeddings_name)
    storage_directory = 'chromadb/' + doc_path.split('/')[-2] + '_' + embed_type + '_' + embed_name.replace('/','-') + '_' + str(chunk_size)
    db_exi = _check_dir_exists(doc_path, embeddings_name, chunk_size)
    vis = set()
    if isinstance(embeddings, str):
        # Split the documents into sentences
        documents = []
        txt_extensions = ['pdf', 'md','docx','txt','csv']
        for extension in txt_extensions:
            documents.extend(_load_files(doc_path, extension, vis))
        if len(documents) == 0 :
            return None
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n'," ", ",",".","。","!" ], chunk_size=chunk_size, chunk_overlap = 40)
        docs = text_splitter.split_documents(documents)
        texts = [doc for doc in docs]
        if len(texts) == 0:
            return None
        return texts
        
    elif db_exi:
        info = "storage db already exist.\n"
        db = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
        mtda = db.get(include=['metadatas'])
        
        for source in mtda['metadatas']:
            vis.add(source['source'])
        del db,mtda
      

    
        

    documents = []
    txt_extensions = ['pdf', 'md','docx','txt','csv']
    for extension in txt_extensions:
        documents.extend(_load_files(doc_path, extension, vis))
    
    
    
    if len(documents) == 0 and not db_exi:
        return None


    if len(documents) != 0 :
        info = "\n\nload files:" +  str(len(documents)) + "\n\n" 
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n'," ", ",",".","。","!" ], chunk_size=chunk_size, chunk_overlap=40)
        k = 0
        cum_ids = 0
        interval = 3
        progress = tqdm(total = len(documents), desc="Vec Storage")
        while k < len(documents):
            
            
            progress.update(min(interval,len(documents)-k))
            cur_doc = documents[k:k+interval]
            texts = text_splitter.split_documents(cur_doc)

            try :
                if k==0 :
                    if db_exi:
                        docsearch = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
                        if len(texts) != 0:
                            docsearch.add_documents(texts)
                    else:
                        if len(texts) != 0:
                            docsearch = Chroma.from_documents(texts, embeddings, persist_directory = storage_directory) 
                        else:
                            docsearch = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
                    
                else:
                    if len(texts) != 0:
                        docsearch.add_documents(texts)
            except:
                time.sleep( sleep_time )
                if k==0:
                    if db_exi:
                        docsearch = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
                        if len(texts) != 0:
                            docsearch.add_documents(texts)
                    else:
                        if len(texts) != 0:
                            docsearch = Chroma.from_documents(texts, embeddings, persist_directory = storage_directory) 
                        else:
                            docsearch = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
                    
                    
                else:
                    if len(texts) != 0:
                        docsearch.add_documents(texts)
                    
            k += interval
            cum_ids += len(texts)
            
            
        docsearch.persist()  
        progress.close()
    
    db = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
    temp_docs = db.get(include=['documents'])['documents'] 
    if len(temp_docs) == 0:

        return None
    
    if verbose:
        print(info)
    
    
    return db




def handle_embeddings(embedding_name:str, verbose:bool)->vars :
    """create model client used in document QA, default if openai "gpt-3.5-turbo"
        use openai:text-embedding-ada-002 as default.
    Args:
        **embedding_name (str)**: embeddings client you want to use.
            format is (type:name), which is the model type and model name.\n
            for example, "openai:text-embedding-ada-002", "huggingface:all-MiniLM-L6-v2".\n
        **logs (list)**: list that store logs\n
        **verbose (bool)**: print logs or not\n

    Returns:
        vars: embeddings client
    """
    
    embedding_type, embedding_name = _separate_name(embedding_name)

    if embedding_type in ["text-embedding-ada-002" , "openai" , "openaiembeddings"]:
        import openai
        if "OPENAI_API_BASE" in os.environ:
            embedding_name = embedding_name.replace(".","")
            embeddings = OpenAIEmbeddings(deployment=embedding_name)
        else:
            openai.api_type = "open_ai"
            embeddings = OpenAIEmbeddings(deployment=embedding_name,model=embedding_name)
        info = "selected openai embeddings.\n"

    elif embedding_type in ["rerank", "re"]:
        if embedding_name == "":
            embedding_name = "BAAI/bge-reranker-base"
        
        embeddings = "rerank:" + embedding_name
        info = "selected rerank embeddings.\n"
    elif embedding_type in ["huggingface" , "huggingfaceembeddings","transformers", "transformer", "hf"]:
        from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name = embedding_name)
        info = "selected hugging face embeddings.\n"


    elif embedding_type in ["tf", "tensorflow", "tensorflowhub", "tensorflowhubembeddings", "tensorflowembeddings"]:
        from langchain.embeddings import TensorflowHubEmbeddings
        embeddings = TensorflowHubEmbeddings()
        info = "selected tensorflow embeddings.\n"

    else:
        embeddings = OpenAIEmbeddings()
        info = "can not find the embeddings, use openai as default.\n"
    
    if verbose:
        print(info)
    return embeddings




def handle_model(model_name:str, verbose:bool, temperature:float = 0.0)->vars:
    """create model client used in document QA, default if openai "gpt-3.5-turbo"

    Args:
       ** model_name (str)**: open ai model name like "gpt-3.5-turbo","text-davinci-003", "text-davinci-002"\n
        **logs (list)**: list that store logs\n
        **verbose (bool)**: print logs or not\n

    Returns:
        vars: model client
    """
    model_type, model_name = _separate_name(model_name)

    
    
    if model_type in ["openai" , "openaiembeddings"]:
        import openai
        if "OPENAI_API_BASE" in os.environ:
            model_name = model_name.replace(".","")
            model = AzureChatOpenAI(deployment_name=model_name, temperature=temperature)
        else:
            openai.api_type = "open_ai"
            model = ChatOpenAI(model=model_name, temperature = temperature)
        info = f"selected openai model {model_name}.\n"

    elif model_type in ["llama-cpu", "llama-gpu", "llama", "llama2", "llama-cpp"] and model_name != "":
        
        model = get_llama_cpp_model(model_type, model_name, temperature)
        info = "selected llama-cpp model\n"
    elif model_type in ["huggingface" , "huggingfacehub","transformers", "transformer", "huggingface-hub", "hf"]:
        model = get_hf_model(model_name, temperature)
        info = f"selected huggingface model {model_name}.\n"

    elif model_type in ["chatglm","chatglm2","glm"]:
        model = chatGLM(model_name=model_name, temperature=temperature)
        info = f"selected chatglm model {model_name}.\n"
    
    elif model_type in ["lora", "peft"]:
        model = peft_Llama2(model_name_or_path=model_name,temperature=temperature)
        info = f"selected peft model {model_name}.\n"
    else:
        model = ChatOpenAI(model = "gpt-3.5-turbo", temperature = temperature)
        info = f"can not find the model {model_type}:{model_name}, use openai as default.\n"
        print(info)
    if verbose:
        print(info)
    
    return model

def handle_search_type(search_type:str, verbose:bool)->str:
    if callable(search_type):
        search_type_str = search_type.__name__
        
    else:
        search_type_str = search_type

    # if verbose:
    #     print("search type is :", search_type_str)

    return search_type_str





def get_doc_length(language:str, doc)->int:
    """calculate the length of terms in a giving Document

    Args:
        **language (str)**: 'ch' for chinese and 'en' for others, default 'ch'\n
        **doc (Document)**: Document object\n

    Returns:
        doc_length: int Docuemtn length
    """
    if language=='ch':
        
        doc_length = len(list(jieba.cut(doc.page_content)))
    else:
        doc_length = len(doc.page_content.split())
    return doc_length

def get_docs_length(language:str, docs:list)->int:
    """calculate the total length of terms in giving documents

    Args:
        language (str): 'ch' for chinese and 'en' for others, default 'ch'\n
        docs (list): list of Documents\n

    Returns:
        docs_length: int total Document length
    """
    docs_length = 0
    for doc in docs:
        docs_length += get_doc_length(language, doc)
    return docs_length



def get_question_from_file(path:str, question_type:str):
    """load questions from file and save the questions into lists.
    a question list include the question, mutiple options, and the answer (the number of the option),
      and they are all separate by space in the file. 

    Args:
        **path (str)**: path of the question file\n

    Returns:
        list: list of question list
    """
    f_path = Path(path)
    with f_path.open(mode='r',encoding='utf-8') as file:
        content = file.read()
    questions = []
    answers = []    
 
        
    if question_type.lower() == "essay":
        content = content.split("\n\n")
        for i in range(len(content)):
            if content[i] == "":
                continue
            process = ''.join(content[i].split("問題：")).split("答案：")
            
            questions.append(process[0])
            answers.append(process[1])
        return questions, answers
    
    for con in content.split('\n'):
        if con=="":
            continue
        questions.append( [word for word in con.split("\t")if word != ""])
    return questions, answers

def extract_result(response:str):
    """to prevent the output of llm format is not what we want, try to extract the answer (digit) from the llm output 

    Args:
        **response (str)**: llm output\n

    Returns:
        int: digit of answer
    """
    try:
        res = str(json.loads(response)['ans']).replace(" ", "")

    except:
        res = -1
        for c in response:
            if c.isdigit():
                res = c

                break
    return res






def get_all_combine(embeddings_list:list, chunk_size_list:list, model_list:list, topK_list:list, search_type_list:list)->list:
    """record all combinations of giving lists

    Args:
        **embeddings_list (list)**: list of embeddings(str)\n
        **chunk_size_list (list)**: list of chunk sizes(int)\n
        **model_list (list)**: list of models(str)\n
        **topK_list (list)**: list of topK(int)\n
        **search_type_list (list)**: list of search types(str)\n

    Returns:
        list: list of tuples of all different combinations
    """
    res = []
    for embed in embeddings_list:
        for chk in chunk_size_list:
            for mod in model_list:
                for tK in topK_list:
                    for st in search_type_list:
                        res.append((embed, chk, mod, tK, st))

    return res


def get_best_combination(result_list:list, idx:int)->list:
    """input list of tuples and find the greatest tuple based on score or cost-effective (index 0 or index 1)
    tuple looks like (score, cost-effective, embeddings, chunk size, model, topK, search type)

    Args:
        **result_list (list)**: list of tuples that save the information of running experiments\n
        **idx (int)**: the index used to find the greatest result 0 is based on score and 1 is based on cost-effective\n

    Returns:
        list: return list of tuples that have same highest criteria
    """
    res = []
    sorted_res = sorted(result_list, key=lambda x:x[idx], reverse=True)
    max_score = sorted_res[0][idx]
    for tup in sorted_res:
        if tup[idx] < max_score:
            break
        res_str = "embeddings: " + tup[-5] + ", chunk size: " + str(tup[-4]) + ", model: " +tup[-3] + ", topK: " + str(tup[-2]) + ", search type: " + tup[-1] + "\n"
        print(res_str)
        res.append(tup[-5:])

    return res




def sim_to_trad(text:str)->str:
    """convert simplified chinese to traditional chinese

    Args:
        **text (str)**: simplified chinese\n

    Returns:
        str: traditional chinese
    """
    cc = opencc.OpenCC('s2t.json')
    return cc.convert(text)



def _get_text(texts:list,previous_summary:str, i:int, max_token:int,model)->(int, str, int):
    """used in summary, combine chunks of texts into one chunk that can fit into llm model

    Args:
        texts (list): chunks of texts
        previous_summary (str): _description_
        i (int): start from i-th chunk
        max_token (int): the max token we want to fit into llm model at one time
        model (var): llm model

    Returns:
        (int, str, int): return the total tokens of combined chunks, combined chunks of texts, and the index of next chunk
    """
    cur_count = model.get_num_tokens(previous_summary) 
    words_len = model.get_num_tokens(texts[i])
    cur_text = ""
    while cur_count+words_len < max_token and i < len(texts):
        cur_count += words_len
        cur_text += texts[i]  + "\n"
        i += 1
        if i < len(texts):
            words_len = model.get_num_tokens(texts[i])
    
    
    return cur_count, cur_text, i



def  call_model(model, prompt:str)->str:
    """call llm model and return the response

    Args:
        model (_type_): llm model
        prompt (str): input prompt

    Returns:
        str: llm response
    """
    try:    ### try call openai llm model 
        response = model.predict(prompt)
                
    except:
        response = model._call(prompt)
    return response



def get_non_repeat_rand_int(vis:set, num:int):
    
    temp = np.random.randint(num)
    if len(vis) >= num:
        vis = set()
    if temp not in vis:
        vis.add(temp)
        return temp
    return get_non_repeat_rand_int(vis, num)



def get_text_md5(text):
    import hashlib
    md5_hash = hashlib.md5(text.encode()).hexdigest()
    
    return md5_hash

