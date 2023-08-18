import time
import datetime
import os
import jieba
import json
from tqdm import tqdm
from pathlib import Path
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter,  RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from akasha.models.hf import chatGLM, get_hf_model
from akasha.models.llama2 import peft_Llama2, get_llama_cpp_model
import os
jieba.setLogLevel(jieba.logging.INFO)  ## ignore logging jieba model information


def _load_files(doc_path:str, extension="pdf")->list:

    res = []
    dir = Path(doc_path)
    pdf_files = dir.glob("*."+extension)
    loaders = [file.name for file in pdf_files]


    for loader in loaders:
    
        try:
            if extension == "pdf":
                res.extend(PyPDFLoader(doc_path+loader).load())
            elif extension == "docx":
                docs = Docx2txtLoader(doc_path+loader).load()
                for i in range(len(docs)):
                    docs[i].metadata['page'] = i
                res.extend(docs)
            else:
                docs = TextLoader(doc_path+loader,encoding='utf-8').load()
                for i in range(len(docs)):
                    docs[i].metadata['page'] = i
                res.extend(docs)
        except:
            print("Load",loader,"failed, ignored.\n")
    
    return res
       



def _check_dir_exists(doc_path:str, embeddings_name:str, chunk_size:int)->bool:
    """create 'chromadb' directory if not exist, and check if the doc db storage exist
    or not. If exist, exit create_chromadb function. 

    Args:
        doc_path (str): the path of documents directory, also used to check if it's in chromabd storage. 

    Returns:
        bool: return True of False if the doc db storage exist
    """

    chroma_path = Path('chromadb')
    if not chroma_path.exists():
        chroma_path.mkdir()
    
    

    suffix_path = doc_path.split('/')[-2]
    chroma_docs_path = Path('chromadb/'+ suffix_path + '_' + embeddings_name.split(':')[0] + '_' + str(chunk_size) ) 

    if chroma_docs_path.exists():
        return True




def _separate_name(name:str):
    """ separate type:name by ':'

    Args:
        name (str): string with format "type:name" 

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



def create_chromadb(doc_path:str, logs:list, verbose:bool, embeddings:vars, embeddings_name:str,chunk_size:int,\
                    sleep_time:int = 60) -> vars:
    """If the documents vector storage not exist, create chromadb based on all .pdf files in doc_path.
        It will create a directory chromadb/ and save documents db in chromadb/{doc directory name}

    Args:
        doc_path (str): the path of directory that store all .pdf documents
        logs (list): list that store logs
        verbose (bool): print logs or not
        embeddings (vars): the embeddings used in transfer documents into vector storage, could be openai 
        tensorflow or huggingface embeddings. 
        sleep_time (int, optional): sleep time to transfer documents into vector storage, this is for 
            preventing rate limit exceed when request too much tokens at a time. Defaults to 60.

    Raises:
        FileNotFoundError: if can not found the doc_path directory, will raise error and return None.

    Returns:
        vars: return the created chroma client. If documents vector storage already exist, load the vector storage
        and return.
    """
    

    ### check if doc path exist ###
    try:
        des_path = Path(doc_path)
        if not des_path.exists:
            raise FileNotFoundError
    except FileNotFoundError as err:
        if verbose:
            print(err)
        logs.append(err)
        return None
    ### ###


    
    if doc_path[-1] != '/':
        doc_path += '/'
    storage_directory = 'chromadb/' + doc_path.split('/')[-2] + '_' + embeddings_name.split(':')[0] + '_' + str(chunk_size)


    if _check_dir_exists(doc_path, embeddings_name, chunk_size):
        info = "storage db already exist.\n"
        


    else:
        

        documents = []
        txt_extensions = ['pdf', 'md','docx','csv','txt']
        for extension in txt_extensions:
            documents.extend(_load_files(doc_path, extension))
        
        info = "\n\nload files:" +  str(len(documents)) + "\n\n" 
        
        if len(documents) == 0 :
            return None


        
        #text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=0)
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n'," ", ",",".","。","!" ], chunk_size=chunk_size, chunk_overlap=40)
        k = 0
        cum_ids = 0
        interval = 5
        progress = tqdm(total = len(documents), desc="Vec Storage")
        while k < len(documents):
            
            
            progress.update(min(interval,len(documents)-k))
            cur_doc = documents[k:k+interval]
            texts = text_splitter.split_documents(cur_doc)
            try :
                if k==0:
                    docsearch = Chroma.from_documents(texts, embeddings, persist_directory = storage_directory) 
                    
                else:
                    docsearch.add_documents(texts)
            except:
                time.sleep( sleep_time )
                if k==0:
                    docsearch = Chroma.from_documents(texts, embeddings, persist_directory = storage_directory) 
                    
                else:
                    docsearch.add_documents(texts)
                    
            k += interval
            cum_ids += len(texts)
            
            
        docsearch.persist()  
        progress.close()
    
    db = Chroma(persist_directory=storage_directory, embedding_function=embeddings)
    if verbose:
        print(info)
    logs.append(info)
    
    return db




def handle_embeddings(embedding_name:str, logs:list, verbose:bool)->vars :
    """create model client used in document QA, default if openai "gpt-3.5-turbo"
        use openai:text-embedding-ada-002 as default.
    Args:
        embedding_name (str): embeddings client you want to use.
            format is (type:name), which is the model type and model name.
            for example, "openai:text-embedding-ada-002", "huggingface:all-MiniLM-L6-v2".
        logs (list): list that store logs
        verbose (bool): print logs or not

    Returns:
        vars: embeddings client
    """
    
    embedding_type, embedding_name = _separate_name(embedding_name)

    if embedding_type in ["text-embedding-ada-002" , "openai" , "openaiembeddings"]:
        embeddings = OpenAIEmbeddings(model = embedding_name)
        info = "selected openai embeddings.\n"

    
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
    logs.append(info)
    return embeddings




def handle_model(model_name:str, logs:list, verbose:bool)->vars:
    """create model client used in document QA, default if openai "gpt-3.5-turbo"

    Args:
        model_name (str): open ai model name like "gpt-3.5-turbo","text-davinci-003", "text-davinci-002"
        logs (list): list that store logs
        verbose (bool): print logs or not

    Returns:
        vars: model client
    """
    model_type, model_name = _separate_name(model_name)

    
    
    if model_type in ["openai" , "openaiembeddings"]:
        model = ChatOpenAI(model=model_name, temperature = 0)
        info = f"selected openai model {model_name}.\n"

    elif model_type in ["llama-cpu", "llama-gpu", "llama", "llama2", "llama-cpp"] and model_name != "":
        
        model = get_llama_cpp_model(model_type, model_name)
        info = "selected llama-cpp model\n"
    elif model_type in ["huggingface" , "huggingfacehub","transformers", "transformer", "huggingface-hub", "hf"]:
        model = get_hf_model(model_name)
        info = f"selected huggingface model {model_name}.\n"

    elif model_type in ["chatglm","chatglm2","glm"]:
        model = chatGLM(model_name=model_name)
        info = f"selected chatglm model {model_name}.\n"
    
    elif model_type in ["lora", "peft"]:
        model = peft_Llama2(model_name_or_path=model_name)
        info = f"selected peft model {model_name}.\n"
    else:
        model = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0)
        info = f"can not find the model {model_type}:{model_name}, use openai as default.\n"
        print(info)
    if verbose:
        print(info)
    logs.append(info)
    
    return model

def save_logs(logs:list)->None:
    """save running logs into logs/logs_{date}.txt

    Args:
        logs (list): list that store logs
    """
    logs = '\n'.join(logs)

    cur_date = datetime.datetime.now().strftime("%Y-%m-%d")

    logs_path = Path('logs')
    if not logs_path.exists():
        logs_path.mkdir()


    file_name = f"log_{cur_date}.txt"
    file_path = Path("logs/"+file_name)

    if file_path.exists():
        with file_path.open("a", encoding='utf-8') as file:
            file.write(logs + "\n\n")
    else:
        with file_path.open("w",  encoding='utf-8') as file:
            file.write(logs + "\n\n")
    
    return





def get_doc_length(language:str, doc)->int:
    """calculate the length of terms in a giving Document

    Args:
        language (str): 'ch' for chinese and 'en' for others, default 'ch'
        doc (Document): Docuemtn object

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
        language (str): 'ch' for chinese and 'en' for others, default 'ch'
        docs (list): list of Documents

    Returns:
        docs_length: int total Document length
    """
    docs_length = 0
    for doc in docs:
        docs_length += get_doc_length(language, doc)
    return docs_length



def get_question_from_file(path:str)->list:
    """load questions from file and save the questions into lists.
    a question list include the question, mutiple options, and the answer (the number of the option),
      and they are all separate by space in the file. 

    Args:
        path (str): path of the question file

    Returns:
        list: list of question list
    """
    f_path = Path(path)
    with f_path.open(mode='r',encoding='utf-8') as file:
        content = file.read()
    questions = []
    for con in content.split('\n'):
        questions.append( [word for word in con.split(" ")if word != ""])
    
    return questions


def extract_result(response:str):
    """to prevent the output of llm format is not what we want, try to extract the answer (digit) from the llm output 

    Args:
        response (str): llm output

    Returns:
        int: digit of answer
    """
    try:
        res = str(json.loads(response[-1])['ans']).replace(" ", "")

    except:
        res = -1
        for c in response[-1]:
            if c.isdigit():
                res = c

                break
    return res






def get_all_combine(embeddings_list:list, chunk_size_list:list, model_list:list, topK_list:list, search_type_list:list)->list:
    """record all combinations of giving lists

    Args:
        embeddings_list (list): list of embeddings(str)
        chunk_size_list (list): list of chunk sizes(int)
        model_list (list): list of models(str)
        topK_list (list): list of topK(int)
        search_type_list (list): list of search types(str)

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


def get_best_combination(result_list:list, idx:int, logs:list=[])->list:
    """input list of tuples and find the greatest tuple based on score or cost-effective (index 0 or index 1)
    tuple looks like (score, cost-effective, embeddings, chunk size, model, topK, search type)

    Args:
        result_list (list): list of tuples that save the information of running experiments
        idx (int): the index used to find the greatest result 0 is based on score and 1 is based on cost-effective

    Returns:
        list: return list of tuples that have same highest criteria
    """
    res = []
    sorted_res = sorted(result_list, key=lambda x:x[idx], reverse=True)
    max_score = sorted_res[0][idx]
    for tup in sorted_res:
        if tup[idx] < max_score:
            break
        res_str = "embeddings: " + tup[2] + ", chunk size: " + str(tup[3]) + ", model: " +tup[4] + ", topK: " + str(tup[5]) + ", search type: " + tup[6] + "\n"
        print(res_str)
        logs.append(res_str)
        res.append(tup[2:])

    return res