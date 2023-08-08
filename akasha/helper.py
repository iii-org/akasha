import time
import datetime
import os
import jieba
import json
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter,  RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

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
    chroma_docs_path = Path('chromadb/'+ suffix_path + '_' + embeddings_name.split(':')[0] + '_' + str(chunk_size)) 

    if chroma_docs_path.exists():
        return True




def _separate_name(name):
    """separate name str by ':' , for examples, separate hf:model/llama2.bin 
      and return hf, model/llama2.bin

    Args:
        name (str): name string

    Returns:
        (str, str): return separated two string
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

def _download_llama():
    import subprocess
    command = "git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git"
    subprocess.run(command, shell=True)
    subprocess.run("set FORCE_CMAKE=1", shell=True)
    subprocess.run("cd llama-cpp-python", shell=True)
    subprocess.run("python setup.py clean", shell=True)
    subprocess.run("python setup.py install", shell=True)

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
        print("building chroma db...\n")
        dir = Path(doc_path)
        pdf_files = dir.glob("*.pdf")
        loaders = [file.name for file in pdf_files]

        
        documents = []
        cou = 0

        for loader in loaders:
        
            try:
                documents.extend(PyPDFLoader(doc_path+loader).load())
                
            except:
                cou+=1
        
        info = "\n\nload files:\n\n" + doc_path+"total files:" + str(len(loaders)) +\
            " .\nfailed to read:" + str(cou) + " files\n"
        



        #text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=0)
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n'," ", ",",".","。","!" ], chunk_size=chunk_size, chunk_overlap=40)
        k = 0
        cum_ids = 0
        interval = 60
        while k < len(documents):
            cur_doc = documents[k:k+interval]
            texts = text_splitter.split_documents(cur_doc)
            if k==0:
                docsearch = Chroma.from_documents(texts, embeddings, persist_directory = storage_directory) 
                
            else:
                docsearch.add_documents(texts)
            print("added ", [doc.metadata['source'] for doc in cur_doc])
            k += interval
            cum_ids += len(texts)
            time.sleep( sleep_time )
        docsearch.persist()  
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
        
        from langchain.llms import LlamaCpp
        from langchain import PromptTemplate, LLMChain
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        
        

        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # Verbose is required to pass to the callback manager

        if model_type in ["llama", "llama2", "llama-cpp", "llama-cpu"]:
            model = LlamaCpp(n_ctx=4096, temperature=0.0,
                model_path = model_name,
                input={"temperature": 0.0, "max_length": 4096, "top_p": 1},
                callback_manager = callback_manager,
                verbose = True,
            )
            info = "selected llama cpu model\n"

        else:
            n_gpu_layers = 40
            n_batch = 512

            model = LlamaCpp(n_ctx=4096, temperature=0.0,
                model_path = model_name,
                n_gpu_layers = n_gpu_layers,
                n_batch = n_batch,
                callback_manager = callback_manager,
                verbose = True,
            )
            info = "selected llama gpu model\n"
    elif model_type in ["huggingface" , "huggingfacehub","transformers", "transformer", "huggingface-hub", "hf"]:
        from langchain import HuggingFaceHub
        from transformers import pipeline
        import torch
        from langchain.llms import HuggingFacePipeline
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            if hf_token is None:
                pipe = pipeline("text-generation", model=model_name,model_kwargs={"temperature":0,"max_length":4096}, device_map="auto"\
                                ,batch_size=1, torch_dtype=torch.float16)
            else:
                
                pipe = pipeline("text-generation", model=model_name, use_auth_token=hf_token,\
                    model_kwargs={"temperature":0,"max_length":4096}, device_map="auto",batch_size=1, torch_dtype=torch.float16)
            model = HuggingFacePipeline(pipeline=pipe)
            #model = HuggingFaceHub(
            #    repo_id = model_name, model_kwargs={"temperature": 0.1})
        
        
        
        info = f"selected huggingface model {model_name}.\n"
    else:
        model = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0)
        info = f"can not find the model {model_type}:{model_name}, use openai as default.\n"

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
    """based on chinese or english, calculate the length of words in a document

    Args:
        language (str): 'ch' for chinese and 'en' for others
        doc (Document): Document file

    Returns:
        int: length of words
    """
    if language=='ch':
        
        doc_length = len(list(jieba.cut(doc.page_content)))
    else:
        doc_length = len(doc.page_content.split())
    return doc_length

def get_docs_length(language:str, docs:list)->int:
    """based on chinese or english, calculate the length of words in documents


    Args:
        language (_type_): 'ch' for chinese and 'en' for others
        docs (list): list of Document files

    Returns:
        int: length of words
    """
    docs_length = 0
    for doc in docs:
        docs_length += get_doc_length(language, doc)
    return docs_length



def get_question_from_file(path:str)->list:
    """ parse questions from question file into a list format like this. Each question has question(q), possible answer(pa), and
    the nth of correct answer(ca), the format looks like :
    list[[q,pa1,pa2,pa3,ca], [q,pa1,pa2,pa3,pa4,ca]...]

    Args:
        path (str): path of question file

    Returns:
        list: list of questions 
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
                res = int(c)

                break
    return res