from fastapi import FastAPI, Query, UploadFile, File
import akasha
import akasha.helper
import akasha.db
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import os
import json
import streamlit_utils as stu
app = FastAPI()

DATASET_CONFIG_PATH = "./config/dataset/"
DOCS_PATH = './docs'
EXPERT_CONFIG_PATH = './config/expert'
OPENAI_CONFIG_PATH = "./config/openai.json"
DEFAULT_CONFIG = {'language_model':"openai:gpt-3.5-turbo",
            'search_type': "svm",
            'top_k': 5,
            'threshold': 0.1,
            'max_token': 3000,
            'temperature':0.0,
            'use_compression':0, # 0 for False, 1 for True
            'compression_language_model':"openai:gpt-3.5-turbo"}
### data class ###

class ConsultModel(BaseModel):
    data_path: Union[str, List[str]]
    prompt: Union[str, List[str]]
    chunk_size:Optional[int]=1000
    model:Optional[str] = "openai:gpt-3.5-turbo"
    topK:Optional[int] = 3 
    threshold:Optional[float] = 0.2
    search_type:Optional[str] = 'svm'
    system_prompt:Optional[str] = ""
    max_token:Optional[int]=3000
    temperature:Optional[float]=0.0
    use_chroma:Optional[bool]=True
    openai_config:Optional[Dict[str, Any]] = {}

class ConsultModelReturn(BaseModel):
    response: Union[str, List[str]]
    status: str
    logs: Dict[str, Any]


class UserID(BaseModel):
    owner: str
class DatasetID(UserID):
    dataset_name: str
class ExpertID(UserID):
    expert_name: str


class DatasetShare(DatasetID):
    shared_users: List[str]

class DatasetShareDelete(DatasetID):
    delete_users: List[str]


class DatasetInfo(DatasetID):
    dataset_description: Optional[str] = ""


class EditDatasetInfo(DatasetInfo):
    new_dataset_name: str
    new_dataset_description: Optional[str] = ""
    upload_files: Optional[List[str]] = []
    delete_files: Optional[List[str]] = []



class ExpertInfo(ExpertID):
    embedding_model: Optional[str] = "openai:text-embedding-ada-002"
    chunk_size: Optional[int] = 1000
    datasets: Optional[List[Dict]] = []
    

class ExpertEditInfo(ExpertInfo):
    new_expert_name: str
    new_embedding_model: Optional[str] = "openai:text-embedding-ada-002"
    new_chunk_size: Optional[int] = 1000
    add_datasets: Optional[List[Dict]]
    delete_datasets: Optional[List[Dict]]


class ExpertConsult(ExpertID):
    language_model : Optional[str] = "openai:gpt-3.5-turbo",
    search_type: Optional[str] = "svm",
    top_k: Optional[int] = 5,
    threshold: Optional[float] = 0.1,
    max_token: Optional[int] = 3000,
    temperature: Optional[float] = 0.0,
    use_compression: Optional[int] = 0, # 0 for False, 1 for True
    compression_language_model: Optional[str] = "openai:gpt-3.5-turbo"


class OpenAIKey(BaseModel):
    openai_key: Optional[str] = ""
    azure_key: Optional[str] = ""
    azure_base: Optional[str] = ""
    



@app.post("/regular_consult")
async def regular_consult(user_input: ConsultModel):
    
    if not stu.load_openai(openai_config=user_input.openai_config):
        return {'status': 'fail', 'response': 'load openai config failed.\n\n'}
     
    qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, topK=user_input.topK, threshold=user_input.threshold\
        , model=user_input.model, temperature=user_input.temperature, max_token=user_input.max_token\
        ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt, use_chroma = user_input.use_chroma)
    response = qa.get_response(doc_path=user_input.data_path, prompt = user_input.prompt)
    
    ## get logs
    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        logs =  qa.logs[qa.timestamp_list[-1]]
    
    
    if response == None or response == "":
        user_output = ConsultModelReturn(response="Can not get response from get_resposne.\n", status="fail", logs=logs)
    else:
        user_output = ConsultModelReturn(response=response, status="success", logs=logs)
    
    del qa    
    return user_output



@app.post("/deep_consult")
async def deep_consult(user_input: ConsultModel):
    
    if not stu.load_openai(openai_config=user_input.openai_config):
        return {'status': 'fail', 'response': 'load openai config failed.\n\n'}
    
    
    qa = akasha.Doc_QA(verbose=True, search_type=user_input.search_type, topK=user_input.topK, threshold=user_input.threshold\
        , model=user_input.model, temperature=user_input.temperature, max_token=user_input.max_token\
        ,chunk_size=user_input.chunk_size, system_prompt=user_input.system_prompt, use_chroma=user_input.use_chroma)
    response = qa.chain_of_thought(doc_path=user_input.data_path, prompt_list = user_input.prompt)
    ## get logs
    if len(qa.timestamp_list) == 0:
        logs = {}
    else:
        logs =  qa.logs[qa.timestamp_list[-1]]
    
    
    if response == None or response == [] or response == "":
        user_output = ConsultModelReturn(response="Can not get response from chain_of_thought.\n", status="fail", logs=logs)
    else:
        user_output = ConsultModelReturn(response=response, status="success", logs=logs)
    
    del qa    
    return user_output




@app.post("/dataset/create")
async def create_dataset(user_input:DatasetInfo):
    
    dataset_name = user_input.dataset_name
    dataset_description = user_input.dataset_description
    owner = user_input.owner
    
    uid = akasha.helper.get_text_md5(dataset_name + '-' + owner)
    if not stu.check_config(DATASET_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    

    save_path = Path(DATASET_CONFIG_PATH) / (uid+'.json')
    own_path = Path(DOCS_PATH) / owner
    if not stu.check_dir(own_path):
        return {'status': 'fail', 'response': 'create document path failed.\n\n'}
    doc_path = (own_path / dataset_name)
    
    ## list all files path in doc_path and get their md5 hash
    file_paths = list(doc_path.glob('*'))
    md5_list = []
    for file_path in file_paths:
        file_doc = akasha.db._load_file(file_path.__str__(), file_path.name.split('.')[-1])
        if file_doc == ""  or len(file_doc) == 0:
            md5_hash = ""
        else:
            md5_hash = akasha.helper.get_text_md5(''.join([fd.page_content for fd in file_doc]))
        
        md5_list.append(md5_hash)
    
    
    ## create dict and save to json file
    data = {
        "uid": uid,
        "name": dataset_name,
        "description": dataset_description,
        "owner": owner,
        "files": [{"filename":file_paths[i].name,"MD5":md5_list[i] } for i in range(len(file_paths))],   
        "last_update": stu.get_lastupdate_of_dataset(dataset_name, owner) 
    }
    
    ## write json file
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    return {'status': 'success', 'response': f'create dataset {dataset_name} successfully.\n\n'}





@app.post("/dataset/update")
async def update_dataset(user_input:EditDatasetInfo):
    
    dataset_name = user_input.dataset_name
    owner = user_input.owner
    new_dataset_name = user_input.new_dataset_name
    dataset_description = user_input.new_dataset_description
    upload_files = user_input.upload_files
    delete_files = user_input.delete_files
    uid = akasha.helper.get_text_md5(dataset_name + '-' + owner)
    if not stu.check_config(DATASET_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}

        
    data_path = Path(DATASET_CONFIG_PATH) / (uid+'.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## delete Path(DATASET_CONFIG_PATH) / (uid+'.json') file
    os.remove(data_path)
    
    
    if dataset_name != new_dataset_name:
        ### edit config name
        uid = akasha.helper.get_text_md5(new_dataset_name + '-' + owner)
        data['name'] = new_dataset_name
        data['uid'] = uid
        dataset_name = new_dataset_name
        
        
    save_path = Path(DATASET_CONFIG_PATH) / (uid+'.json')
    own_path = Path(DOCS_PATH) / owner
    if not stu.check_dir(own_path):
        return {'status': 'fail', 'response': 'create document path failed.\n\n'}
    doc_path = own_path / dataset_name
    new_files = []
    ### remove delete_files config from data['files']
    
    for dic in data['files']:
        if dic['filename'] not in delete_files:
            new_files.append(dic)
    
    exi_files = set([dic['filename'] for dic in data['files']])
    ### add upload_files config to data['files']
    for file in upload_files:
        if file in exi_files:
            continue
        file_doc = akasha.db._load_file((doc_path / file).__str__(), file.split('.')[-1])
        if file_doc == ""  or len(file_doc) == 0:
            md5_hash = ""
        else:
            md5_hash = akasha.helper.get_text_md5(''.join([fd.page_content for fd in file_doc]))
        new_files.append({"filename":file, "MD5":md5_hash})
        
        
    ## update dict and save to json file
    data['files'] = new_files
    data['description'] = dataset_description
    data['last_update'] = stu.get_lastupdate_of_dataset(dataset_name, owner)
    
    
    ## write json file
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    
    
    ### update expert which has this dataset
    ### update chromadb if some expert use this dataset
    return {'status': 'success', 'response': f'update dataset {dataset_name} successfully.\n\n'}




@app.post("/dataset/delete")
async def delete_dataset(user_input:DatasetID):
    owner = user_input.owner
    dataset_name = user_input.dataset_name
    
    if not stu.check_config(DATASET_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    uid = akasha.helper.get_text_md5(dataset_name + '-' + owner)
    data_path = Path(DATASET_CONFIG_PATH) / (uid+'.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': 'dataset config file not found.\n\n'}
    ## delete Path(DATASET_CONFIG_PATH) / (uid+'.json') file
    os.remove(data_path)
    
    ## delete this dataset in every expert's dataset list
    stu.check_and_delete_dataset(dataset_name, owner)
    return {'status': 'success', 'response': f'delete dataset {dataset_name} successfully.\n\n'}




@app.post("/dataset/share")
async def share_dataset(user_input:DatasetShare):
    owner = user_input.owner
    dataset_name = user_input.dataset_name
    shared_users = user_input.shared_users
    
    if not stu.check_config(DATASET_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    uid = akasha.helper.get_text_md5(dataset_name + '-' + owner)
    data_path = Path(DATASET_CONFIG_PATH) / (uid+'.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': 'dataset config file not found.\n\n'}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if 'shared_users' not in data.keys():
        data['shared_users'] = []
    
    ### add shared users into data['shared_users']
    vis = set(data['shared_users'])
    for user in shared_users:
        vis.add(user)
        
    data['shared_users'] = list(vis)
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return {'status': 'success', 'response': f'share dataset {dataset_name} successfully.\n\n'}



@app.post("/dataset/delete_share")
async def delete_share_dataset(user_input:DatasetShare):
    owner = user_input.owner
    dataset_name = user_input.dataset_name
    delete_users = user_input.delete_users
    
    if not stu.check_config(DATASET_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    uid = akasha.helper.get_text_md5(dataset_name + '-' + owner)
    data_path = Path(DATASET_CONFIG_PATH) / (uid+'.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': 'dataset config file not found.\n\n'}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if 'shared_users' not in data.keys() or len(data['shared_users']) == 0:
        data['shared_users'] = []
        return {'status': 'fail', 'response': 'dataset not shared to any user.\n\n'}
    
    ### add shared users into data['shared_users']
    vis = set(data['shared_users'])
    for user in delete_users:
        if user in vis:
            vis.remove(user)
        
    data['shared_users'] = list(vis)
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return {'status': 'success', 'response': f'edit shared users {dataset_name} successfully.\n\n'}









@app.get("/dataset/get_dcp")
async def get_description_from_dataset(user_input:DatasetID):
    """input the current user id and dataset name, return the description of dataset(str)

    Args:
        user_input (DatasetID): _description_

    Returns:
        _type_: _description_
    """
    owner = user_input.owner
    dataset_name = user_input.dataset_name
    if not stu.check_config(DATASET_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    
    uid = akasha.helper.get_text_md5(dataset_name + '-' + owner)
    data_path = Path(DATASET_CONFIG_PATH) / (uid + '.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': 'dataset config file not found.\n\n'}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if "description" not in data.keys():
        return {'status': 'fail', 'response': 'dataset description not found.\n\n'}
    return {'status': 'success', 'response': data['description']}
    

@app.get("/dataset/get_md5")
async def get_MD5_list_from_dataset(user_input:DatasetID):
    """input the current user id and dataset name, return the dataset's all file's md5 hash(list)

    Args:
        user_input (DatasetID): _description_

    Returns:
        _type_: _description_
    """
    
    owner = user_input.owner
    dataset_name = user_input.dataset_name
    if not stu.check_config(DATASET_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    
    uid = akasha.helper.get_text_md5(dataset_name + '-' + owner)
    data_path = Path(DATASET_CONFIG_PATH) / (uid + '.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': 'dataset config file not found.\n\n'}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    response = []
    for file in data['files']:
        if file['MD5'] != "":
            response.append(file['MD5'])
    return {'status': 'success', 'response': response}



@app.get("/dataset/get_filename")
async def get_filename_list_from_dataset(user_input:DatasetID):
    """input the current user id and dataset name, return the dataset's all file's file name(list)

    Args:
        user_input (DatasetID): _description_

    Returns:
        _type_: _description_
    """
    
    owner = user_input.owner
    dataset_name = user_input.dataset_name
    if not stu.check_config(DATASET_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    
    uid = akasha.helper.get_text_md5(dataset_name + '-' + owner)
    data_path = Path(DATASET_CONFIG_PATH) / (uid + '.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': 'dataset config file not found.\n\n'}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    response = []
    for file in data['files']:
        if file['filename'] != "":
            response.append(file['filename'])
    return {'status': 'success', 'response': response}






@app.get("/dataset/show")
async def get_info_of_dataset(user_input:DatasetID):
    """input the current user id and dataset name, return the dataset info(dict)

    Args:
        user_input (DatasetID): _description_

    Returns:
        _type_: _description_
    """
    owner = user_input.owner
    dataset_name = user_input.dataset_name
    if not stu.check_config(DATASET_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    
    uid = akasha.helper.get_text_md5(dataset_name + '-' + owner)
    data_path = Path(DATASET_CONFIG_PATH) / (uid+'.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': 'dataset not found.\n\n'}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {'status': 'success', 'response': data}   


@app.get("/dataset/get_owner")
async def get_owner_dataset_list(user_input:UserID):
    """input current user id, return all dataset name and its owner name that owner is current user(list of dict)

    Args:
        user_input (UserID): _description_

    Returns:
        _type_: _description_
    """
    owner = user_input.owner
    dataset_names = []
    if not stu.check_config(DATASET_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    ## get all dataset name
    p = Path(DATASET_CONFIG_PATH)
    for file in p.glob("*"):
        with open(file, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        if dataset['owner'] == owner:
            dataset_names.append({'dataset_name':dataset['name'], 'owner':dataset['owner']})
    return {'status': 'success', 'response': dataset_names}




@app.get("/dataset/get")
async def get_use_dataset_list(user_input:UserID):
    """input current user id, return all dataset name and its owner name that current user can use(list of dict)

    Args:
        user_input (UserID): _description_

    Returns:
        _type_: _description_
    """
    owner = user_input.owner
    dataset_names = []
    if not stu.check_config(DATASET_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    ## get all dataset name
    p = Path(DATASET_CONFIG_PATH)
    for file in p.glob("*"):
        with open(file, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        if dataset['owner'] == owner:
            dataset_names.append({'dataset_name':dataset['name'], 'owner':dataset['owner']})
        elif 'shared_users' in dataset.keys() and owner in dataset['shared_users']:
            dataset_names.append({'dataset_name':dataset['name'], 'owner':dataset['owner']})
    return {'status': 'success', 'response': dataset_names}
















@app.get("/expert/show")
async def get_info_of_expert(user_input:ExpertID):
    """input the current user id and dataset name, return the expert info(dict)

    Args:
        user_input (ExpertID): _description_

    Returns:
        _type_: _description_
    """
    owner = user_input.owner
    expert_name = user_input.expert_name
    if not stu.check_config(EXPERT_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    
    uid = akasha.helper.get_text_md5(expert_name + '-' + owner)
    data_path = Path(DATASET_CONFIG_PATH) / (uid+'.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': 'dataset not found.\n\n'}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {'status': 'success', 'response': data}   





@app.get("/expert/get_owner")
async def get_owner_expert_list(user_input:UserID):
    """input current user id, return all expert name and its owner name that current user has(list of dict)

    Args:
        user_input (UserID): _description_

    Returns:
        _type_: _description_
    """
    owner = user_input.owner
    expert_names = []
    if not stu.check_config(EXPERT_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    ## get all dataset name
    p = Path(EXPERT_CONFIG_PATH)
    for file in p.glob("*"):
        with open(file, 'r', encoding='utf-8') as file:
            expert = json.load(file)
        if expert['owner'] == owner:
            expert_names.append({'dataset_name':expert['name'], 'owner':expert['owner']})
       
    return {'status': 'success', 'response': expert_names}




@app.get("/expert/get")
async def get_use_expert_list(user_input:UserID):
    """input current user id, return all expert name and its owner name that current user can use(list of dict)

    Args:
        user_input (UserID): _description_

    Returns:
        _type_: _description_
    """
    owner = user_input.owner
    expert_names = []
    if not stu.check_config(EXPERT_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    ## get all dataset name
    p = Path(EXPERT_CONFIG_PATH)
    for file in p.glob("*"):
        with open(file, 'r', encoding='utf-8') as file:
            expert = json.load(file)
        if expert['owner'] == owner:
            expert_names.append({'dataset_name':expert['name'], 'owner':expert['owner']})
        elif 'shared_users' in expert.keys() and owner in expert['shared_users']:
            expert_names.append({'dataset_name':expert['name'], 'owner':expert['owner']})
    return {'status': 'success', 'response': expert_names}



@app.get("/expert/get_consult")
async def get_consult_from_expert(user_input:ExpertID):
    
    owner = user_input.owner
    expert_name = user_input.expert_name
    
    
    
    if not stu.check_config(EXPERT_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    
    uid = akasha.helper.get_text_md5(expert_name + '-' + owner)
    data_path = Path(EXPERT_CONFIG_PATH) / (uid+'.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': 'expert config file not found.\n\n'}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'consultation' not in data.keys():
        data['consultation'] = {}
        
    for key in DEFAULT_CONFIG.keys():
        if key not in data['consultation'] or data['consultation'][key] == "":
            data['consultation'][key] = DEFAULT_CONFIG[key]
    
    return {'status': 'success', 'response': data['consultation']}   







@app.post("/expert/save_consult")
async def save_consult_to_expert(user_input:ExpertConsult):
    
    owner = user_input.owner
    expert_name = user_input.expert_name
    
    if not stu.check_config(EXPERT_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    
    uid = akasha.helper.get_text_md5(expert_name + '-' + owner)
    data_path = Path(EXPERT_CONFIG_PATH) / (uid+'.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': 'expert config file not found.\n\n'}
    
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        
    consult_info = user_input.dict()
    consult_info.pop('owner')
    consult_info.pop('expert_name')
    data['consultation'] = consult_info
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    return {'status': 'success', 'response': 'save consultation successfully.\n\n'}








@app.post("/expert/create")
async def create_expert(user_input:ExpertInfo):
    
    expert_name = user_input.expert_name
    embedding_model = user_input.embedding_model
    chunk_size = user_input.chunk_size
    datasets = user_input.datasets
    owner = user_input.owner
    
    uid = akasha.helper.get_text_md5(expert_name + '-' + owner)
    warning = []
    if not stu.check_config(EXPERT_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    

    save_path = Path(EXPERT_CONFIG_PATH) / (uid+'.json')
    
    ## create chromadb for each file
    for dataset in datasets:
        doc_path = DOCS_PATH + '/' + dataset['owner'] + '/' + dataset['name']
        for file in dataset['files']:
            file_path = doc_path  + '/' + file
            
            suc, text = akasha.db.create_single_file_db(file_path , embedding_model, chunk_size,)            
            if not suc:
                
                warning.append(f'create chromadb for {file_path} failed, {text}.\n\n')
    
    ## create dict and save to json file
    data = {
        "uid": uid,
        "name": expert_name,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "owner": owner,
        "datasets": datasets,   
        "consultation": DEFAULT_CONFIG 
    }
    
    ## write json file
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    
    return {'status': 'success', 'response': f'create expert {expert_name} successfully.\n\n', 'warning': warning}





@app.post("/expert/update")
async def update_expert(user_input:ExpertEditInfo):
    
    owner = user_input.owner
    expert_name = user_input.expert_name
    new_expert_name = user_input.new_expert_name
    embedding_model = user_input.embedding_model
    chunk_size = user_input.chunk_size
    delete_datasets = user_input.delete_datasets
    new_embedding_model = user_input.new_embedding_model
    new_chunk_size = user_input.new_chunk_size
    add_datasets = user_input.add_datasets
    
    warning = []
    uid = akasha.helper.get_text_md5(expert_name + '-' + owner)
    
    if not stu.check_config(EXPERT_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    data_path = Path(EXPERT_CONFIG_PATH) / (uid+'.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': 'expert config file not found.\n\n'}

    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## delete Path(DATASET_CONFIG_PATH) / (uid+'.json') file
    os.remove(data_path)    
    
    if expert_name != new_expert_name:
        ### edit config name
        uid = akasha.helper.get_text_md5(new_expert_name + '-' + owner)
        if (Path(EXPERT_CONFIG_PATH) / (uid+'.json')) .exists():
            return {'status': 'fail', 'response': 'new expert name already exists.\n\n'}
        data['name'] = new_expert_name
        data['uid'] = uid
        expert_name = new_expert_name
        
    
    if new_embedding_model == data['embedding_model'] and new_chunk_size == data['chunk_size']:
        
        ### check chromadb that need to delete
        for dataset in delete_datasets:
            oner = dataset['owner']
            dataset_name = dataset['name']
            id = akasha.helper.get_text_md5(dataset_name + '-' + oner)
            for file in dataset['files']:
                stu.check_and_delete_chromadb(data['chunk_size'], data['embedding_model'], file, dataset_name, oner, id)
        
        
        ## create chromadb for each new file
        for dataset in add_datasets:
            doc_path = DOCS_PATH + '/' + dataset['owner'] + '/' + dataset['name']
            for file in dataset['files']:
                file_path = doc_path  + '/' + file
                
                suc, text = akasha.db.create_single_file_db(file_path , data['embedding_model'], data['chunk_size'],)            
                if not suc:
                    warning.append(f'create chromadb for {file_path} failed, {text}.\n\n')
    
    
                  
    else:
        ### check all delete all chromadb
        for dataset in data['datasets']:
            oner = dataset['owner']
            dataset_name = dataset['name']
            id = akasha.helper.get_text_md5(dataset_name + '-' + oner)
            for file in dataset['files']:
                stu.check_and_delete_chromadb(data['chunk_size'], data['embedding_model'], file, dataset_name, oner, id)
    
    
    if len(delete_datasets) > 0 :
        data['datasets'] = stu.delete_datasets_from_expert(data['datasets'], delete_datasets)
    if len(add_datasets) > 0 :
        data['datasets'] = stu.add_datasets_to_expert(data['datasets'], add_datasets)
    
    
    if new_embedding_model != data['embedding_model'] or new_chunk_size != data['chunk_size']:
        
        data['embedding_model'] = new_embedding_model
        data['chunk_size'] = new_chunk_size
        
        ## create chromadb for each new file
        for dataset in data['datasets']:
            doc_path = DOCS_PATH + '/' + owner + '/' + dataset['name']
            for file in dataset['files']:
                file_path = doc_path  + '/' + file
                
                suc, text = akasha.db.create_single_file_db(file_path , data['embedding_model'], data['chunk_size'],)            
                if not suc:
                    warning.append(f'create chromadb for {file_path} failed, {text}.\n\n')
        
        
    save_path = Path(EXPERT_CONFIG_PATH) / (uid+'.json')
    
    ## write json file
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    return {'status': 'success', 'response': f'update expert {expert_name} successfully.\n\n', 'warning': warning}




@app.post("/expert/delete")
async def delete_expert(user_input:ExpertID):
    owner = user_input.owner
    expert_name = user_input.expert_name
    
    if not stu.check_config(EXPERT_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    uid = akasha.helper.get_text_md5(expert_name + '-' + owner)
    data_path = Path(EXPERT_CONFIG_PATH) / (uid+'.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': f'expert config file {expert_name} not found.\n\n'}
    ## delete Path(DATASET_CONFIG_PATH) / (uid+'.json') file
    os.remove(data_path)

    return {'status': 'success', 'response': f'delete expert {expert_name} successfully.\n\n'}





@app.post("/openai/save")
async def save_openai_key(user_input:OpenAIKey):
    """save openai key into config file

    Args:
        user_input (_type_, optional): _description_.
    """
    
    if not stu.check_config(""):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}

    save_path = Path(OPENAI_CONFIG_PATH)
    if save_path.exists():
        with open(save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
    else :
        data = {}
        
    flag = False
    if user_input.openai_key.replace(' ','') != "":
        data['openai_key'] = user_input.openai_key.replace(' ','')
        flag = True
    if user_input.azure_base.replace(' ','') != "" and user_input.azure_key.replace(' ','') != "":
        flag = True        
        data['azure_key'] = user_input.azure_key.replace(' ','')
        data['azure_base'] = user_input.azure_base.replace(' ','')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    if not flag:
        return {'status': 'fail', 'response': 'can not save any openai configuration.\n\n'}
    return {'status': 'success', 'response': 'save openai key successfully.\n\n'}
    



@app.get("/openai/test_openai")
async def test_openai_key(user_input:OpenAIKey):
    import openai
    openai_key = user_input.openai_key.replace(' ','')
    if openai_key != "":
        openai.api_type = "open_ai"
        openai.api_key = openai_key
        try:
            openai.Model.list()
        except openai.error.AuthenticationError as e:
            return {'status': 'fail', 'response': 'openai key is invalid.\n\n'}
        else:
            return {'status': 'success', 'response': 'openai key is valid.\n\n'}
        
    return {'status': 'fail', 'response': 'openai key empty.\n\n'}




@app.get("/openai/test_azure")
async def test_azure_key(user_input:OpenAIKey):
    import openai
    azure_key = user_input.azure_key.replace(' ','')
    azure_base = user_input.azure_base.replace(' ','')
    if azure_key != "" and azure_base != "":
        openai.api_type = "azure"
        openai.api_base = azure_base
        openai.api_version = "2023-05-15"
        openai.api_key = azure_key

        try:
            openai.Model.list()
        except openai.error.AuthenticationError as e:
            return {'status': 'fail', 'response': 'azure openai key is invalid.\n\n'}
        else:
            return {'status': 'success', 'response': 'azure openai key is valid.\n\n'}
        
    return {'status': 'fail', 'response': 'azure openai key or base url is empty.\n\n'}