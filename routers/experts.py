from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import akasha
import akasha.helper
import akasha.db
import os
import json
import streamlit_utils as stu


DOCS_PATH = './docs'
EXPERT_CONFIG_PATH = './config/expert'
DATASET_CONFIG_PATH = "./config/dataset/"
DEFAULT_CONFIG = {'language_model':"openai:gpt-3.5-turbo",
            'search_type': "svm",
            'top_k': 5,
            'threshold': 0.1,
            'max_token': 3000,
            'temperature':0.0,
            'use_compression':0, # 0 for False, 1 for True
            'compression_language_model':"openai:gpt-3.5-turbo"}


### data class ###
class UserID(BaseModel):
    owner: str

class ExpertID(UserID):
    expert_name: str

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

### data class ###




router = APIRouter()








@router.get("/expert/show")
async def get_info_of_expert(user_input:ExpertID):
    """input the current user id and expert name, return the expert info(dict)

    Args:
        user_input (ExpertID): _description_
        owner : str
        expert_name : str
    Returns:
        dict: status, response(expert config dictionary)
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





@router.get("/expert/get_owner")
async def get_owner_expert_list(user_input:UserID):
    """input current user id, return all expert name and its owner name that current user has(list of dict)

    Args:
        user_input (UserID): _description_
        owner : str
        
    Returns:
        dict: status, response(list of dict)
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




@router.get("/expert/get")
async def get_use_expert_list(user_input:UserID):
    """input current user id, return all expert name and its owner name that current user can use(list of dict)

    Args:
        user_input (UserID): _description_
        owner : str
        
    Returns:
        dict: status, response(list of dict)
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



@router.get("/expert/get_consult")
async def get_consult_from_expert(user_input:ExpertID):
    """get the last consultation parameter of the expert

    Args:
        user_input (ExpertID): _description_
        owner : str
        expert_name : str
    Returns:
        dict: status, response(expert consultation parameter dictionary)
    """
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







@router.post("/expert/save_consult")
async def save_consult_to_expert(user_input:ExpertConsult):
    """save the consultation parameter to expert config file

    Args:
        user_input (ExpertConsult): 
        owner : str
        expert_name : str
        language_model : Optional[str] = "openai:gpt-3.5-turbo",
        search_type: Optional[str] = "svm",
        top_k: Optional[int] = 5,
        threshold: Optional[float] = 0.1,
        max_token: Optional[int] = 3000,
        temperature: Optional[float] = 0.0,
        use_compression: Optional[int] = 0, # 0 for False, 1 for True
        compression_language_model: Optional[str] = "openai:gpt-3.5-turbo"
    Returns:
        dict: status, response
    """
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








@router.post("/expert/create")
async def create_expert(user_input:ExpertInfo):
    """create expert config file and create chromadb for each file

    Args:
        user_input (ExpertInfo): _description_
        expert_name : str
        embedding_model : str
        chunk_size : int
        datasets : list of dict
        owner : str
    Returns:
        dict: status, response
    """
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





@router.post("/expert/update")
async def update_expert(user_input:ExpertEditInfo):
    """update expert config file and create chromadb for each file, also delete chromadb that need to delete(check if other expert use it or not)

    Args:
        user_input (ExpertEditInfo): _description_
        owner : str
        expert_name : str
        new_expert_name : str
        embedding_model : str
        chunk_size : int
        delete_datasets : list of dict
        new_embedding_model : str
        new_chunk_size : int
        add_datasets : list of dict
    Returns:
        dict: status, response
    """
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




@router.post("/expert/delete")
async def delete_expert(user_input:ExpertID):
    """delete expert config file and delete chromadb for each file it other expert not use it

    Args:
        user_input (ExpertID): _description_
        owner : str
        expert_name : str
    Returns:
        dict: status, response
    """
    owner = user_input.owner
    expert_name = user_input.expert_name
    
    if not stu.check_config(EXPERT_CONFIG_PATH):
        return {'status': 'fail', 'response': 'create config path failed.\n\n'}
    
    uid = akasha.helper.get_text_md5(expert_name + '-' + owner)
    data_path = Path(EXPERT_CONFIG_PATH) / (uid+'.json')
    if not data_path.exists():
        return {'status': 'fail', 'response': f'expert config file {expert_name} not found.\n\n'}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    ### check all delete all chromadb
    for dataset in data['datasets']:
        oner = dataset['owner']
        dataset_name = dataset['name']
        id = akasha.helper.get_text_md5(dataset_name + '-' + oner)
        for file in dataset['files']:
            stu.check_and_delete_chromadb(data['chunk_size'], data['embedding_model'], file, dataset_name, oner, id)
    
    ## delete Path(DATASET_CONFIG_PATH) / (uid+'.json') file
    os.remove(data_path)

    return {'status': 'success', 'response': f'delete expert {expert_name} successfully.\n\n'}

