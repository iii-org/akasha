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


class ConsultModelReturn(BaseModel):
    response: Union[str, List[str]]
    status: str
    logs: Dict[str, Any]


class UserID(BaseModel):
    owner: str
class DatasetID(UserID):
    dataset_name: str
    
class DatasetInfo(DatasetID):
    dataset_description: Optional[str] = ""


class EditDatasetInfo(DatasetInfo):
    new_dataset_name: str
    new_dataset_description: Optional[str] = ""
    upload_files: Optional[List[str]] = []
    delete_files: Optional[List[str]] = []



@app.post("/regular_consult")
async def regular_consult(user_input: ConsultModel):
    
    print(DATASET_CONFIG_PATH)  
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
    
    print(user_input)
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
    doc_path = Path(DOCS_PATH) / dataset_name
    
    ## list all files path in doc_path and get their md5 hash
    file_paths = list(doc_path.glob('*'))
    md5_list = []
    for file_path in file_paths:
        file_doc = akasha.db._load_file(file_path, file_path.name.split('.')[-1])
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
        "last_update": stu.get_lastupdate_of_dataset(dataset_name) 
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
    doc_path = Path(DOCS_PATH) / dataset_name
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
    data['last_update'] = stu.get_lastupdate_of_dataset(dataset_name)
    
    
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

    return {'status': 'success', 'response': f'delete dataset {dataset_name} successfully.\n\n'}









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
async def get_MD5_list_from_dataset(user_input:DatasetID):
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

