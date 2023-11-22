import os
import json
from typing import List, Union, Optional
import hashlib
import datetime as dt
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

PATH_DATASET = './config/datasets'
PATH_EXPERT= './config/experts'

#%% Dataset
class dataset(object):
    
    base = None
    
    @staticmethod
    def list(owner:str=None):
        dataset_list = []
        json_files = os.listdir(PATH_DATASET)
        for f in json_files:
            with open(os.path.join(PATH_DATASET, f), 'r') as file:
                content = json.load(file)
            if (owner is None) or (content['owner'] == owner):
                dataset_list.append(content)
        return dataset_list
    
    @staticmethod
    def generate_hash(owner, dataset_name):
        combined_string = f"{owner}-{dataset_name}"
        sha256 = hashlib.sha256()
        sha256.update(combined_string.encode('utf-8'))
        hex_digest = sha256.hexdigest()
        return hex_digest
    
    @classmethod
    def create(cls, owner, name, description, files:List[dict], shared_users:list=[], path=PATH_DATASET):
        uid = cls.generate_hash(owner, name)
        cls.base = {
            "uid":uid,
            "name":name,
            "files":files,
            "description":description,
            "shared_users":shared_users,
            "owner":owner,
            "last_update":dt.datetime.now().strftime(DATETIME_FORMAT)
        }
        with open(os.path.join(path, f'{uid}.json'), 'w') as file:
            json.dump(cls.base, file)
        
        return cls 
    
    @classmethod
    def from_json(cls, owner:str, name:str, path=PATH_DATASET):
        dataset_jsons = os.listdir(path)
        uid = cls.generate_hash(owner, name)
        for dataset_json in dataset_jsons:
            if dataset_json == f'{uid}.json':
                with open(os.path.join(path, dataset_json)) as file:
                    cls.base = json.load(file)
                return cls
    
    @classmethod
    def save(cls, path=PATH_DATASET):
        try:
            with open(os.path.join(path, f'{cls.uid()}.json'), 'w') as file:
                json.dump(cls.base, file)
            return True
        except:
            return False
    @classmethod
    def delete(cls):
        os.remove(os.path.join(PATH_DATASET, f'{cls.uid()}.json'))
        return cls
    
    @classmethod        
    def uid(cls) -> str:
        return cls.base['uid']
    
    @classmethod
    def name(cls) -> str:
        return cls.base['name']
    
    @classmethod
    def owner(cls) -> str:
        return cls.base['owner']
    
    @classmethod
    def shared_users(cls) -> List[str]:
        return cls.base['shared_users']
    
    @classmethod
    def files(cls) -> List[dict]:
        return cls.base['files']
    
    @classmethod
    def filenames(cls, name_only=False) -> List[str]:
        return [f['filename'] for f in cls.base['files']]  
    
    @classmethod
    def description(cls) -> str:
        return cls.base['description']
    
    @classmethod
    def last_update(cls) -> str:
        return cls.base['last_update']
    
    @classmethod
    def file_md5(cls, filename:str) -> str:                
        return cls.base['files'][filename]['MD5']

    # actions
    @classmethod
    def set_uid(cls, owner:str=None, name:str=None):
        if owner is None:
            owner = cls.owner()
        if name is None:
            name = cls.name()
        cls.base['uid'] = cls.generate_hash(owner, name)
        return cls.base
    @classmethod
    def set_name(cls, name:str):
        cls.base['name'] = name
        return cls.base
    @classmethod
    def set_owner(cls, owner:str):
        cls.base['owner'] = owner
        return cls.base
    @classmethod
    def set_description(cls, description:str):
        cls.base['description'] = description
        return cls.base
    @classmethod
    def set_last_update(cls, last_update:dt.datetime=dt.datetime.now()):
        cls.base['last_update'] = last_update.strftime(DATETIME_FORMAT)
        return cls.base
    @classmethod
    def set_share_users(cls, users:List[str]):
        cls.base['shared_users'] = users
        return cls.base
    
    @classmethod
    def add_share_users(cls, users:List[str]):
        cls.base['shared_users'].extend(users)
        return cls.base
    @classmethod
    def remove_share_users(cls, users:List[str]):
        for user in users:
            cls.base['shared_users'].remove(user)
        return cls.base
    @classmethod
    def clean_shared_users(cls):
        cls.base['shared_users'] = []
        return cls.base
    @classmethod
    def add_files(cls, files:List[dict]):
        if cls.base['files'] is None:
            cls.base['files'] = []
        cls.base['files'].extend(files)
        return cls.base
    @classmethod
    def remove_files(cls, files:List[dict]):
        for file in files:
            for f in cls.base['files']:
                if f['MD5'] == file['MD5']:
                    cls.base['files'].remove(f)
                    break
        return cls.base
    @classmethod
    def clean_files(cls):
        cls.base['files'] = []
        return cls.base
    
    
#%% Expert
class expert(object):
    
    base = None
    
    @staticmethod
    def list(owner:str=None):
        expert_list = []
        json_files = os.listdir(PATH_EXPERT)
        for f in json_files:
            with open(os.path.join(PATH_EXPERT, f), 'r') as f:
                content = json.load(f)
            if (owner is None) or (content['owner'] == owner) :
                expert_list.append(content)
        return expert_list
    
    @staticmethod
    def generate_hash(owner, expert_name):
        combined_string = f"{owner}-{expert_name}"
        sha256 = hashlib.sha256()
        sha256.update(combined_string.encode('utf-8'))
        hex_digest = sha256.hexdigest()
        return hex_digest
    
    @classmethod
    def create(cls, owner, name, embedding_model, chunk_size, datasets:List[dict], shared_users:list=[], consultation:dict={}, path=PATH_EXPERT):
        uid = cls.generate_hash(owner, name)
        cls.base = {
            "uid":uid,
            "name":name,
            "embedding_model":embedding_model,
            "chunk_size":chunk_size,
            "datasets":datasets,
            "shared_users":shared_users,
            "owner":owner,
            "consultation":consultation
        }
        with open(os.path.join(path, f'{uid}.json'), 'w') as file:
            json.dump(cls.base, file)
        
        return cls
    
    @classmethod
    def from_json(cls, owner:str, name:str, path=PATH_EXPERT):
        expert_jsons = os.listdir(path)
        uid = cls.generate_hash(owner, name)
        for expert_json in expert_jsons:
            #if expert_json == f'{uid}.json':
            if expert_json == 'ex-expert.json':
                with open(os.path.join(path, expert_json)) as file:
                    cls.base = json.load(file)
                return cls
    
    @classmethod
    def save(cls, path=PATH_EXPERT):
        with open(os.path.join(path, f'{cls.uid()}.json'), 'w') as file:
            json.dump(cls.base, file)
        return cls
    
    @classmethod
    def delete(cls):
        os.remove(os.path.join(PATH_EXPERT, f'{cls.uid()}.json'))
        return cls
    
    @classmethod
    def uid(cls) -> str:
        return cls.base['uid']
    
    @classmethod
    def name(cls) -> str:
        return cls.base['name']
    
    @classmethod
    def owner(cls) -> str:
        return cls.base['owner']
    
    @classmethod
    def embedding_model(cls) -> Optional[str]:
        return cls.base['embedding_model']
    
    @classmethod
    def chunk_size(cls) -> Optional[int]:
        return cls.base['chunk_size']
    
    @classmethod
    def shared_users(cls) -> List[str]:
        return cls.base['shared_users']
    
    @classmethod
    def datasets(cls) -> List[dict]:
        return cls.base['datasets']
    
    @classmethod
    def dataset_files(cls, dataset_owner:str, dataset_name:str) -> List[str]:
        for d in cls.datasets():
            if d['owner'] == dataset_owner and d['name'] == dataset_name:
                return d['files']
        return []
    
    @classmethod
    def consultation(cls) -> dict:
        return cls.base['consultation']
    
    # actions
    @classmethod
    def set_uid(cls, owner:str=None, name:str=None):
        if owner is None:
            owner = cls.owner()
        if name is None:
            name = cls.name()
        cls.base['uid'] = cls.generate_hash(owner, name)
        return cls.base
    
    @classmethod
    def set_name(cls, name:str):
        cls.base['name'] = name
        return cls.base
    
    @classmethod
    def set_owner(cls, owner:str):
        cls.base['owner'] = owner
        return cls.base
    
    @classmethod
    def set_embedding_model(cls, embedding_model):
        cls.base['embedding_model'] = embedding_model
        return cls.base
    
    @classmethod
    def set_chunk_size(cls, chunk_size:int):
        cls.base['chunk_size'] = chunk_size
        return cls.base
    
    @classmethod
    def add_share_users(cls, users:List[str]):
        cls.base['shared_users'].extend(users)
        return cls.base
    
    @classmethod
    def remove_share_users(cls, users:List[str]):
        for user in users:
            cls.base['shared_users'].remove(user)
        return cls.base
    
    @classmethod
    def clean_shared_users(cls):
        cls.base['shared_users'] = []
        return cls.base
    
    @classmethod
    def add_dataset(cls, dataset_owner:str, dataset_name:str, dataset_files:List[str]):
        if not dataset_files:
            print(f'Cannot add dataset={dataset_name}@{dataset_owner} with empty file list to expert={cls.name()}')
            return cls.base 
        for d in cls.datasets():
            if d['owner'] == dataset_owner and d['name'] == dataset_name:
                break
        else:
            new_dataset = {'owner':dataset_owner, 'name':dataset_name, 'files':dataset_files}
            cls.base['datasets'].append(new_dataset)
        return cls.base
    
    @classmethod
    def remove_dataset(cls, dataset_owner:str, dataset_name:str):
        for d in cls.datasets():
            if d['owner'] == dataset_owner and d['name'] == dataset_name:
                cls.base['datasets'].remove(d)
                break
        return cls.base
    
    @classmethod
    def clean_datasets(cls):
        cls.base['datasets'] = []
        return cls.base
    
    @classmethod
    def add_files_of_dataset(cls, dataset_owner:str, dataset_name:str, dataset_files:List[str]):
        for idx,d in enumerate(cls.datasets()):
            if d['owner'] == dataset_owner and d['name'] == dataset_name:
                cls.base['datasets'][idx]['files'].extend(dataset_files)
                cls.base['datasets'][idx]['files'] = list(set(cls.base['datasets'][idx]['files']))
                break
        return cls.base
    
    @classmethod
    def remove_files_of_dataset(cls, dataset_owner:str, dataset_name:str, dataset_files:List[str]):
        for idx,d in enumerate(cls.datasets()):
            if d['owner'] == dataset_owner and d['name'] == dataset_name:
                for f in dataset_files:
                    if f in cls.base['datasets'][idx]['files']:
                        cls.base['datasets'][idx]['files'].remove(f)
                break  
        return cls.base
    
    @classmethod
    def clean_files_of_dataset(cls, dataset_owner:str, dataset_name:str):
        for idx,d in enumerate(cls.datasets()):
            if d['owner'] == dataset_owner and d['name'] == dataset_name:
                cls.base['datasets'][idx]['files'] = []
                break
        return cls.base 
    
    @classmethod
    def add_consultations(cls, consultations:dict):
        cls.base['consultation'].update(consultations)
        return cls.base 
    
    @classmethod
    def remove_consultations(cls, consultations:dict):
        for c in consultations:
            if c in cls.base['consultation']:
                del cls.base['consultation'][c]
        return cls.base
    
    @classmethod
    def clean_consultations(cls):
        cls.base['consultation'] = {}
        return cls.base
    
    @classmethod
    def change_owner(cls, new_owner:str, save_and_delete=False):
        if save_and_delete:
            cls.delete()
        cls.set_owner(new_owner)
        cls.set_uid(new_owner, cls.name())
        if save_and_delete:
            cls.save()
        return cls.base  
    
    @classmethod
    def change_name(cls, new_name:str, save_and_delete=False):
        if save_and_delete:
            cls.delete()
        cls.set_name(new_name)
        cls.set_uid(cls.owner, new_name)
        if save_and_delete:
            cls.save()
        return cls.base 
    
    @classmethod
    def reset_dataset(cls, dataset_owner:str, dataset_name:str, new_name:str, files:List[str]):
        for d in cls.base['datasets']:
            if d['owner'] == dataset_owner and d['name'] == dataset_name:
                d['name'] = new_name
                d['files'] = files
                break
            
    @classmethod
    def consultable(cls, return_reason=False) -> bool:
        if not cls.base.get('datasets'):
            msg = 'no dataset used.'
            return False if not return_reason else (False, msg)
        if all([not d.get('files') for d in cls.base['datasets']]):
            msg = 'no files exist in used datasets.'
            return False if not return_reason else (False, msg)
        if not cls.base.get('embedding_model'):
            msg = 'no embedding model used.'
            return False if not return_reason else (False, msg)
        if not cls.base.get('chunk_size'):
            msg = 'no chunksize set.'
            return False if not return_reason else (False, msg)
        return True if not return_reason else (True, '')
        