import streamlit as st
import pandas as pd
import os, json
from typing import List, Union, Dict, Any, Tuple
import requests
from pathlib import Path
import api_utils as apu
import akasha.db
import subprocess
import time
import datetime
import traceback
import opencc
import gc, torch

cc = opencc.OpenCC("s2t.json")
CHUNKSIZE = 3000
SPINNER_MESSAGE = "Wait for api response..."
HOST = os.getenv("API_HOST", "http://127.0.0.1")
PORT = os.getenv("API_PORT", "8000")

api_urls = {
    "load_openai": f"{HOST}:{PORT}/openai/load_openai",
    "get_docs_path": f"{HOST}:{PORT}/get_docs_path",
    "get_dataset": f"{HOST}:{PORT}/dataset/get",
    "get_model_path": f"{HOST}:{PORT}/get_model_path",
    "get_filename_list": f"{HOST}:{PORT}/dataset/get_filename",
    "show_dataset": f"{HOST}:{PORT}/dataset/show",
    "get_owner_dataset": f"{HOST}:{PORT}/dataset/get_owner",
    "share_dataset": f"{HOST}:{PORT}/dataset/share",
    "create_dataset": f"{HOST}:{PORT}/dataset/create",
    "update_dataset": f"{HOST}:{PORT}/dataset/update",
    "delete_dataset": f"{HOST}:{PORT}/dataset/delete",
    "get_expert": f"{HOST}:{PORT}/expert/get",
    "get_owner_expert": f"{HOST}:{PORT}/expert/get_owner",
    "create_chromadb": f"{HOST}:{PORT}/expert/create_chromadb",
    "create_expert": f"{HOST}:{PORT}/expert/create",
    "create_expert2": f"{HOST}:{PORT}/expert/create2",
    "update_expert": f"{HOST}:{PORT}/expert/update",
    "update_expert2": f"{HOST}:{PORT}/expert/update2",
    "delete_expert": f"{HOST}:{PORT}/expert/delete",
    "share_expert": f"{HOST}:{PORT}/expert/share",
    "show_expert": f"{HOST}:{PORT}/expert/show",
    "get_dataset_dcp": f"{HOST}:{PORT}/dataset/get_dcp",
    "test_openai": f"{HOST}:{PORT}/openai/test_openai",
    "test_azure": f"{HOST}:{PORT}/openai/test_azure",
    "save_openai": f"{HOST}:{PORT}/openai/save",
    "choose_openai": f"{HOST}:{PORT}/openai/choose",
    "is_default_api": f"{HOST}:{PORT}/openai/is_default_api",
    "get_consult": f"{HOST}:{PORT}/expert/get_consult",
    "save_consult": f"{HOST}:{PORT}/expert/save_consult",
    "get_default_consult": f"{HOST}:{PORT}/expert/get_default_consult",
    "get_chromadb_path": f"{HOST}:{PORT}/expert/get_chromadb_path",
    "get_md5_name": f"{HOST}:{PORT}/dataset/get_md5",
    "regular_consult": f"{HOST}:{PORT}/regular_consult",
    "regular_consult_stream": f"{HOST}:{PORT}/regular_consult_stream",
    "get_summary": f"{HOST}:{PORT}/get_summary",
    "chat": f"{HOST}:{PORT}/chat",
    "chat_stream": f"{HOST}:{PORT}/chat_stream",
    "get_nickname": f"{HOST}:{PORT}/get_nickname",
    "get_nicknames": f"{HOST}:{PORT}/get_all_nicknames",
}


def check_dataset_is_shared(dataset_name: str) -> bool:
    """if the dataset is a shared dataset, return True, else return False.
    the format of shared dataset is dataset_name@owner_name, for example: dataset1@user1
    the format of owned dataset is dataset_name, for example: dataset1

    Args:
        dataset_name (str): the dataset name

    Returns:
         bool: return True if the dataset is a shared dataset, else return False.
    """
    return "@" in dataset_name


def check_expert_is_shared(expert_name: str) -> bool:
    """if the expert is a shared expert, return True, else return False.
    the format of shared expert is expert_name@owner_name, for example: expert1@user1
    the format of owned expert is expert_name, for example: expert1

    Args:
        expert_name (str): the expert name

    Returns:
         bool: return True if the expert is a shared expert, else return False.
    """
    return "@" in expert_name


def ask_chat(
    username: str,
    prompt: str,
    expert_owner: str,
    expert_name: str,
    advanced_params: dict,
    col_answer: st.columns = None,
):
    """ask chat and get response from akasha, the question and resposne will be saved in st.session_state['history_messages'].
    first check if prompt empty or not, then add openai config if needed.
    second, call get_data_path function to get all chromadb path and format all data into a dict and pass to regular consult api.
    if auto_clean is True, clean question area after ask question; if username is the owner of expert, save last consult config for expert.

    Args:
        username (str): the current user account name, used to check if the user is the owner of expert.
        sys_prompt (str): system prompt of the question.
        prompt (str): question.
        expert_owner (str): the owner of expert.
        expert_name (str): the name of expert.
        advanced_params (dict): advanced parameters of expert, include 'datasets', 'model', 'search_type', 'topK','threshold',
            'max_doc_len', 'temperature', 'use_compression', 'chunk_size', 'embedding_model', 'compression_language_model'.
        auto_clean (bool, optional): if True, clean the question area after ask question. Defaults to False.

    Returns:
        bool: return True if ask question successfully, else return False.
    """

    if not prompt or prompt == "":
        st.error("❌ Please input question")
        time.sleep(3)
        return False

    if (advanced_params["embedding_model"].split(":")[0] == "openai"
            or advanced_params["model"].split(":")[0] == "openai"):
        openai_config = get_openai_config(username)
    else:
        openai_config = {}

    ## get all chromadb path and format all data into a dict ##
    data_path = get_data_path(
        username,
        advanced_params["datasets"],
        advanced_params["embedding_model"],
        advanced_params["chunk_size"],
    )

    data = {
        "data_path": data_path,
        "prompt": prompt,
        "system_prompt": advanced_params["system_prompt"],
        "embedding_model": advanced_params["embedding_model"],
        "chunk_size": advanced_params["chunk_size"],
        "model": advanced_params["model"],
        "temperature": advanced_params["temperature"],
        "topK": advanced_params["topK"],
        "threshold": advanced_params["threshold"],
        "search_type": advanced_params["search_type"],
        "max_doc_len": advanced_params["max_doc_len"],
        "openai_config": openai_config,
        "history_messages": st.session_state.history_messages,
    }
    try:
        with st.spinner(SPINNER_MESSAGE):
            # using streaming mode #
            if "openai" in advanced_params["model"] or "hf:" in advanced_params[
                    "model"] or "huggingface" in advanced_params[
                        "model"] or "remote:" in advanced_params["model"]:
                with col_answer.chat_message("user"):
                    st.markdown(prompt)

                with col_answer.chat_message("assistant"):
                    placeholder = st.empty()
                    chat_response = requests.post(api_urls["chat_stream"],
                                                  json=data,
                                                  stream=True)
                    metadata = None
                    response = ""
                    doc_metadata = " "
                    for chunk in chat_response.iter_content(
                            chunk_size=1024, decode_unicode=True):

                        try:
                            # Try to parse the chunk as JSON
                            metadata = json.loads(chunk)
                        except json.JSONDecodeError:
                            # If it's not JSON, write it to the stream
                            response += chunk
                            placeholder.write(response)
                    trans_result = cc.convert(response)

                    if metadata is None:
                        metadata = {"doc_metadata": []}
                    if not isinstance(metadata, dict):
                        metadata = {"doc_metadata": []}

                    doc_metadata = '\n\n'.join(metadata["doc_metadata"])

                    st.session_state.history_messages.append({
                        "role": "user",
                        "content": prompt
                    })
                    st.session_state.history_messages.append({
                        "role":
                        "assistant",
                        "content":
                        trans_result,
                        "doc_metadata":
                        doc_metadata
                    })
                    placeholder.empty()
                    placeholder.markdown(trans_result, help=doc_metadata)

                collect_logs(data, trans_result, "chat")

            else:
                with col_answer.chat_message("user"):
                    st.markdown(prompt)

                response = requests.post(api_urls["chat"], json=data).json()

                if len(response["warnings"]) > 0:
                    for w in response["warnings"]:
                        st.warning(
                            f"Encountered issues while reading the file: {w} ")

                if response["status"] != "success":
                    api_fail(response["response"])
                    return False
                doc_metadata = '\n\n'.join(response["logs"]["doc_metadata"])
                st.session_state.history_messages.append({
                    "role": "user",
                    "content": prompt
                })

                st.session_state.history_messages.append({
                    "role":
                    "assistant",
                    "content":
                    response["response"],
                    "doc_metadata":
                    doc_metadata
                })
                with col_answer.chat_message("assistant"):
                    st.markdown(response["response"], help=doc_metadata)
                st.session_state.logs[response["timestamp"]] = response["logs"]
        clean()

        # save last consult config for expert if is the owner of expert
        if username == expert_owner:
            data = {
                "system_prompt":
                advanced_params["system_prompt"],
                "language_model":
                advanced_params["model"],
                "search_type":
                advanced_params["search_type"],
                "top_k":
                advanced_params["topK"],
                "threshold":
                advanced_params["threshold"],
                "max_doc_len":
                advanced_params["max_doc_len"],
                "temperature":
                advanced_params["temperature"],
                "use_compression":
                advanced_params["use_compression"],
                "compression_language_model":
                advanced_params["compression_language_model"],
            }
            data["owner"] = expert_owner
            data["expert_name"] = expert_name
            with st.spinner(SPINNER_MESSAGE):
                response = requests.post(api_urls["save_consult"],
                                         json=data).json()
            if response["status"] != "success":
                st.warning("cannot save last consult config for expert")

    except Exception as e:
        api_fail(e.__str__())
    return True


def ask_question(
    username: str,
    prompt: str,
    expert_owner: str,
    expert_name: str,
    advanced_params: dict,
    col_answer: st.columns = None,
):
    """ask question and get response from akasha, the question and resposne will be saved in session_state['que'] and session_state['ans'].
    first check if prompt empty or not, then add openai config if needed.
    second, call get_data_path function to get all chromadb path and format all data into a dict and pass to regular consult api.
    if auto_clean is True, clean question area after ask question; if username is the owner of expert, save last consult config for expert.

    Args:
        username (str): the current user account name, used to check if the user is the owner of expert.
        sys_prompt (str): system prompt of the question.
        prompt (str): question.
        expert_owner (str): the owner of expert.
        expert_name (str): the name of expert.
        advanced_params (dict): advanced parameters of expert, include 'datasets', 'model', 'search_type', 'topK','threshold',
            'max_doc_len', 'temperature', 'use_compression', 'chunk_size', 'embedding_model', 'compression_language_model'.
        auto_clean (bool, optional): if True, clean the question area after ask question. Defaults to False.

    Returns:
        bool: return True if ask question successfully, else return False.
    """

    if not prompt or prompt == "":
        st.error("❌ Please input question")
        time.sleep(3)
        return False

    if (advanced_params["embedding_model"].split(":")[0] == "openai"
            or advanced_params["model"].split(":")[0] == "openai"):
        openai_config = get_openai_config(username)
    else:
        openai_config = {}

    ## get all chromadb path and format all data into a dict ##
    data_path = get_data_path(
        username,
        advanced_params["datasets"],
        advanced_params["embedding_model"],
        advanced_params["chunk_size"],
    )

    data = {
        "data_path": data_path,
        "prompt": prompt,
        "system_prompt": advanced_params["system_prompt"],
        "embedding_model": advanced_params["embedding_model"],
        "chunk_size": advanced_params["chunk_size"],
        "model": advanced_params["model"],
        "temperature": advanced_params["temperature"],
        "topK": advanced_params["topK"],
        "threshold": advanced_params["threshold"],
        "search_type": advanced_params["search_type"],
        "max_doc_len": advanced_params["max_doc_len"],
        "openai_config": openai_config,
    }
    try:
        with st.spinner(SPINNER_MESSAGE):

            if "openai" in advanced_params["model"] or "hf:" in advanced_params[
                    "model"] or "huggingface" in advanced_params[
                        "model"] or "remote:" in advanced_params["model"]:

                with col_answer.chat_message("assistant"):
                    placeholder = st.empty()
                    chat_response = requests.post(
                        api_urls["regular_consult_stream"],
                        json=data,
                        stream=True)
                    metadata = None
                    response = ""
                    doc_metadata = " "
                    for chunk in chat_response.iter_content(
                            chunk_size=1024, decode_unicode=True):

                        try:
                            # Try to parse the chunk as JSON
                            metadata = json.loads(chunk)
                        except json.JSONDecodeError:
                            # If it's not JSON, write it to the stream
                            response += chunk
                            placeholder.write(response)
                    st.session_state["que"] = prompt
                    st.session_state["ans"] = cc.convert(response)

                    if metadata is None:
                        metadata = {"doc_metadata": []}
                    elif not isinstance(metadata, dict):
                        metadata = {"doc_metadata": []}

                    doc_metadata = '\n\n'.join(metadata["doc_metadata"])
                    placeholder.empty()
                    placeholder.markdown(st.session_state["ans"],
                                         help=doc_metadata)
                collect_logs(data, st.session_state["ans"], "get_response")

            else:
                response = requests.post(api_urls["regular_consult"],
                                         json=data).json()

                if len(response["warnings"]) > 0:
                    for w in response["warnings"]:
                        st.warning(
                            f"Encountered issues while reading the file: {w} ")

                if response["status"] != "success":
                    api_fail(response["response"])
                    return False

                st.session_state["que"] = prompt
                st.session_state["ans"] = response["response"]
                with col_answer.chat_message("assistant"):
                    st.markdown(response["response"],
                                help=response["logs"]["doc_metadata"])
                st.session_state.logs[response["timestamp"]] = response["logs"]
        clean()

        # save last consult config for expert if is the owner of expert
        if username == expert_owner:
            data = {
                "system_prompt":
                advanced_params["system_prompt"],
                "language_model":
                advanced_params["model"],
                "search_type":
                advanced_params["search_type"],
                "top_k":
                advanced_params["topK"],
                "threshold":
                advanced_params["threshold"],
                "max_doc_len":
                advanced_params["max_doc_len"],
                "temperature":
                advanced_params["temperature"],
                "use_compression":
                advanced_params["use_compression"],
                "compression_language_model":
                advanced_params["compression_language_model"],
            }
            data["owner"] = expert_owner
            data["expert_name"] = expert_name
            with st.spinner(SPINNER_MESSAGE):
                response = requests.post(api_urls["save_consult"],
                                         json=data).json()
            if response["status"] != "success":
                st.warning("cannot save last consult config for expert")

    except Exception as e:
        api_fail(e.__str__())
    return True


def list_experts(owner: str = None,
                 name_only: bool = False,
                 include_shared=True):
    """list all expert names that the owner can use if include_shared is True, else list all experts that the owner owned.

    Args:
        owner (str, optional): owner name, usually is current username. Defaults to None.
        name_only (bool, optional): not use. Defaults to False.
        include_shared (bool, optional): if True return shared experts. Defaults to True.

    Returns:
        list[str]: list of expert names, not that if it's shared expert from other owner,
            the format is expertname@ownername, for example: expert1@user1
    """
    # list all experts (of specific owner)
    if include_shared:
        with st.spinner(SPINNER_MESSAGE):
            response = requests.get(api_urls["get_expert"],
                                    json={
                                        "owner": owner
                                    }).json()

    else:
        with st.spinner(SPINNER_MESSAGE):
            response = requests.get(api_urls["get_owner_expert"],
                                    json={
                                        "owner": owner
                                    }).json()

    if response["status"] != "success":
        api_fail(response["response"])
        return []

    experts = [
        e["dataset_name"]
        if e["owner"] == owner else f"{e['dataset_name']}@{e['owner']}"
        for e in response["response"]
    ]
    # if name_only:
    #     return [e['name'] if e['owner'] == owner else f"{e['name']}@{e['owner']}" for e in experts]
    return experts


def list_models() -> list:
    """list all models that can be used in akasha, besides based openai models, this function will check all directorys
    and .gguf files under 'modes_dir' directory.

    Returns:
        list[str]: list of models, noted that the format is model_type:model_name, for example: openai:gpt-3.5-turbo
    """
    base = [
        "openai:gpt-3.5-turbo", "openai:gpt-3.5-turbo-16k", "openai:gpt-4",
        "openai:gpt-4-32k"
    ]
    try:
        response = requests.get(api_urls["get_model_path"]).json()
        if response["status"] != "success":
            raise Exception(response["response"])
        base = response["response"]
    except Exception as e:
        api_fail(e.__str__())

    return base


def create_expert(
    owner: str,
    expert_name: str,
    expert_embedding: str,
    expert_chunksize: int,
    expert_add_files: dict,
):
    """change the format of expert_add_files from dict to list of dict, each dict contains owner, name and files.
    call create_expert api to create expert configuration file, then create chromadb for each file.

    Args:
        owner (str): owner name
        expert_name (str): expert name
        expert_embedding (str): expert embedding model name, the format is model_type:model_name, for example: openai:text-embedding-ada-002
        expert_chunksize (int): the max chunk size of each segment of documents
        expert_add_files (dict): dict of files, the format is {dataset_name: set(file_name)}, for example: {'dataset1': {'file1', 'file2'}, 'dataset2': {'file3', 'file4'}}

    Raises:
        Exception: show the error messages if create expert failed.

    Returns:
        bool: return True if create expert successfully, else return False.
    """

    # validate inputs
    ## check chunksize is valid: not extremely large
    if expert_chunksize > CHUNKSIZE:
        st.warning(f"❌ Chunksize should be less than {CHUNKSIZE}")
        return False
    ## check expert name is valid: not used already
    user_experts = list_experts(owner, name_only=True)
    if expert_name in user_experts:
        st.error(f"❌ Expert '{expert_name}' already exists")
        time.sleep(3)
        return False

    try:
        ## check datasets is valid: not empty ##
        for k, v in expert_add_files.items():
            if len(v) == 0:
                raise Exception(
                    f" Dataset '{k}' should select at least 1 file")

        ## get openai config if needed ##
        if expert_embedding.split(":")[0] == "openai":
            openai_config = get_openai_config(owner)
        else:
            openai_config = {}

        datasets = []
        for k, v in expert_add_files.items():
            dataset_owner = (owner if not check_dataset_is_shared(k) else
                             k.split("@")[-1])
            dataset_name = k if not check_dataset_is_shared(k) else k.split(
                "@")[0]
            filename_list = list(v)
            datasets.append({
                "owner": dataset_owner,
                "name": dataset_name,
                "files": filename_list
            })

        data = {
            "owner": owner,
            "expert_name": expert_name,
            "embedding_model": expert_embedding,
            "chunk_size": expert_chunksize,
            "datasets": datasets,
            "openai_config": openai_config,
        }
        with st.spinner(SPINNER_MESSAGE):
            response = requests.post(api_urls["create_expert"],
                                     json=data).json()
        if response["status"] != "success":
            raise Exception(response["response"])

        # st.success(f'Expert\'{expert_name}\' has been created successfully')

        ## create chromadb for each file ##
        apu.load_openai(config=openai_config)
        progress_count, progress_itv = 0.0, 1 / len(response["file_paths"])
        loading_bar = st.progress(progress_count, text="Creating chromadb...")

        for file_path in response["file_paths"]:

            file_name = file_path.split("/")[-1]
            loading_bar.progress(progress_count,
                                 text=f"Creating chromadb for {file_name}...")
            suc, text = akasha.db.create_single_file_db(
                file_path, expert_embedding, expert_chunksize)
            progress_count += progress_itv
            if not suc:
                st.warning(
                    f"❌ Create chromadb for file {file_name} failed, {text}")

        response = requests.post(api_urls["create_expert2"], json=data).json()
        if response["status"] != "success":
            raise Exception(response["response"])

        loading_bar.empty()

    except Exception as e:
        st.error("❌ Expert creation failed, " + e.__str__())
        time.sleep(3)
        return False

    return True


def edit_expert(
    owner: str,
    expert_name: str,
    new_expert_name: str,
    default_expert_embedding: str,
    new_expert_embedding: str,
    default_expert_chunksize: int,
    new_expert_chunksize: int,
    default_expert_datasets: list,
    expert_used_dataset_files_dict: dict,  #:Dict[set],
    share_or_not: bool,
    shared_user_accounts: list = [],
) -> bool:
    """update the expert configuration file, first check if the new expert name is used already, then check if the new chunksize is valid.
    then get the delete_datasets and add_datasets from default_expert_datasets(original selected files)\
    and expert_used_dataset_files_dict(current selected files), 
    call update_expert api to update expert configuration file, then delete chromadb for non-used file(need to check all other experts)\
    and create chromadb for new file(s).

    Args:
        owner (str): owner name
        expert_name (str): old expert name
        new_expert_name (str): new expet name, may be the same as old expert name
        default_expert_embedding (str): old expert embedding model name, the format is model_type:model_name, for example: openai:text-embedding-ada-002
        new_expert_embedding (str): new expert embedding model name, may be the same as old expert embedding model name
        default_expert_chunksize (int): old expert chunksize
        new_expert_chunksize (int): new expert chunksize, may be the same as old expert chunksize
        default_expert_datasets (list): old expert datasets, the format is [{'owner':owner, 'name':dataset_name, 'files':filename_list}, ...]
        expert_used_dataset_files_dict (dict): new expert datasets, the format is {dataset_name: set(file_name)}, for example: {'dataset1': {'file1', 'file2'}, 'dataset2': {'file3', 'file4'}}
        share_or_not (bool): if True, need to add shared_users in expert configuration file, else not share expert with other users
        shared_user_accounts (list, optional): the list of users that can read the expert config file. Defaults to [].

    Returns:
        bool : return True if update expert successfully, else return False.
    """

    # validate inputs
    ## update_expert_name is valid: not used already
    user_experts = list_experts(owner, name_only=True)
    if (new_expert_name != expert_name) and (new_expert_name in user_experts):
        st.error(f"❌ Expert={expert_name} already exists")
        return False
    ## update_chunksize is valid: not extremely large
    if new_expert_chunksize > CHUNKSIZE:
        st.warning(f"❌ Chunksize should be less than {CHUNKSIZE}")
        return False
    ## new_expert_datasets is valid: not empty
    if len(expert_used_dataset_files_dict) == 0:
        st.error(f"❌ Expert should use at least one dataset")
        return False
    ## at least one file is selected among all datasets
    # for _,fileset in expert_used_dataset_files_dict.items():
    #     if len(fileset) == 0:
    #         st.error(f'❌ Dataset \'{_}\' should select at least 1 file.')
    #         return False
    ## must select at least one user to share expert when share_or_not=True
    if share_or_not:
        if len(shared_user_accounts) == 0:
            st.error(
                f"❌ Please select user(s) to share expert, or disable user-sharing."
            )
            return False

    # get delete_datasets and add_datasets
    try:
        if new_expert_embedding.split(":")[0] == "openai":
            openai_config = get_openai_config(owner)
        else:
            openai_config = {}
        delete_datasets = []
        default_expert_datasets_dict = {}
        for ds in default_expert_datasets:
            cur_dataset_name = (ds["name"] if ds["owner"] == owner else
                                f"{ds['name']}@{ds['owner']}")
            del_list = []
            default_expert_datasets_dict[cur_dataset_name] = ds["files"]

            if cur_dataset_name not in expert_used_dataset_files_dict:
                delete_datasets.append({
                    "owner": ds["owner"],
                    "name": ds["name"],
                    "files": ds["files"]
                })
            else:
                for f in ds["files"]:
                    if f not in expert_used_dataset_files_dict[
                            cur_dataset_name]:
                        del_list.append(f)
                if len(del_list) > 0:
                    delete_datasets.append({
                        "owner": ds["owner"],
                        "name": ds["name"],
                        "files": del_list
                    })

        add_datasets = []

        for k, v in expert_used_dataset_files_dict.items():
            add_list = []
            if k not in default_expert_datasets_dict:
                add_datasets.append({
                    "owner":
                    owner
                    if not check_dataset_is_shared(k) else k.split("@")[-1],
                    "name":
                    k if not check_dataset_is_shared(k) else k.split("@")[0],
                    "files":
                    list(v),
                })
            else:
                for f in v:
                    if f not in default_expert_datasets_dict[k]:
                        add_list.append(f)
                if len(add_list) > 0:
                    add_datasets.append({
                        "owner":
                        owner if not check_dataset_is_shared(k) else
                        k.split("@")[-1],
                        "name":
                        k
                        if not check_dataset_is_shared(k) else k.split("@")[0],
                        "files":
                        add_list,
                    })

        data = {
            "owner": owner,
            "expert_name": expert_name,
            "new_expert_name": new_expert_name,
            "embedding_model": default_expert_embedding,
            "chunk_size": default_expert_chunksize,
            "new_embedding_model": new_expert_embedding,
            "new_chunk_size": new_expert_chunksize,
            "delete_datasets": delete_datasets,
            "add_datasets": add_datasets,
            "openai_config": openai_config,
        }

    except Exception as e:
        st.error("❌ Expert edition failed during process datasets, " +
                 e.__str__())
        time.sleep(3)
        return False

    with st.spinner(SPINNER_MESSAGE):
        response = requests.post(api_urls["update_expert"], json=data).json()
    if response["status"] != "success":
        api_fail(response["response"])
        return False

    if len(response["delete_chromadb"]) > 0:
        delete_chromadb(response["delete_chromadb"])

    ## create chromadb for each file ##
    if len(response["file_paths"]) > 0:
        apu.load_openai(config=openai_config)
        progress_count, progress_itv = 0.0, 1 / len(response["file_paths"])
        loading_bar = st.progress(progress_count, text="Creating chromadb...")

        for file_path in response["file_paths"]:

            file_name = file_path.split("/")[-1]
            loading_bar.progress(progress_count,
                                 text=f"Creating chromadb for {file_name}...")
            suc, text = akasha.db.create_single_file_db(
                file_path, new_expert_embedding, new_expert_chunksize)
            progress_count += progress_itv
            if not suc:
                st.warning(
                    f"❌ Create chromadb for file {file_name} failed, {text}")

        loading_bar.empty()

    response2 = requests.post(api_urls["update_expert2"],
                              json={
                                  'data': response['new_json_data'],
                                  'old_uid': response['old_uid']
                              }).json()
    if response2["status"] != "success":
        return False

    if add_shared_users_to_expert(owner, new_expert_name, share_or_not,
                                  shared_user_accounts):

        return True

    return False


def delete_expert(username: str, expert_name: str) -> bool:
    """call delete_expert api to delete expert config file and non-used chromadb.

    Args:
        username (str): user name
        expert_name (str): expert name

    Returns:
        bool: return True if delete expert successfully, else return False.
    """
    # delete expert from all experts in config
    with st.spinner(SPINNER_MESSAGE):
        response = requests.post(
            api_urls["delete_expert"],
            json={
                "owner": username,
                "expert_name": expert_name
            },
        ).json()
    if response["status"] != "success":
        api_fail(response["response"])
        return False

    if len(response["delete_chromadb"]) > 0:
        delete_chromadb(response["delete_chromadb"])

    os.remove(response["delete_json_data"])

    st.success(f"Expert '{expert_name}' has been deleted successfully")
    return True


def list_datasets(owner: str = None,
                  name_only: bool = False,
                  include_shared: bool = False):
    """list all dataset names that the owner can use if include_shared is True, else list all datasets that the owner owned.

    Args:
        owner (str, optional): owner name, usually is current username. Defaults to None.
        name_only (bool, optional): not use. Defaults to False.
        include_shared (bool, optional): if True return shared datasets. Defaults to True.

    Returns:
        list[str]: list of dataset names, not that if it's shared dataset from other owner,
            the format is datasetname@ownername, for example: dataset1@user1
    """
    # list all datasets (of specific owner)

    if include_shared:
        with st.spinner(SPINNER_MESSAGE):
            response = requests.get(api_urls["get_dataset"],
                                    json={
                                        "owner": owner
                                    }).json()
    else:
        with st.spinner(SPINNER_MESSAGE):
            response = requests.get(api_urls["get_owner_dataset"],
                                    json={
                                        "owner": owner
                                    }).json()

    if response["status"] != "success":
        api_fail(response["response"])
        return []

    datasets = [
        e["dataset_name"]
        if e["owner"] == owner else f"{e['dataset_name']}@{e['owner']}"
        for e in response["response"]
    ]
    # if name_only:
    #     return [d['name'] if d['owner'] == username else f"{d['name']}@{d['owner']}" for d in datasets]

    return datasets


def add_shared_users_to_expert(owner: str,
                               expert_name: str,
                               share_boolean: bool,
                               shared_users: list = []) -> bool:
    """call share_expert api to add shared_users list in the expert configuration file.

    Args:
        owner (str): owner name
        expert_name (str): expert name
        share_boolean (bool): if True, add shared_users list in the expert configuration file, else not share expert with other users
        shared_users (list, optional): shared_users list. Defaults to [].

    Raises:
        Exception: show the error messages if share expert failed.

    Returns:
        bool : return True if share expert successfully or not need to add shared_users, else return False.
    """
    if not share_boolean:
        return True

    try:
        with st.spinner(SPINNER_MESSAGE):
            response = requests.post(
                api_urls["share_expert"],
                json={
                    "owner": owner,
                    "expert_name": expert_name,
                    "shared_users": shared_users,
                },
            ).json()

        if response["status"] != "success":
            raise Exception(response["response"])

    except Exception as e:
        api_fail("Expert sharing failed, " + e.__str__())
        return False
    return True


def add_shared_users_to_dataset(owner: str, dataset_name: str,
                                share_boolean: bool,
                                shared_users: list) -> bool:
    """call share_dataset api to add shared_users list in the dataset configuration file.

    Args:
        owner (str): owner name
        dataset_name (str): dataset name
        share_boolean (bool): if True, add shared_users list in the dataset configuration file, else not share dataset with other users
        shared_users (list, optional): shared_users list. Defaults to [].

    Raises:
        Exception: show the error messages if share dataset failed.

    Returns:
        bool : return True if share dataset successfully or not need to add shared_users, else return False.
    """
    if not share_boolean:
        return True

    try:
        with st.spinner(SPINNER_MESSAGE):
            response = requests.post(
                api_urls["share_dataset"],
                json={
                    "owner": owner,
                    "dataset_name": dataset_name,
                    "shared_users": shared_users,
                },
            ).json()

        if response["status"] != "success":
            raise Exception(response["response"])

    except Exception as e:
        api_fail("Dataset sharing failed, " + e.__str__())
        return False
    return True


def create_dataset(dataset_name: str, dataset_description: str,
                   uploaded_files: vars, owner: str) -> bool:
    """create doc files in DOC_PATH/{owner}/{dataset_name} , and call api to create dataset config.

    Args:
        dataset_name (str): dataset name
        dataset_description (str): dataset description
        uploaded_files (vars): bytes uploaded files
        owner (str): owner name

    Raises:
        Exception: _description_
        Exception: _description_
        Exception: _description_
        Exception: _description_
        Exception: _description_
    """
    # validate inputs
    suc_count = 0

    try:
        with st.spinner(SPINNER_MESSAGE):
            DOCS_PATH = requests.get(
                api_urls["get_docs_path"]).json()["response"]

        if dataset_name.replace(" ", "") == "":
            raise Exception("dataset name cannot be empty")

        if not apu.check_dir(DOCS_PATH):
            raise Exception(f"can not create {DOCS_PATH} directory")

        owner_path = Path(DOCS_PATH) / owner

        if not apu.check_dir(owner_path):
            raise Exception(f"can not create {owner} directory in {DOCS_PATH}")

        save_path = owner_path / dataset_name

        ## check if save_path is already exist
        if save_path.exists():
            raise Exception(f"Dataset '{dataset_name}' is already exist")
        else:
            save_path.mkdir()
        ## check file size/extension is non-empty/extremely large
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            if len(bytes_data) == 0:
                st.warning(f"File={uploaded_file.name} is empty")
                continue
            if len(bytes_data) > 100000000:
                st.warning(f"File={uploaded_file.name} is too large")
                continue

            with open(save_path.joinpath(uploaded_file.name), "wb") as f:
                f.write(bytes_data)
            suc_count += 1
        #  st.write("uploaded file:", uploaded_file.name)
        if suc_count == 0:
            # delete save_path
            save_path.rmdir()
            raise Exception("No file is uploaded successfully")
        # save data file(s) to local path={new_dataset_name}

        data = {
            "dataset_name": dataset_name,
            "dataset_description": dataset_description,
            "owner": owner,
        }
        with st.spinner(SPINNER_MESSAGE):
            response = requests.post(api_urls["create_dataset"],
                                     json=data).json()

        if response["status"] != "success":
            raise Exception(response["response"])

        # st.success(f'Dataset\'{dataset_name}\' has been created successfully')
    except Exception as e:
        st.error("❌ Dataset creation failed, " + e.__str__())
        time.sleep(3)
        return False
    return True


def edit_dataset(
    dataset_name: str,
    new_dataset_name: str,
    new_description: str,
    uploaded_files: vars,
    delete_file_set: set,
    owner: str,
) -> bool:
    """doing files and directory edition for update dataset.
     1. check if DOCS_PATH exist or create it
     2. check if owner directory exist or create it
     3. check if new dataset name is already exist
     4. check if old dataset name is exist
     5. rename old dataset name if we use new dataset name
     6. delete files in delete_file_set from local path={new_dataset_name}
     7. check files and save uploaded files to local path={new_dataset_name}
     8. collect params and call api to update dataset config, related chromadbs.


    Args:
        dataset_name (str): old dataset name
        new_dataset_name (str): new dataset name
        new_description (str): new dataset description
        uploaded_files (vars): byte uploaded files
        delete_file_set (set): filename that need to be deleted
        owner (str): owner name

    Raises:
        Exception: can not create DOCS_PATH directory
        Exception: can not create owner directory in DOCS_PATH
        Exception: new dataset name is already exist or old dataset name is not exist
        Exception: Dataset={dataset_name} is not exist
        Exception: api response status is not success
    """
    # validate inputs

    # save uploaded files to local path={new_dataset_name}
    # delete files in delete_file_set from local path={new_dataset_name}

    try:
        with st.spinner(SPINNER_MESSAGE):
            DOCS_PATH = requests.get(
                api_urls["get_docs_path"]).json()["response"]
        if not apu.check_dir(DOCS_PATH):
            raise Exception(f"can not create {DOCS_PATH} directory")

        owner_path = Path(DOCS_PATH) / owner

        if not apu.check_dir(owner_path):
            raise Exception(f"can not create {owner} directory in {DOCS_PATH}")

        save_path = owner_path / dataset_name
        upload_files_list = []
        delete_files_list = []
        ## rename dataset name to new_dataset_name if there're any changes
        if new_dataset_name != dataset_name:
            if (owner_path / new_dataset_name).exists():
                raise Exception(
                    f"New Dataset Name={new_dataset_name} is already exist")

            ## check if save_path is already exist
            if not save_path.exists():
                raise Exception(f"Old Dataset={dataset_name} is not exist")
            else:
                save_path.rename(owner_path / new_dataset_name)
                save_path = owner_path / new_dataset_name

        ## delete files in delete_file_set from local path={new_dataset_name}
        for file in delete_file_set:
            if (save_path / file).exists():
                (save_path / file).unlink()
                delete_files_list.append(file)

        ## check file size/extension is non-empty/extremely large
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            if len(bytes_data) == 0:
                st.warning(f"File={uploaded_file.name} is empty")
                continue
            if len(bytes_data) > 100000000:
                st.warning(f"File={uploaded_file.name} is too large")
                continue

            with open(save_path.joinpath(uploaded_file.name), "wb") as f:
                f.write(bytes_data)

            upload_files_list.append(uploaded_file.name)

        data = {
            "dataset_name": dataset_name,
            "new_dataset_name": new_dataset_name,
            "new_dataset_description": new_description,
            "owner": owner,
            "upload_files": upload_files_list,
            "delete_files": delete_files_list,
        }
        with st.spinner(SPINNER_MESSAGE):
            response = requests.post(api_urls["update_dataset"],
                                     json=data).json()

        if response["status"] != "success":
            raise Exception(response["response"])

        # st.success(f'Dataset\'{dataset_name}\' has been updated successfully')

    except Exception as e:
        st.error("❌ Dataset edition failed, " + e.__str__())
        time.sleep(3)
        return False

    return True


def delete_dataset(dataset_name: str, owner: str):
    """delete doc files in DOC_PATH/{owner}/{dataset_name} , and call api to delete dataset config, related chromadbs.

    Args:
        dataset_name (str): dataset name
        owner (str): owner name

    Raises:
        Exception: can not create DOCS_PATH directory
        Exception: can not create owner directory in DOCS_PATH
        Exception: Dataset={dataset_name} is not exist
        Exception: api response status is not success
    """
    try:
        with st.spinner(SPINNER_MESSAGE):
            DOCS_PATH = requests.get(
                api_urls["get_docs_path"]).json()["response"]

        if not apu.check_dir(DOCS_PATH):
            raise Exception(f"can not create {DOCS_PATH} directory")

        owner_path = Path(DOCS_PATH) / owner

        if not apu.check_dir(owner_path):
            raise Exception(f"can not create {owner} directory in {DOCS_PATH}")

        save_path = owner_path / dataset_name
        if not save_path.exists():
            raise Exception(f"Dataset '{dataset_name}' is not exist")

        import shutil

        shutil.rmtree(save_path)

        data = {"dataset_name": dataset_name, "owner": owner}
        with st.spinner(SPINNER_MESSAGE):
            response = requests.post(api_urls["delete_dataset"],
                                     json=data).json()

        if response["status"] != "success":
            raise Exception(response["response"])

        st.success(f"Dataset'{dataset_name}' has been deleted successfully")
    except Exception as e:
        st.error("Dataset deletion failed, " + e.__str__())
        time.sleep(3)
        return

    return


def get_file_list_of_dataset(username: str,
                             dataset_name: str,
                             name_only: bool = False) -> list:
    """call 'get_filename_list' api to get the all file names list of the dataset.

    Args:
        username (str): owner name
        dataset_name (str): dataset name
        name_only (bool, optional): _description_. Defaults to False.

    Returns:
        list: list of file names
    """
    if "@" in dataset_name:
        dataset_name, username = dataset_name.split("@")
    # dataset = config.dataset.from_json(username, dataset_name)
    # files = dataset.files()
    with st.spinner(SPINNER_MESSAGE):
        response = requests.get(
            api_urls["get_filename_list"],
            json={
                "owner": username,
                "dataset_name": dataset_name
            },
        ).json()

    return response["response"]


def get_lastupdate_of_file_in_dataset(dataset_name: str, file_name: str,
                                      owner: str) -> str:
    """call the 'get_docs_path' to get the file path and get last update time of the file, change to string format.

    Args:
        dataset_name (str): dataset name
        file_name (str): file name
        owner (str): owner name

    Returns:
        str : date time string
    """
    with st.spinner(SPINNER_MESSAGE):
        DOCS_PATH = requests.get(api_urls["get_docs_path"]).json()["response"]
    file_path = os.path.join(DOCS_PATH, owner, dataset_name, file_name)
    last_update = os.path.getmtime(file_path)
    return last_update


def get_datasets_of_expert(username: str,
                           datasets: list,
                           candidate_datasets: list = None) -> list:
    """get dataset names of expert, if candidate_datasets is not None, only return the dataset names that in candidate_datasets.

    Args:
        username (str): owner name
        datasets (list): list of datasets, the format is [{'owner':owner, 'name':dataset_name, 'files':filename_list}, ...]
        candidate_datasets (list, optional): _description_. Defaults to None.

    Returns:
        list : list of dataset names
    """
    dataset_names = [
        e["name"] if e["owner"] == username else f"{e['name']}@{e['owner']}"
        for e in datasets
    ]

    if candidate_datasets is None:
        return dataset_names
    return [d for d in dataset_names if d in candidate_datasets]


def check_expert_use_shared_dataset(expert_datasets: list,
                                    username: str = None) -> bool:
    """check if this expert use any shared dataset.

    Args:
        expert_datasets (list): the dataset list of expert, the format is [{'owner':owner, 'name':dataset_name, 'files':filename_list}, ...]
        username (str, optional): the current user name. Defaults to None.

    Returns:
        bool : return True if this expert use any shared dataset, else return False.
    """
    # check if expert use shared dataset
    for d in expert_datasets:
        if d["owner"] != username:
            return True
    return False


# settings
def _save_openai_configuration(key: str, add_session: bool = True) -> bool:
    """first check if openai api key is valid, then save openai api key to session state['openai_key'].

    Args:
        key (str): openai api key

    Returns:
        bool : return True if save openai api key successfully, else return False.
    """
    # check if openai api key is valid
    # save openai api key
    with st.spinner(SPINNER_MESSAGE):
        response = requests.get(api_urls["test_openai"],
                                json={
                                    "openai_key": key
                                }).json()
    if response["status"] != "success":
        api_fail(response["response"])
        return False

    if add_session:
        st.session_state["openai_key"] = key
    return True


def _save_azure_openai_configuration(key: str,
                                     endpoint: str,
                                     add_session: bool = True) -> bool:
    """first check if azure openai credentials are valid, then save azure openai credentials to session state['azure_key']
    and session state['azure_base'].

    Args:
        key (str): azure openai api key
        endpoint (str): azure openai endpoint url

    Returns:
        bool: return True if save azure openai credentials successfully, else return False.
    """
    # check if azure openai credentials are valid
    # save azure openai credentials
    with st.spinner(SPINNER_MESSAGE):
        response = requests.get(api_urls["test_azure"],
                                json={
                                    "azure_key": key,
                                    "azure_base": endpoint
                                }).json()
    if response["status"] != "success":
        api_fail(response["response"])
        return False

    if add_session:
        st.session_state["azure_key"] = key
        st.session_state["azure_base"] = endpoint
    return True


def save_api_configs(
    use_openai: bool = False,
    use_azure_openai: bool = False,
    openai_key: str = "",
    azure_openai_key: str = "",
    azure_openai_endpoint: str = "",
) -> bool:
    """check if use_openai or use_azure_openai is True, then call _save_openai_configuration or _save_azure_openai_configuration

    Args:
        use_openai (bool, optional): use openai . Defaults to False.
        use_azure_openai (bool, optional): use azure openai. Defaults to False.
        openai_key (str, optional): key value of openai_key. Defaults to ''.
        azure_openai_key (str, optional): key value of azure_key. Defaults to ''.
        azure_openai_endpoint (str, optional): url of azure_base. Defaults to ''.

    Returns:
        bool : True if save api configs successfully, else return False.
    """
    # save api configs
    if use_openai:
        if not _save_openai_configuration(openai_key):
            return False
        st.success("OpenAI configurations have been saved successfully")
    if use_azure_openai:
        if not _save_azure_openai_configuration(azure_openai_key,
                                                azure_openai_endpoint):
            return False
        st.success("Azure configurations have been saved successfully")
    return True


def save_openai_to_file(
    owner: str,
    use_openai: bool = False,
    use_azure_openai: bool = False,
    openai_key: str = "",
    azure_openai_key: str = "",
    azure_openai_endpoint: str = "",
) -> bool:
    """call save_openai api to save openai api configs to config file.

    Args:
        owner (str): owner name
        use_openai (bool, optional): use openai. Defaults to False.
        openai_key (str, optional): key value of openai_key. Defaults to ''.
        azure_openai_key (str, optional): key value of azure_key. Defaults to ''.
        azure_openai_endpoint (str, optional): url of azure_base. Defaults to ''.

    Returns:
        bool : True if save openai api configs to config file successfully, else return False.
    """
    if not use_openai:
        openai_key = ""
    if not use_azure_openai:
        azure_openai_key = ""
        azure_openai_endpoint = ""
    data = {
        "owner": owner,
        "openai_key": openai_key,
        "azure_key": azure_openai_key,
        "azure_base": azure_openai_endpoint,
    }

    with st.spinner(SPINNER_MESSAGE):
        response = requests.post(api_urls["save_openai"], json=data).json()

    if response["status"] != "success":
        api_fail(response["response"])
        return False

    st.success("OpenAI configuration file has been saved successfully")
    return True


def api_fail(response: Union[str, list]):
    """if call api return status is not 'success', show the error message.

    Args:
        response (str,list): _description_
    """
    if isinstance(response, str):
        st.error(f"❌ API failed: {response}")
    elif isinstance(response, list):
        res = "".join(response)
        st.error(f"❌ API failed: {res}")
    time.sleep(3)

    return


def check_file_selected_by_expert(datasets: list, dataset_name: str,
                                  dataset_owner: str, filename: str) -> bool:
    """check if the filename is in the dataset of expert config file

    Args:
        datasets (list): list of datasets in expert config file
        dataset_name (str): the file's dataset name that we want to check
        dataset_owner (str): the file's dataset owner that we want to check
        filename (str): the file name that we want to check

    Returns:
        bool: return True if the filename is in the expert config file, else return False.
    """
    # check if the filename is in the dataset of expert config file

    for ds in datasets:
        if ds["name"] == dataset_name and ds["owner"] == dataset_owner:
            if filename in ds["files"]:
                return True
    return False


def delete_chromadb(dir_name_list: list):
    """input the list of chromadb path, delete all chromadb in the list.

    Args:
        dir_name_list (list): list of chromadb path
    """
    import shutil, time

    for db_storage_path in dir_name_list:
        suc, try_num = False, 0
        while (not suc) and try_num <= 3:
            try:
                shutil.rmtree(Path(db_storage_path))
                suc = True

            except Exception as e:
                time.sleep(1)
                err_msg = e.__str__()
                try_num += 1
                continue
        if not suc:
            st.warning("cannot delete " + err_msg)

    return


def get_last_consult_for_expert(expert_owner: str, expert_name: str) -> dict:
    """get the last consult for expert, if can not get last consult, get default consult.

    Args:
        expert_owner (str): owner name of expert
        expert_name (str): expert name

    Returns:
        dict: the last consult dictionary
    """
    with st.spinner(SPINNER_MESSAGE):
        response = requests.get(
            api_urls["get_consult"],
            json={
                "owner": expert_owner,
                "expert_name": expert_name
            },
        ).json()

    ### if can not get last consult, get default consult ###
    if response["status"] != "success":
        with st.spinner(SPINNER_MESSAGE):
            response = requests.get(api_urls["get_default_consult"]).json()
        if response["satatus"] != "success":
            api_fail(response["response"])
            return {}
        return response["response"]

    return response["response"]


def check_consultable(datasets: list, embed: str,
                      chunk_size: int) -> Tuple[bool, str]:
    """check if the datasets, embed model, chunksize are valid.

    Args:
        datasets (list): datasets in the expert
        embed (str): embedding model in the expert
        chunk_size (int): chunksize in the expert
    Returns:
        (bool, str) : return True if all datasets have files, embed model is not empty, chunksize is not 0, else return False and
        return the error message.
    """

    if len(datasets) == 0:
        msg = "no dataset used."
        return False, msg
    if all([not d.get("files") for d in datasets]):
        msg = "no files exist in used datasets."
        return False, msg
    if embed == "" or embed == None:
        msg = "no embedding model used."
        return False, msg
    if chunk_size == 0 or isinstance(chunk_size, str):
        msg = "no chunksize set."
        return False, msg
    return True, ""


def get_dataset_info(owner: str, dataset_name: str) -> Tuple[list, str, str]:
    """get all parameters of dataset.

    Args:
        owner (str): owner name
        dataset_name (str): dataset name


    Returns:
        (list,str,str): return the list of file names, description, last update time of dataset.
    """
    # get dataset info from config
    with st.spinner(SPINNER_MESSAGE):
        response = requests.get(
            api_urls["show_dataset"],
            json={
                "owner": owner,
                "dataset_name": dataset_name
            },
        ).json()

    if response["status"] != "success":
        api_fail(response["response"])
        return [], "", ""
    filelist = [f["filename"] for f in response["response"]["files"]]

    old_shared_users = []
    if "shared_users" in response["response"]:
        name_dic = get_all_nicknames()
        for u in response["response"]["shared_users"]:
            nick_name = name_dic.get(u, "")
            if nick_name == "":
                continue

            old_shared_users.append(f"{u} ({nick_name})")
    return (
        filelist,
        response["response"]["description"],
        response["response"]["last_update"],
        old_shared_users,
    )


def get_expert_info(owner: str,
                    expert_name: str) -> Tuple[list, str, str, list]:
    """get all parameters of expert config file.

    Args:
        owner (str): owner name
        expert_name (str): expert name


    Returns:
        (list, str, str, list): return the list of datasets, embed model, chunk size, shared_users list.
    """
    # get dataset info from config
    with st.spinner(SPINNER_MESSAGE):
        response = requests.get(api_urls["show_expert"],
                                json={
                                    "owner": owner,
                                    "expert_name": expert_name
                                }).json()

    if response["status"] != "success":
        api_fail(response["response"])
        return [], "", "", []
    shared_users = []
    if "shared_users" in response["response"]:
        name_dic = get_all_nicknames()
        for u in response["response"]["shared_users"]:
            nick_name = name_dic.get(u, "")
            if nick_name == "":
                continue

            shared_users.append(f"{u} ({nick_name})")

    return (
        response["response"]["datasets"],
        response["response"]["embedding_model"],
        str(response["response"]["chunk_size"]),
        shared_users,
    )


def get_openai_config(owner: str) -> dict:
    """get openai api configs from st.seesion_state and openai config file."""

    data = {
        "owner":
        owner,
        "openai_key":
        st.session_state.openai_key if st.session_state.openai_on else "",
        "azure_key":
        st.session_state.azure_key if st.session_state.azure_openai_on else "",
        "azure_base":
        st.session_state.azure_base
        if st.session_state.azure_openai_on else "",
    }
    with st.spinner(SPINNER_MESSAGE):
        response = requests.get(api_urls["choose_openai"], json=data).json()

    if response["status"] != "success":
        api_fail(response["response"])
        return {}

    return response["response"]


def get_data_path(owner: str, datasets: list, embedding_model: str,
                  chunk_size: int):
    """get the chromadb path of all files in datasets.

    Args:
        owner (str): _description_
        datasets (list): _description_
        embedding_model (str): _description_
        chunk_size (int): _description_
    """
    try:
        with st.spinner(SPINNER_MESSAGE):
            chromadb_path = requests.get(
                api_urls["get_chromadb_path"]).json()["response"]

    except Exception as e:
        chromadb_path = "./chromadb"

    try:
        embed_type, embed_name = (
            embedding_model.split(":")[0].lower(),
            embedding_model.split(":")[1],
        )
    except:
        embed_type, embed_name = embedding_model.split(":")[0].lower(), ""

    res_list = []
    for dataset in datasets:
        dataset_name = dataset["name"]
        dataset_owner = dataset["owner"]
        dataset_files = dataset["files"]

        response = requests.get(
            api_urls["get_md5_name"],
            json={
                "owner": dataset_owner,
                "dataset_name": dataset_name
            },
        ).json()

        if response["status"] != "success":
            continue
        for file in dataset_files:
            if file in response["response"]:
                cur_path = Path(chromadb_path) / (
                    dataset_name + "_" +
                    file.split(".")[0].replace(" ", "").replace("_", "") +
                    "_" + response["response"][file] + "_" + embed_type + "_" +
                    embed_name.replace("/", "-") + "_" + str(chunk_size))
                res_list.append(cur_path.__str__())
    return res_list


def get_log_data() -> str:
    """change the log data in session state to plain text.

    Returns:
        str: log text data
    """
    plain_txt = ""
    for key in st.session_state.logs:
        plain_txt += key + ":\n"
        for k in st.session_state.logs[key]:
            if type(st.session_state.logs[key][k]) == list:
                text = (k + ": " + "\n".join(
                    [str(w) for w in st.session_state.logs[key][k]]) + "\n\n")
            else:
                text = k + ": " + str(st.session_state.logs[key][k]) + "\n\n"

            plain_txt += text
        plain_txt += "\n\n\n\n"

    return plain_txt


def download_txt(file_name: str):
    """download the log data in session state to plain text file.

    Args:
        file_name (str): _description_
    """
    file_name = "log_" + file_name + ".txt"
    txt_data = get_log_data()
    txt_filename = file_name
    st.download_button(
        "Download Text Log",
        txt_data.encode("utf-8"),
        key="txt",
        file_name=txt_filename,
        mime="text/plain",
    )
    # Path(f"./logs/{file_name}").unlink()


# Create a button to download a JSON file
def download_json(file_name: str):
    """download log data in session state to json file.

    Args:
        file_name (str): filename
    """
    import json

    file_name = "log_" + file_name + ".json"
    json_data = st.session_state.logs
    json_filename = file_name
    st.download_button(
        "Download JSON Log",
        json.dumps(json_data, indent=4, ensure_ascii=False).encode("utf-8"),
        key="json",
        file_name=json_filename,
        mime="application/json",
    )


def get_openai_from_file(username: str):

    response = requests.get(api_urls["load_openai"],
                            json={
                                "user_name": username
                            }).json()

    return response["response"]["openai_key"], response["response"][
        "azure_key"], response["response"]["azure_base"]


def run_command(command: str,
                capture_output: bool = False) -> subprocess.CompletedProcess:
    """Execute the command and return the result.

    Parameters
    ----------
    command : str
        The command to be executed.

    Returns
    -------
    subprocess.CompletedProcess
        The object (subprocess.CompletedProcess) after execution.
    """
    result = subprocess.run(command,
                            shell=True,
                            capture_output=capture_output,
                            text=True)

    return result


def is_default_api():

    try:
        response = requests.get(api_urls["is_default_api"]).json()
        if response["status"] != "success":
            raise Exception(response["response"])
    except Exception as e:
        api_fail(e.__str__())
        return False

    return response["response"]


def get_nickname(username: str) -> str:
    """get the nickname of the user.

    Args:
        username (str): user name

    Returns:
        str: nickname
    """
    response = requests.get(api_urls["get_nickname"],
                            json={
                                "user_name": username
                            }).json()

    if response["status"] != "success":
        api_fail(response["response"])
        return ""

    return response["response"]


def get_all_nicknames() -> dict:
    """get the nickname of the user.

    Args:
        username (str): user name

    Returns:
        str: nickname
    """
    response = requests.get(api_urls["get_nicknames"],
                            json={
                                "user_name": "all"
                            }).json()

    if response["status"] != "success":
        api_fail(response["response"])
        return {}

    return response["response"]


def get_other_users_names(username: str, all_users: list):
    """get all other users' usernames and nick names, output one list and a dictionary. one is the list of "username (nickname)", 
    the dictionary is the mapping of username and "username (nickname)".
    

    Args:
        username (str): current user name
        all_users (list): all users' names
    """

    other_users_nicknames = []
    users_mapping = {}
    nick_dic = get_all_nicknames()
    for u in all_users:
        if u == username:
            continue

        name_nick = f"{u} ({nick_dic.get(u, u)})"
        #other_users.append(u)
        other_users_nicknames.append(name_nick)
        users_mapping[name_nick] = u

    return other_users_nicknames, users_mapping


def save_tmp_file(uploaded_file: vars, username: str) -> str:
    """check the upload file and save it to the .uploaded directory in the user doc directory.

    Args:
        uploaded_file (vars): user uploaded file

    Returns:
        str: saved path of the file
    """
    try:
        save_path = requests.get(api_urls["get_docs_path"]).json()["response"]
    except Exception as e:
        save_path = "./docs/"
        api_fail("can not get docs path" + e.__str__())

    uploaded_path = save_path + f"/{username}/" + "/.uploaded/"
    ## check save_path exist or create the directory, and create the user directory
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + f"/{username}/", exist_ok=True)
    os.makedirs(uploaded_path, exist_ok=True)
    bytes_data = uploaded_file.read()

    if len(bytes_data) == 0:
        st.error(f"File={uploaded_file.name} is empty")
        return ""
    if len(bytes_data) > 100000000:
        st.error(f"File={uploaded_file.name} is too large")
        return ""
    file_path = uploaded_path + (uploaded_file.name)
    with open(Path(file_path), "wb") as f:
        f.write(bytes_data)

    delete_outdated_files(uploaded_path, 60 * 60 * 24 * 7)
    return file_path


def delete_outdated_files(directory: str, second_threshold: int):
    """Delete all files in the directory that were last modified more than `time` seconds ago.

    Args:
        directory (str): The directory to delete files from.
        time (int): The maximum file age in seconds.
    """
    now = time.time()

    for filename in os.listdir(directory):
        file_path = Path(directory) / filename
        if file_path.is_file():
            last_update = file_path.stat().st_mtime
            if now - last_update > second_threshold:
                file_path.unlink()

    return


def get_doc_file_path(dataset_owner, dataset_name, file_name) -> str:
    try:
        DOCS_PATH = requests.get(api_urls["get_docs_path"]).json()["response"]
    except Exception as e:
        api_fail(e.__str__())
        return ""

    return DOCS_PATH + f"/{dataset_owner}/{dataset_name}/{file_name}"

    # return (Path(DOCS_PATH) / dataset_owner / dataset_name /
    #         file_name).__fspath__()


def ask_summary(system_prompt: str, username: str, tmp_file_name: str,
                language_model: str, summary_type: str, summary_len: int):

    if (language_model.split(":")[0] == "openai"):
        openai_config = get_openai_config(username)
    else:
        openai_config = {}

    data = {
        "file_path": tmp_file_name,
        "language_model": language_model,
        "summary_type": summary_type,
        "summary_len": summary_len,
        "system_prompt": system_prompt,
        "openai_config": openai_config
    }

    try:
        with st.spinner(SPINNER_MESSAGE):
            response = requests.post(api_urls["get_summary"], json=data).json()
            if response["status"] != "success":
                api_fail(response["response"])
                return False

        # akasha: get response from expert

        st.session_state[
            "que"] = f"(Instruction: {system_prompt}) Summary of \"{tmp_file_name.split('/')[-1]}\":"
        st.session_state["ans"] = response["response"]
        st.session_state.logs[response["timestamp"]] = response["logs"]

    except Exception as e:
        api_fail(e.__str__())

    return True


def collect_logs(data: dict, response: str, fn_type: str):
    """since streaming can not get log from akasha, collect log from response and data.

    Args:
        data (dict): data dictionary of the request
        response (str): LLM response(after streaming)
    """

    cur_time = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
    log = data.copy()
    log['response'] = response
    log['fn_type'] = fn_type
    if "openai_config" in log:
        log.pop("openai_config")
    st.session_state.logs[cur_time] = log

    return


def clean():
    try:
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    except:
        pass
