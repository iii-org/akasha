import streamlit as st
from pathlib import Path
from streamlit_option_menu import option_menu
from interface.cot_page import cot_page
from interface.res_page import response_page
from interface.upload_file import up_load
st.set_page_config(layout="wide")
menu_list = ['Get Response','Chain Of Thoughts', 'Upload Files']

icon_list = ['chat-left-text', 'puzzle', 'upload']


with st.sidebar:
    user_menu = option_menu('AKASHA', menu_list, menu_icon='house',
        icons= icon_list, styles={"container": {"padding": "5!important", "background-color":"#fafafa"}, \
                "icon": {"color":"orange", "font-size": "25px"}, "nav-link":  {"font-size": "16px", "text-align": "left", "margin": "0px",
                "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#467500"} })
    



### variable ###
if 'embed_list' not in st.session_state:
    st.session_state.embed_list = ["openai:text-embedding-ada-002", "hf:shibing624/text2vec-base-chinese"]

if 'model_list' not in st.session_state:
    st.session_state.model_list = ["openai:gpt-3.5-turbo", "openai:gpt-3.5-turbo-16k","hf:model/Llama2-Chinese-13b-Chat-4bit",\
        "hf:model/Llama2-Chinese-7b-Chat", "llama-cpu:model/llama-2-13b-chat-hf.Q5_K_S.gguf",\
            "llama-gpu:model/llama-2-7b-chat.Q5_K_S.gguf"]

if 'search_list' not in st.session_state:
    st.session_state.search_list = ["merge", "svm", "tfidf", "mmr"]



if 'docs_path' not in st.session_state:
    st.session_state.docs_path = "./docs"
if 'response_list' not in st.session_state:
    st.session_state.response_list = []

if 'prompt_list' not in st.session_state:
    st.session_state.prompt_list = []

if 'docs_list' not in st.session_state:
    st.session_state.docs_list = []
    docs_dir = Path("./docs")
    for dir_path in docs_dir.iterdir():
        if dir_path.is_dir():
            st.session_state.docs_list.append(dir_path.name)
    
if 'n_text' not in st.session_state:
    st.session_state.n_text = 1

if 'openai_key' not in st.session_state:
    st.session_state.openai_key = ""
    

################

if user_menu == 'Get Response':
    response_page()

elif user_menu == 'Chain Of Thoughts':
    cot_page()
elif user_menu == 'Upload Files':
    up_load()