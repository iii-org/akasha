import streamlit as st
from pathlib import Path
from streamlit_option_menu import option_menu
from interface.cot_page import cot_page
from interface.res_page import response_page
from interface.upload_file import upload_page
from interface.setting import setting_page, set_model_dir
from interface.sum_page import summary_page
import akasha
import datetime
st.set_page_config(layout="wide")
menu_list = ['Get Response','Chain Of Thoughts',  'Summary', 'Upload Files', 'Setting',]

icon_list = ['chat-left-text', 'puzzle', 'chat-quote', 'upload', 'gear', ]


def get_log_data():
    
    plain_txt = ""
    for key in st.session_state.logs:
        plain_txt += key + ":\n"
        for k in st.session_state.logs[key]:
            if type(st.session_state.logs[key][k]) == list:
                text = k + ": " + '\n'.join([str(w) for w in st.session_state.logs[key][k]]) + "\n\n"             
            else:
                text = k + ": " + str(st.session_state.logs[key][k]) + "\n\n"
            
            plain_txt += text
        plain_txt += "\n\n\n\n"
        
    return plain_txt
    
    
    
def download_txt(file_name:str):
    file_name = "log_" + file_name + ".txt"
    txt_data = get_log_data()
    txt_filename = file_name
    st.download_button(
        "Download Text Log",
        txt_data.encode('utf-8'),
        key='txt',
        file_name=txt_filename,
        mime='text/plain'
    )
    #Path(f"./logs/{file_name}").unlink()

# Create a button to download a JSON file
def download_json(file_name:str):
    import json
    file_name =  "log_" + file_name + ".json"
    json_data = st.session_state.logs
    json_filename = file_name
    st.download_button(
        "Download JSON Log",
        json.dumps(json_data,indent=4,ensure_ascii=False).encode('utf-8'),
        key='json',
        file_name=json_filename,
        mime='application/json'
    )

### variable ###
if 'embed_list' not in st.session_state:
    st.session_state.embed_list = ["openai:text-embedding-ada-002", "hf:shibing624/text2vec-base-chinese"]
    
if 'mdl_dir' not in st.session_state:
    st.session_state.mdl_dir = "model"
    
if 'model_list' not in st.session_state:
    set_model_dir()
    # st.session_state.model_list = ["openai:gpt-3.5-turbo", "openai:gpt-3.5-turbo-16k","hf:model/Llama2-Chinese-13b-Chat-4bit",\
    #     "hf:model/Llama2-Chinese-7b-Chat", "llama-cpu:model/llama-2-13b-chat-hf.Q5_K_S.gguf",\
    #         "llama-gpu:model/llama-2-7b-chat.Q5_K_S.gguf"]

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
    docs_dir = Path(st.session_state.docs_path)
    for dir_path in docs_dir.iterdir():
        if dir_path.is_dir():
            st.session_state.docs_list.append(dir_path.name)
        
if 'n_text' not in st.session_state:
    st.session_state.n_text = 1

if 'openai_key' not in st.session_state:
    st.session_state.openai_key = ""

# if 'hugging_face_key' not in st.session_state:
#     st.session_state.hugging_face_key = ""

if 'select_idx' not in st.session_state:
    st.session_state.select_idx = [0,0,0,0]


### function argument ###
if 'chose_doc_path' not in st.session_state:
    if len(st.session_state.docs_list) > 0:
        st.session_state.chose_doc_path = st.session_state.docs_path + '/'  + st.session_state.docs_list[0]
    else:
        st.info("Please upload your documents first.",icon="🚨")

if 'embed' not in st.session_state:
    st.session_state.embed = st.session_state.embed_list[0]
if 'model'  not in st.session_state:
    st.session_state.model = st.session_state.model_list[0]

if 'chunksize'  not in st.session_state:
    st.session_state.chunksize = 500
if 'topK'  not in st.session_state:
    st.session_state.topK = 2

if 'search_type' not in st.session_state:
    st.session_state.search_type = st.session_state.search_list[0]
    
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.2

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.0
    
if 'max_token' not in st.session_state:
    st.session_state.max_token = 2500
if 'sys_prompt' not in st.session_state:
    st.session_state.sys_prompt = ""
if 'logs' not in st.session_state:
    st.session_state.logs = {}

if 'akasha_obj' not in st.session_state:
    st.session_state.akasha_obj = ""

################






with st.sidebar:
    user_menu = option_menu('akasha', menu_list, menu_icon='house',
        icons= icon_list, styles={"container": {"padding": "5!important",}, \
                "icon": {"color":"orange", "font-size": "25px"}, "nav-link":  {"font-size": "16px", "text-align": "left", "margin": "0px",
                },
                })
    
    st.session_state.openai_key = st.text_input("OpenAI Key", type="password")
    # st.session_state.hugging_face_key = st.text_input("Hugging Face Key", type="password")
    
    
    st.markdown('##')
    st.markdown('##')
    
    if st.button("Download Log", type="primary", use_container_width=True):
    #    file_name = get_log_file()
    
        file_date = datetime.datetime.now().strftime( "%Y-%m-%d-%H-%M-%S")
        tx, js = st.columns([1,1])
        with tx:
            download_txt(file_date)
        with js:
            download_json(file_date)
if user_menu == 'Get Response':
    response_page()

elif user_menu == 'Chain Of Thoughts':
    cot_page()
elif user_menu == 'Upload Files':
    upload_page()
elif user_menu == 'Setting':
    setting_page()
elif user_menu == 'Summary':
    summary_page()