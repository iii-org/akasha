import yaml
import os

import streamlit as st
from streamlit_authenticator import Authenticate
from streamlit_option_menu import option_menu
from yaml.loader import SafeLoader
from views.consult import consult_page
from views.experts import experts_page
from views.dataset import dataset_page
from views.settings import settings_page
from views.userguide import user_guide_page
from views.signup import signup_page
from views.forgetpwd import forgetpwd_page
from utils import list_experts, list_datasets, list_models, get_openai_from_file, run_command

# info
VERSION = '0.14'

# get host ip
if 'host_ip' not in st.session_state:
    tmp = run_command('curl ifconfig.me', capture_output=True)
    st.session_state.host_ip = tmp.stdout

HOST = os.getenv('HOST', st.session_state.host_ip)
PORT = os.getenv('PORT', '8501')
USE_PREFIX = os.getenv('USE_PREFIX', False)
PREFIX = os.getenv('PREFIX', "")

# session state parameters
if 'openai_on' not in st.session_state:
    st.session_state.openai_on = False
if 'azure_openai_on' not in st.session_state:
    st.session_state.azure_openai_on = False

if 'sum_dataset_on' not in st.session_state:
    st.session_state.sum_dataset_on = True

if 'logs' not in st.session_state:
    st.session_state['logs'] = {}
if 'que' not in st.session_state:
    st.session_state['que'] = ''

if 'ans' not in st.session_state:
    st.session_state['ans'] = ''

if 'history_messages' not in st.session_state:
    st.session_state['history_messages'] = []

# page config
st.set_page_config(
    page_title="akasha",
    layout="wide",
)

# load existed accounts and initialize authentication
ACCOUNTS_PATH = os.path.join('.', 'accounts.yaml')
with open(ACCOUNTS_PATH) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = Authenticate(config['credentials'], config['cookie']['name'],
                             config['cookie']['key'],
                             config['cookie']['expiry_days'],
                             config['preauthorized'])

# get params from browser url
url_params = st.query_params.to_dict()

_, col_title, _ = st.columns([2, 6, 1])
placeholder_title = col_title.empty()
with placeholder_title:
    st.title('_:rainbow[akasha: Your Personal Domain Expert]_')

_, col_version, _ = st.columns([6, 1, 6])
placeholder_version = col_version.empty()
with placeholder_version:
    st.caption(f'version {VERSION}')

# login page
if url_params == {}:

    name, authentication_status, username = authenticator.login(
        'Login', 'main')

    # signup/forgot password
    _, col_signup, col_forgetpwd = st.columns([7, 1, 1])
    placeholder_signup = col_signup.empty()
    placeholder_forget = col_forgetpwd.empty()
    with placeholder_signup:
        if (not USE_PREFIX) or USE_PREFIX == "false":
            st.markdown(f'[Sign Up](http://{HOST}:{PORT}/?signup=True)')
        else:
            st.markdown(f'[Sign Up](http://{HOST}/{PREFIX}/?signup=True)')
    with placeholder_forget:
        if (not USE_PREFIX) or USE_PREFIX == "false":
            st.markdown(
                f'[Forget Password](http://{HOST}:{PORT}/?forgetpwd=True)')
        else:
            st.markdown(
                f'[Forget Password](http://{HOST}/{PREFIX}/?forgetpwd=True)')

    # authenticate
    if authentication_status is False:
        st.error('Username/password is incorrect')
    elif authentication_status:

        # load configurations
        if ('openai_key' not in st.session_state) or (
                'azure_key'
                not in st.session_state) or ('azure_base'
                                             not in st.session_state):
            st.session_state.save_openai = True
            st.session_state.openai_key, st.session_state.azure_key, st.session_state.azure_base = get_openai_from_file(
                username)
            if st.session_state.openai_key != "" or (
                    st.session_state.azure_key != ""
                    and st.session_state.azure_base != ""):
                st.session_state.save_openai = False

        EXPERTS = list_experts(
            username, name_only=True,
            include_shared=True)  # may filtered by logged-in user
        DATASETS = list_datasets(
            username, name_only=True,
            include_shared=True)  # may filtered by logged-in user
        EMBEDDING_MODELS = ['openai:text-embedding-ada-002', 'hf:shibing624/text2vec-base-chinese-paraphrase', \
            'hf:shibing624/text2vec-base-multilingual',"hf:BAAI/bge-large-en-v1.5", "hf:BAAI/bge-large-zh-v1.5"]
        SEARCH_TYPES = ['merge', 'svm', 'auto', 'tfidf', 'mmr', 'bm25']
        LANGUAGE_MODELS = list_models()

        # layout after successfully login
        placeholder_title.empty()
        placeholder_signup.empty()
        placeholder_forget.empty()
        placeholder_version.empty()
        with st.sidebar:
            st.markdown(
                "<h1 style='text-align: center; color: black;'>akasha</h1>",
                unsafe_allow_html=True)
            st.markdown(
                "<h5 style='text-align: center; color: black;'>Your Personal Domain Expert</h1>",
                unsafe_allow_html=True)
            selected = option_menu(f'Hi, {name}', [
                'Consult', 'Knowledges', 'Datasets', 'Settings', 'User Guide'
            ],
                                   icons=[
                                       'chat-dots', 'lightbulb',
                                       'file-earmark-arrow-up', 'gear',
                                       'book-half'
                                   ],
                                   menu_icon='house',
                                   default_index=0)
            placeholder_hint = st.empty()

            authenticator.logout('Logout', location='main', key='logout_key')
            st.caption('©️2024 Institute for Information Industry')
            st.caption(f'version {VERSION}')

        user_accounts = config['credentials']['usernames']
        if selected == 'Consult':
            consult_page(DATASETS, EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS,
                         username)

        elif selected == 'Knowledges':
            experts_page(EXPERTS, EMBEDDING_MODELS, DATASETS, username,
                         user_accounts)

        elif selected == 'Datasets':
            dataset_page(DATASETS, username, user_accounts)

        elif selected == 'Settings':
            settings_page(authenticator, username, config, ACCOUNTS_PATH)

        elif selected == 'User Guide':
            user_guide_page()

# forget password page
if 'forgetpwd' in url_params.keys():
    forgetpwd_page(url_params, authenticator, config, ACCOUNTS_PATH)

# sign-up page
if 'signup' in url_params.keys():
    signup_page(url_params, authenticator, config, ACCOUNTS_PATH)
