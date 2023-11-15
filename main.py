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

from utils import list_experts, list_datasets

# info
VERSION = '0.8'
HOST = os.getenv('HOST', 'localhost') 
PORT = os.getenv('PORT', '8501')

# page config
st.set_page_config(
    page_title="Akasha",
    layout="wide",
)

# load existed accounts and initialize authentication
ACCOUNTS_PATH = os.path.join('.', 'accounts.yaml')
with open(ACCOUNTS_PATH) as file:
    config = yaml.load(file, Loader=SafeLoader)   
    
authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# get params from browser url
url_params = st.experimental_get_query_params()

_, col_title, _ = st.columns([2, 6, 1])
placeholder_title = col_title.empty()
with placeholder_title:
    st.title('_:rainbow[AKASHA: Your Personal Domain Expert]_')
    
_, col_version, _ = st.columns([6, 1, 6])
placeholder_version = col_version.empty()
with placeholder_version:
    st.caption(f'version {VERSION}') 

# login page
if url_params == {}:

    name, authentication_status, username = authenticator.login('Login', 'main')
    
    # signup/forgot password
    _, col_signup, col_forgetpwd = st.columns([7, 1, 1])
    placeholder_signup = col_signup.empty()
    placeholder_forget = col_forgetpwd.empty()
    with placeholder_signup:
        st.markdown(f'[Sign Up](http://{HOST}:{PORT}/?signup=True)')
    with placeholder_forget:
        st.markdown(f'[Forget Password](http://{HOST}:{PORT}/?forgetpwd=True)')
    
    # authenticate
    if authentication_status is False:
        st.error('Username/password is incorrect')
    elif authentication_status:
        
        # load configurations
        EXPERTS = list_experts(username, name_only=True, include_shared=True) # may filtered by logged-in user
        DATASETS = list_datasets(username, name_only=True, include_shared=True) # may filtered by logged-in user
        EMBEDDING_MODELS = ['embedding_model_A', 'embedding_model_B', 'embedding_model_C']
        SEARCH_TYPES = ['merge', 'svm', 'tfidf', 'mmr']
        LANGUAGE_MODELS = ['gpt2', 'gpt3']
        
        # layout after successfully login
        placeholder_title.empty()
        placeholder_signup.empty()
        placeholder_forget.empty()
        placeholder_version.empty()
        with st.sidebar:
            st.markdown("<h1 style='text-align: center; color: black;'>AKASHA</h1>", unsafe_allow_html=True)
            st.markdown("<h5 style='text-align: center; color: black;'>Your Personal Domain Expert</h1>", unsafe_allow_html=True)
            selected = option_menu(f'Hi, {username}', 
                                    ['Consult', 'Experts', 'Datasets', 'Settings', 'User Guide'], 
                                    icons=['chat-dots', 'lightbulb', 'file-earmark-arrow-up', 'gear', 'book-half'], 
                                    menu_icon='house', 
                                    default_index=0)
            placeholder_hint = st.empty()
                
            authenticator.logout('Logout', location='main', key='logout_key')
            st.caption('©️2023 Institute for Information Industry')
            st.caption(f'version {VERSION}')
        
        user_accounts = config['credentials']['usernames']    
        if selected == 'Consult':
            consult_page(placeholder_hint, EXPERTS, SEARCH_TYPES, LANGUAGE_MODELS, username)
            
        elif selected == 'Experts':
            experts_page(EXPERTS, EMBEDDING_MODELS, DATASETS, username, user_accounts)
            
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
    