import streamlit as st
import yaml
from utils import save_api_configs

def settings_page(authenticator, username, config, ACCOUNTS_PATH):
    st.title('Settings')
    settings_option = st.radio('Option', ['API Settings', 'History', 'Reset Password'], horizontal=True, label_visibility='collapsed')
    st.markdown('')
    
    if settings_option == 'API Settings':
        _api_settings()
    elif settings_option == 'History':
        _history()
        
    elif settings_option == 'Reset Password':
        _reset_password(authenticator, username, config, ACCOUNTS_PATH)
            
def _api_settings():
    st.subheader('* Open AI', divider='grey')
    openai_on = st.toggle('Use Open AI', value=False, key='openai') 
    openai_api_key = st.text_input('OpenAI Key', help='OpenAI Key', type='password', disabled=not openai_on)
    
    st.subheader('* Azure Open AI', divider='grey')
    azure_openai_on = st.toggle('Use Azure OpenAI', value=False, key='azure_openai')
    col_azure_key, col_azure_url = st.columns([1, 1])
    azure_openai_api_key = col_azure_key.text_input('Azure OpenAI Key', help='Azure OpenAI Key', type='password', disabled=not azure_openai_on)
    azure_openai_base_url = col_azure_url.text_input('Azure OpenAI Base URL', help='Azure OpenAI Base URL', type='password', disabled=not azure_openai_on)
    
    st.markdown('')
    st.markdown('')
    st.markdown('')
    save_api_config_button = st.button('Save', f'btn-save-api-configs', use_container_width=True, type='primary',
                                    on_click=save_api_configs, 
                                    args=(openai_on, azure_openai_on, 
                                          openai_api_key if openai_on else None, 
                                          azure_openai_api_key if azure_openai_on else None, 
                                          azure_openai_base_url if azure_openai_on else None))
    
def _history():
    st.subheader('History', divider='rainbow')
    st.button('Download History', f'btn-download-history', type='secondary')
    
def _reset_password(authenticator, username, config, ACCOUNTS_PATH):
    # reset password  
    try:
        if authenticator.reset_password(username, 'Reset Password', 'main'):
            st.success('Password modified successfully')
            with open(ACCOUNTS_PATH, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.sidebar.error(e)