import streamlit as st
import yaml
from utils import save_api_configs

def settings_page(authenticator, username, config, ACCOUNTS_PATH):
    st.title('Settings')
    settings_option = st.radio('Option', ['API Settings', 'History', 'Account'], horizontal=True, label_visibility='collapsed')
    st.markdown('')
    
    if settings_option == 'API Settings':
        _api_settings()
    elif settings_option == 'History':
        _history()
        
    elif settings_option == 'Account':
        _account_settings(authenticator, username, config, ACCOUNTS_PATH)
            
def _api_settings():
    st.header('API Settings', divider='rainbow')
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
    st.header('History', divider='rainbow')
    st.button('Download History', f'btn-download-history', type='secondary')
    
def _account_settings(authenticator, username, config, ACCOUNTS_PATH):
    st.header('Account', divider='rainbow')
    # reset password  
    st.subheader('* Reset Password', divider='grey')  
    try:
        if authenticator.reset_password(username, '', 'main'):
            st.success('Password modified successfully')
            with open(ACCOUNTS_PATH, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.sidebar.error(e)
    # delete account
    st.subheader('* Delete Account', divider='grey')
    col_password, col_delete = st.columns([4, 1])
    col_password.text_input('Password', help='Password', type='password',
                            placeholder='Type your password to verify',
                            key='delete-account-password',
                            label_visibility='collapsed')
    col_delete.button('Delete Account', f'btn-delete-account', type='primary')