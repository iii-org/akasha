import streamlit as st
import yaml
from utils import save_api_configs, save_openai_to_file, save_single_api_configuration, save_single_api_to_file
from utils import download_txt, download_json
from utils import is_default_api


def settings_page(authenticator, username, config, ACCOUNTS_PATH):
    st.title('Settings')
    if is_default_api():
        settings_option = st.radio('Option', ['History', 'Account'],
                                   horizontal=True,
                                   label_visibility='collapsed')
        st.markdown('')
        if settings_option == 'History':
            _history()

        elif settings_option == 'Account':
            _account_settings(authenticator, username, config, ACCOUNTS_PATH)
    else:
        settings_option = st.radio('Option',
                                   ['API Settings', 'History', 'Account'],
                                   horizontal=True,
                                   label_visibility='collapsed')
        st.markdown('')
        if settings_option == 'API Settings':
            _api_settings(username)
        elif settings_option == 'History':
            _history()

        elif settings_option == 'Account':
            _account_settings(authenticator, username, config, ACCOUNTS_PATH)


def _api_settings(username: str):

    st.header('API Settings', divider='rainbow')
    api_option = st.radio('Option',
                          ['OpenAI API', 'Gemini API', 'Anthropic API'],
                          horizontal=True,
                          label_visibility='collapsed')

    st.markdown('')
    if api_option == 'OpenAI API':
        _openai_settings(username)
    elif api_option == 'Gemini API':
        _gemini_settings(username)
    elif api_option == 'Anthropic API':
        _anthropic_settings(username)


def _gemini_settings(username: str):

    st.header('Gemini API Settings', divider='rainbow')

    # gemini setting ##

    gemini_api_key = st.text_input('Gemini Key',
                                   help='Gemini Key',
                                   type='password',
                                   value=st.session_state.gemini_key)

    st.markdown('')
    st.markdown('')
    st.markdown('')
    save_gemini_config, save_gemini_file, sp___ce = st.columns([1, 3, 1])
    res = False
    with save_gemini_config:
        st.session_state.save_gemini = st.toggle(
            'Save Permanently',
            value=st.session_state.save_gemini,
            key='gemini_save_file')

    with save_gemini_file:

        ### if submit, first check if the api keys are valid, if not, show error; if yes, check if need to save to file
        if st.button('Save',
                     f'btng-save-api-configs',
                     use_container_width=True,
                     type='primary'):

            res = save_single_api_configuration(gemini_api_key, 'gemini')

            if res and st.session_state.save_gemini:
                save_single_api_to_file(username, gemini_api_key, 'gemini')


def _anthropic_settings(username: str):
    st.header('Anthropic API Settings', divider='rainbow')

    # anthropic setting ##

    anthropic_api_key = st.text_input('Anthropic Key',
                                      help='Anthropic Key',
                                      type='password',
                                      value=st.session_state.anthropic_key)

    st.markdown('')
    st.markdown('')
    st.markdown('')
    save_anthropic_config, save_anthropic_file, sp___ce = st.columns([1, 3, 1])
    res = False
    with save_anthropic_config:
        st.session_state.save_anthropic = st.toggle(
            'Save Permanently',
            value=st.session_state.save_anthropic,
            key='anthropic_save_file')

    with save_anthropic_file:

        ### if submit, first check if the api keys are valid, if not, show error; if yes, check if need to save to file
        if st.button('Save',
                     f'btnanth-save-api-configs',
                     use_container_width=True,
                     type='primary'):

            res = save_single_api_configuration(anthropic_api_key, 'anthropic')

            if res and st.session_state.save_anthropic:
                save_single_api_to_file(username, anthropic_api_key,
                                        'anthropic')


def _openai_settings(username: str):

    st.header('OpenAI API Settings', divider='rainbow')
    st.subheader('* Open AI', divider='grey')

    # openai setting ##
    tmp_openai_on = st.toggle(
        'Use Open AI',
        value=st.session_state.openai_on,
    )
    if tmp_openai_on != st.session_state.openai_on:
        st.session_state.openai_on = tmp_openai_on
        if st.session_state.openai_on and st.session_state.azure_openai_on:
            st.session_state.azure_openai_on = False
        st.rerun()
    openai_api_key = st.text_input('OpenAI Key', help='OpenAI Key', type='password', disabled=not st.session_state.openai_on,\
        value=st.session_state.openai_key)

    # azure openai setting ##
    st.subheader('* Azure Open AI', divider='grey')
    tmp_azure_openai_on = st.toggle('Use Azure OpenAI',
                                    value=st.session_state.azure_openai_on,
                                    key='azure_openai')
    if tmp_azure_openai_on != st.session_state.azure_openai_on:
        st.session_state.azure_openai_on = tmp_azure_openai_on
        if st.session_state.openai_on and st.session_state.azure_openai_on:
            st.session_state.openai_on = False
        st.rerun()

    col_azure_key, col_azure_url = st.columns([1, 1])
    azure_openai_api_key = col_azure_key.text_input('Azure OpenAI Key', help='Azure OpenAI Key', type='password', disabled=not st.session_state.azure_openai_on,\
        value=st.session_state.azure_key)
    azure_openai_base_url = col_azure_url.text_input('Azure OpenAI Base URL', help='Azure OpenAI Base URL', type='password',\
        disabled=not st.session_state.azure_openai_on, value= st.session_state.azure_base)

    st.markdown('')
    st.markdown('')
    st.markdown('')
    save_config, save_file, sp___ce = st.columns([1, 3, 1])
    res = False
    with save_config:
        st.session_state.save_openai = st.toggle(
            'Save Permanently',
            value=st.session_state.save_openai,
            key='openai_save_file')

    with save_file:

        ### if submit, first check if the api keys are valid, if not, show error; if yes, check if need to save to file
        if st.button('Save',
                     f'btn-save-api-configs',
                     use_container_width=True,
                     type='primary'):

            res = save_api_configs(
                st.session_state.openai_on, st.session_state.azure_openai_on,
                openai_api_key if st.session_state.openai_on else None,
                azure_openai_api_key if st.session_state.azure_openai_on else
                None, azure_openai_base_url
                if st.session_state.azure_openai_on else None)

            if res and st.session_state.save_openai:
                #if st.button('Save to File', f'btn-save-api-configs-to-file', use_container_width=True, type='primary',disabled= not res):
                save_openai_to_file(
                    username, st.session_state.openai_on,
                    st.session_state.azure_openai_on,
                    openai_api_key if st.session_state.openai_on else None,
                    azure_openai_api_key if st.session_state.azure_openai_on
                    else None, azure_openai_base_url
                    if st.session_state.azure_openai_on else None)


def _history():
    st.header('History', divider='rainbow')
    #dl_history = st.button('Download History', f'btn-download-history', type='secondary')
    #if dl_history:
    tx, js = st.columns([1, 1])
    with tx:
        download_txt('')
    with js:
        download_json('')


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
    col_password.text_input('Password',
                            help='Password',
                            type='password',
                            placeholder='Type your password to verify',
                            key='delete-account-password',
                            label_visibility='collapsed')
    col_delete.button('Delete Account', f'btn-delete-account', type='primary')
