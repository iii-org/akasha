import yaml
import streamlit as st


def forgetpwd_page(url_params, authenticator, config, ACCOUNTS_PATH):
    if url_params['forgetpwd']:
        try:
            username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password(
                'Forgot password')
            if username_forgot_pw is None:
                pass
            elif username_forgot_pw != '':
                if username_forgot_pw in list(
                        config['credentials']['usernames'].keys()):
                    with open(ACCOUNTS_PATH, 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)
                    st.success(
                        f'Your new password is "{random_password}", remember to change your password after login.'
                    )
                else:
                    st.error('Username not found')
            else:
                st.error('Username not found')
        except Exception as e:
            st.error(e)
