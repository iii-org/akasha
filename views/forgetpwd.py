import yaml
import streamlit as st
from streamlit_authenticator import Authenticate, Hasher
import time
from utils import generate_verification_code, gmail_send_message

EMAIL_TITLE = 'akasha-lab Email Verification'
EMAIL_TEXT = 'Verification Code: '


class ForgotError(Exception):
    """
    Exceptions raised for the forgotten username/password widgets.

    Attributes
    ----------
    message: str
        The custom error message to display.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def forgot_password(authenticator: Authenticate, form_name: str,
                    username: str) -> tuple:
    """
    Creates a forgot password widget.

    Parameters
    ----------
    form_name: str
        The rendered name of the forgot password form.
    location: str
        The location of the forgot password form i.e. main or sidebar.
    Returns
    -------
    str
        Username associated with forgotten password.
    str
        Email associated with forgotten password.
    str
        New plain text password that should be transferred to user securely.
    """

    forgot_password_form = st.form('Enter new password')

    forgot_password_form.subheader(form_name)
    password = forgot_password_form.text_input('Password', type='password')
    repeat_password = forgot_password_form.text_input('Repeat Password',
                                                      type='password')

    if forgot_password_form.form_submit_button('Submit'):
        if len(password) > 0:
            if password == repeat_password:
                authenticator.credentials['usernames'][username][
                    'password'] = Hasher([password]).generate()[0]
                return username, authenticator.credentials['usernames'][
                    username]['email'], password
            else:
                raise ForgotError('Passwords do not match')
        else:
            raise ForgotError('Password not provided')
    return None, None, None


def verify_email(authenticator: Authenticate, sender_mail: str,
                 sender_pass: str) -> str:

    if st.session_state.get('is_email_verified'):
        return
    else:
        st.session_state.is_email_verified = False
        register_user_form = st.form('Verify Email')
        register_user_form.subheader('Verify Email')

        forg_username = register_user_form.text_input(
            'Username', value=st.session_state.forg_username)

        if forg_username != st.session_state.forg_username:
            st.session_state.forg_username = forg_username

        if register_user_form.form_submit_button("Send Verification Code"):
            if len(st.session_state.forg_username) > 0:
                if st.session_state.forg_username in authenticator.credentials[
                        'usernames']:
                    st.session_state.email = authenticator.credentials[
                        'usernames'][st.session_state.forg_username]['email']
                    st.session_state.email_valid = True
                else:
                    st.session_state.email_valid = False
                    raise ForgotError('Username not found')
            else:
                st.session_state.email_valid = False
                raise ForgotError('Username not provided')

            st.session_state.verification_code = generate_verification_code()
            if gmail_send_message(
                    st.session_state.email, sender_mail, sender_pass,
                    EMAIL_TITLE,
                    EMAIL_TEXT + st.session_state.verification_code):
                mask_len = len(st.session_state.email.split('@')[0]) // 2
                mask_email = st.session_state.email[:mask_len] + ''.join(
                    '*' for _ in range(
                        len(st.session_state.email.split('@')[0]) - mask_len)
                ) + st.session_state.email[st.session_state.email.index('@'):]
                st.info(
                    f"Verification code sent. Please check your email at {mask_email}."
                )
            else:
                st.error("Failed to send verification code. Please try again.")

        user_code = register_user_form.text_input("Enter Verification Code")
        if register_user_form.form_submit_button("Verify Code"):
            if not st.session_state.email_valid:
                st.error("Please send verification code first.")

            elif user_code == st.session_state['verification_code']:
                st.success("Email verified.")
                st.session_state.is_email_verified = True
                time.sleep(1.5)
                st.rerun()
            else:
                st.error("Incorrect verification code.")


def forgetpwd_page(url_params, authenticator: Authenticate, config,
                   ACCOUNTS_PATH, sender_mail: str, sender_pass: str):
    if url_params['forgetpwd']:
        try:

            if st.session_state.get('is_email_verified'):

                username_forgot_pw, email_forgot_password, new_password = forgot_password(
                    authenticator, 'Enter New Password',
                    st.session_state.forg_username)
                if username_forgot_pw is None:
                    pass
                elif username_forgot_pw != '':
                    if username_forgot_pw in list(
                            config['credentials']['usernames'].keys()):
                        with open(ACCOUNTS_PATH, 'w') as file:
                            yaml.dump(config, file, default_flow_style=False)
                        st.success('Password changed successfully.')
                        st.session_state.is_email_verified = False
                        st.session_state.verification_code = ""
                        st.session_state.email = ""
                        st.session_state.email_valid = False
                        st.session_state.forg_username = ""

                    else:
                        st.error('Username not found')
                else:
                    st.error('Username not found')
            else:
                verify_email(authenticator, sender_mail, sender_pass)
        except Exception as e:
            st.error(e)


def non_verify_forgetpwd_page(url_params, authenticator: Authenticate, config,
                              ACCOUNTS_PATH):
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
