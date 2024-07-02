import yaml
import re
import streamlit as st
import time
from streamlit_authenticator import Authenticate, Hasher
from utils import generate_verification_code, gmail_send_message

USER_NAME_REGEX = r'^[a-z0-9]+$'
EMAIL_REGEX = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
EMAIL_TITLE = 'akasha-lab Email Verification'
EMAIL_TEXT = 'Verification Code: '


class RegisterError(Exception):
    """
    Exceptions raised for the register user widget.

    Attributes
    ----------
    message: str
        The custom error message to display.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def register_user(form_name: str, authenticator: Authenticate,
                  new_email: str) -> bool:
    """
        Creates a register new user widget.

        Parameters
        ----------
        form_name: str
            The rendered name of the register new user form.
        location: str
            The location of the register new user form i.e. main or sidebar.
        preauthorization: bool
            The preauthorization requirement, True: user must be preauthorized to register, 
            False: any user can register.
        Returns
        -------
        bool
            The status of registering the new user, True: user registered successfully.
        """

    register_user_form = st.form('Register user')

    register_user_form.subheader(form_name)
    new_username = register_user_form.text_input('Username').lower()
    new_name = register_user_form.text_input('Name')
    new_password = register_user_form.text_input('Password', type='password')
    new_password_repeat = register_user_form.text_input('Repeat password',
                                                        type='password')

    if register_user_form.form_submit_button('Register'):
        if len(new_username) and len(new_name) and len(new_password) > 0:
            if new_username not in authenticator.credentials['usernames']:
                if new_password == new_password_repeat:

                    if not authenticator.validator.validate_username(
                            new_username):
                        raise RegisterError('Username is not valid')
                    if not authenticator.validator.validate_name(new_name):
                        raise RegisterError('Name is not valid')

                    authenticator.credentials['usernames'][new_username] = {
                        'name': new_name,
                        'password': Hasher([new_password]).generate()[0],
                        'email': new_email
                    }
                    return True
                else:
                    raise RegisterError('Passwords do not match')
            else:
                raise RegisterError('Username already taken')
        else:
            raise RegisterError('Please enter a username, name, and password')


def verify_email(authenticator: Authenticate, sender_mail: str,
                 sender_pass: str) -> str:

    if st.session_state.get('is_email_verified'):
        return
    else:
        st.session_state.is_email_verified = False
        register_user_form = st.form('Verify Email')
        register_user_form.subheader('Verify Email')

        new_email = register_user_form.text_input('Email',
                                                  value=st.session_state.email)
        if new_email != st.session_state.email:
            st.session_state.email = new_email

        if register_user_form.form_submit_button("Send Verification Code"):
            if not authenticator.validator.validate_email(
                    st.session_state.email):
                st.session_state.email_valid = False
                raise RegisterError('Email is not valid')
            else:
                st.session_state.email_valid = True

            st.session_state.verification_code = generate_verification_code()
            if gmail_send_message(
                    st.session_state.email, sender_mail, sender_pass,
                    EMAIL_TITLE,
                    EMAIL_TEXT + st.session_state.verification_code):
                st.info("Verification code sent. Please check your email.")
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


def signup_page(url_params: dict, authenticator: Authenticate, config,
                ACCOUNTS_PATH, sender_mail: str, sender_pass: str):
    if url_params['signup']:
        try:

            if st.session_state.get('is_email_verified'):

                existed_users = list(config['credentials']['usernames'].keys())

                ### register user ###
                if register_user('Sign-Up',
                                 authenticator=authenticator,
                                 new_email=st.session_state.email):
                    st.session_state['register'] = True

                    new_user_list = list(
                        config['credentials']['usernames'].keys())

                    # check new user name added
                    new_user_ids = [
                        u for u in new_user_list if u not in existed_users
                    ]
                    if len(new_user_ids) == 0:
                        st.error('Fail to register new user.')
                    else:
                        # update config
                        with open(ACCOUNTS_PATH, 'w') as file:
                            yaml.dump(config, file, default_flow_style=False)
                        st.success(
                            'User registered successfully, close this window for login'
                        )
                        st.session_state.is_email_verified = False
                        st.session_state.verification_code = ""
                        st.session_state.email = ""
                        st.session_state.email_valid = False
            else:
                verify_email(authenticator, sender_mail, sender_pass)
        except Exception as e:
            st.error(e)


def non_verify_signup_page(url_params, authenticator, config, ACCOUNTS_PATH):
    if url_params['signup']:
        try:
            existed_users = list(config['credentials']['usernames'].keys())

            if authenticator.register_user('Sign-Up', preauthorization=False):
                st.session_state['register'] = True
                st.button("verify")
                new_user_list = list(config['credentials']['usernames'].keys())
                # check new user name added
                new_user_ids = [
                    u for u in new_user_list if u not in existed_users
                ]
                if len(new_user_ids) == 0:
                    st.error('Fail to register new user.')
                else:
                    new_user_id = new_user_ids[0]

                    # validate form input
                    valid = True
                    if not re.match(USER_NAME_REGEX, new_user_id):
                        st.error(
                            f'User name can only contain lowercase alphabet and numbers.'
                        )
                        valid = False
                    new_user_email = config['credentials']['usernames'][
                        new_user_id]['email']
                    if not re.match(EMAIL_REGEX, new_user_email):
                        st.error('Invalid Email format.')
                        valid = False

                    if valid:
                        # update config
                        with open(ACCOUNTS_PATH, 'w') as file:
                            # config['credentials']['usernames'][new_user_id] = {'email': new_user_email, 'name': , 'password': st.session_state['password']}
                            yaml.dump(config, file, default_flow_style=False)
                        st.success(
                            'User registered successfully, close this window for login'
                        )
        except Exception as e:
            st.error(e)
