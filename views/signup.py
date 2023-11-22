import yaml
import re
import streamlit as st

USER_NAME_REGEX = r'^[a-z0-9]+$'
EMAIL_REGEX = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

def signup_page(url_params, authenticator, config, ACCOUNTS_PATH):
    if url_params['signup']:
        try:
            existed_users = list(config['credentials']['usernames'].keys())
            
            if authenticator.register_user('Sign-Up', preauthorization=False):
                st.session_state['register'] = True
                new_user_list = list(config['credentials']['usernames'].keys())
                # check new user name added
                new_user_ids = [u for u in new_user_list if u not in existed_users]
                if len(new_user_ids) == 0:
                    st.error('Fail to register new user.')
                else:
                    new_user_id = new_user_ids[0] 
                
                    # validate form input
                    valid = True
                    if not re.match(USER_NAME_REGEX, new_user_id):
                        st.error(f'User name can only contain lowercase alphabet and numbers.')
                        valid = False  
                    new_user_email = config['credentials']['usernames'][new_user_id]['email']
                    if not re.match(EMAIL_REGEX, new_user_email):
                        st.error('Invalid Email format.')
                        valid = False
                        
                    if valid:
                        # update config
                        with open(ACCOUNTS_PATH, 'w') as file:
                            # config['credentials']['usernames'][new_user_id] = {'email': new_user_email, 'name': , 'password': st.session_state['password']}
                            yaml.dump(config, file, default_flow_style=False)
                        st.success('User registered successfully, close this window for login')   
        except Exception as e:
            st.error(e)