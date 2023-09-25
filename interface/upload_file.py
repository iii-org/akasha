import streamlit as st
from pathlib import Path

def up_load():
    
    st.title("Upload Files")
    st.markdown('##')
    st.markdown('##')
    
    
    uploaded_files = st.file_uploader("Upload document files", accept_multiple_files=True,type=['txt', 'pdf', 'docx'])
    
    path_name = st.text_input("path name")
    sb1,sb2,sb3 = st.columns([1,1,1])
    with sb2:
        submit_but = st.button("Submit", type="primary",use_container_width=True)
    if submit_but:
        if path_name in st.session_state.docs_list:
            st.error("Path name already exist, please use another path name.")
        
        else:
            # create path
            if path_name[-1]!='/':
                path_name = path_name + '/'
            save_path = Path(st.session_state.docs_path, path_name)
            # check if path exist
            if not save_path.exists():
                save_path.mkdir(parents=True)
                st.session_state.docs_list.append(path_name[:-1])    
            ### write file to doc folder ###
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                
                with open(save_path.joinpath(uploaded_file.name), "wb") as f:
                    f.write(bytes_data)

                st.write("uploaded file:", uploaded_file.name)
        
        print(st.session_state.docs_list)