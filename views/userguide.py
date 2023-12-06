import streamlit as st

def user_guide_page():
    st.title('User Guide')
            
    st.header('📖 Terminology', divider='rainbow')
    with st.expander('Terminology', expanded=False):
        st.markdown('1. Datasets: a collection of documents.')
        st.markdown('2. Experts: a collection of knowledge bases based on dataset(s).')
        st.markdown('3. Embedding (Model): a model to convert contents (of documents) into knowledge bases.')
        st.markdown('4. Chunk Size: maximal batch number of characters in content-convertion.')
        st.markdown('5. Search Type: the type of search algorithm to be used when searching for similar contents with question in knowledge bases.')
        st.markdown('6. Top K: the maximal number of content results to be returned, which is the most similar/relative to the question.')
        st.markdown('7. Threshold: the least threshold of similarity between question and answer.')
        st.markdown('8. Max Token: the maximal number of tokens in question (include prompt & system prompt).')
        st.markdown('9. Temperature: the temperature of language model, usually represents the degrees of freedom/imagination when the model answers.')
        st.markdown('10. Compression: compress the question/documents into shorter contents.')
    
    st.header('📌 Regulations', divider='rainbow')
    with st.expander('Regulations', expanded=False):
        st.markdown('1. Each dataset must contain at least 1 valid file.')
        st.markdown('2. Each expert must include at least 1 valid dataset.')
        st.markdown("3. Delete file(s) from dataset will update all experts' knowledge referencing the dataset.")
        st.markdown('4. Delete dataset will disable all experts referencing the dataset, if it results in experts without any valid dataset, those experts will be deleted.')
        st.markdown('5. Delete expert will not delete the dataset it is referencing.')
        
    st.header('📁 Datasets', divider='rainbow')
    with st.expander('Datasets', expanded=False):
        st.subheader('1. My Datasets', divider='grey')
        st.text('"My Datasets" page lists all datasets created by the user.')
        st.text('If you open "Shared Datasets", you can see all datasets shared to you. The name format is "dataset_name@dataset_owner".')
        st.text('You can use delete button to delete your own datasets, but you cannot delete the dataset shared to you.')
        
        
        st.subheader('2. New Dataset', divider='grey')
        st.text('"New Dataset" page allow user to create a new dataset, noted that the dataset name can not be empty or duplicated.')
        st.text('You can upload files by clicking "Browse files" button, you can not create a dataset without any files.')
        st.text('You can also share your dataset with other users by choosing the usernames you want to share.')
        
        
        st.subheader('3. Update Dataset', divider='grey')
        st.text('"Update Dataset" allow user to update their own datasets.')
        
        
    st.header('👑 Experts', divider='rainbow')
    with st.expander('Experts', expanded=False):
        st.subheader('1. My Experts', divider='grey')
        st.text('"My Experts" page lists all experts created by the user.')
        st.text('If you open "Shared Experts", you can see all experts shared to you. The name format is "expert_name@expert_owner".')
        st.text('You can use delete button to delete your own experts, but you cannot delete the expert shared to you.')
        
        
        
        st.subheader('2. New Expert', divider='grey')
        st.text('"New Expert" page allow user to create a new expert by choosing the expertname, embedding model, chunk size and datasets.')
        st.text('You need to add openai config in setting page before creating an expert if you want to use openai embedding model.')
        
        
        
        
        st.subheader('3. Update Expert', divider='grey')
        st.text('"Update Expert" allow user to update their own experts.')
        
    st.header('🧭 Consult', divider='rainbow')
    with st.expander('Consult', expanded=False):
        st.subheader('1. Regular Consult', divider='grey')
        st.text('You can choose the expert you want to consult and input your question.')
        st.text('You can also expand the "Advanced" label to change more parameters.')
        st.text('"Auto Clean" will clean the question area after you submit the question.')
        
        
        st.subheader('2. Deep Consult', divider='grey')
        st.text('"Deep Consult" allow user to ask multiple questions first before ask the final question.')
        st.text('User can separate a complicated question into multiple questions and ask them one by one.')
        st.text('Each response of previous question will be the reference of final question, so the response of final question will be more accurate.')
        
    st.header('⚙️ Settings', divider='rainbow')
    with st.expander('Settings', expanded=False):
        st.subheader('1. API Settings', divider='grey')
        st.text('If users want to use any openai embedding or language model, they need to add openai config in this page.')
        st.text('For openai api, users need to add openai key; for azure api, users need to add azure key and azure base url.')
        st.text('Click the "Save" button and it will check if the config is valid, if so, it will save the config.')
        st.text('User can also click "Save to File" button to save the config to a json file, so next time the config will load automatically.')
        
        
        st.subheader('2. History', divider='grey')
        st.text('After user consults an expert, the consult history will be saved in the log.')
        st.text('User can download the log by clicking the "Download" button to download .txt file or .json file.')