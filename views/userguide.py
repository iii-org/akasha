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
        st.text('Some Instructions of "My Datasets"')
        st.subheader('2. New Dataset', divider='grey')
        st.text('Some Instructions of "New Dataset"')
        st.subheader('3. Update Dataset', divider='grey')
        st.text('Some Instructions of "Update Dataset"')
        
    st.header('👑 Experts', divider='rainbow')
    with st.expander('Experts', expanded=False):
        st.subheader('1. My Experts', divider='grey')
        st.text('Some Instructions of "My Experts"')
        st.subheader('2. New Expert', divider='grey')
        st.text('Some Instructions of "New Expert"')
        st.subheader('3. Update Expert', divider='grey')
        st.text('Some Instructions of "Update Expert"')
        
    st.header('🧭 Consult', divider='rainbow')
    with st.expander('Consult', expanded=False):
        st.subheader('1. Regular Consult', divider='grey')
        st.text('Some Instructions of "Regular Consult"')
        st.subheader('2. Deep Consult', divider='grey')
        st.text('Some Instructions of "Deep Consult"')
        
    st.header('⚙️ Settings', divider='rainbow')
    with st.expander('Settings', expanded=False):
        st.subheader('1. API Settings', divider='grey')
        st.text('Some Instructions of "API Settings"')
        st.subheader('2. History', divider='grey')
        st.text('Some Instructions of "History"')