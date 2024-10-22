import streamlit as st
import akasha
import sys
from interface.setting import handle_api_key

sys.path.append("../")


def response_page():
    """implement get response ui"""

    if "prompt_list" not in st.session_state:
        st.session_state.prompt_list = []

    st.title("Get Response")
    st.markdown("##")
    st.markdown("##")
    run_flag = True
    response_board, para_set = st.columns([2, 1])

    with para_set:
        st.session_state.sys_prompt = st.text_area(
            "System Prompt",
            st.session_state.sys_prompt,
            help="The special instruction you want to give to the model.",
        )
        prompt = st.text_area("Prompt",
                              "",
                              help="The prompt you want to ask the model.")

        sb1, sb2 = st.columns([1, 1])
        with sb1:
            if st.button(
                    "Clear",
                    type="primary",
                    use_container_width=True,
                    help="Clear the prompt and response history.",
            ):
                st.session_state.prompt_list = []
                st.session_state.response_list = []
                run_flag = True
                st.rerun()
        with sb2:
            if st.button("Submit", type="primary", use_container_width=True):
                run_flag = handle_api_key()

                if run_flag:
                    ## check if the object is created correctly ##
                    if not isinstance(st.session_state.akasha_obj,
                                      akasha.Doc_QA):
                        st.session_state.akasha_obj = akasha.Doc_QA(
                            embeddings=st.session_state.embed,
                            chunk_size=st.session_state.chunksize,
                            model=st.session_state.model,
                            search_type=st.session_state.search_type,
                            threshold=st.session_state.threshold,
                            language="ch",
                            verbose=True,
                            record_exp="",
                            max_doc_len=st.session_state.max_doc_len,
                            temperature=st.session_state.temperature,
                        )

                    ans = st.session_state.akasha_obj.get_response(
                        st.session_state.chose_doc_path,
                        prompt,
                        embeddings=st.session_state.embed,
                        chunk_size=st.session_state.chunksize,
                        model=st.session_state.model,
                        threshold=st.session_state.threshold,
                        search_type=st.session_state.search_type,
                        system_prompt=st.session_state.sys_prompt,
                        max_doc_len=st.session_state.max_doc_len,
                        temperature=st.session_state.temperature,
                        keep_logs=True,
                    )
                    st.session_state.prompt_list.append(prompt)
                    st.session_state.response_list.append(ans)
                    timesp = st.session_state.akasha_obj.timestamp_list[-1]
                    if timesp in st.session_state.akasha_obj.logs:
                        st.session_state.logs[
                            timesp] = st.session_state.akasha_obj.logs[timesp]

    if not run_flag:
        st.error("Please input your openAI api key.")
    with response_board:
        for i in range(len(st.session_state.response_list)):
            with st.chat_message("user"):
                st.markdown(st.session_state.prompt_list[i])
            with st.chat_message("assistant"):
                st.markdown(st.session_state.response_list[i])
