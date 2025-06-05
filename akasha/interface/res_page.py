import streamlit as st
import akasha
import sys

sys.path.append("../")


def response_page():
    """implement rag ui"""

    if "prompt_list" not in st.session_state:
        st.session_state.prompt_list = []

    st.title("RAG")
    response_board, para_set = st.columns([3, 1])

    with para_set:
        st.session_state.sys_prompt = st.text_area(
            "System Prompt",
            st.session_state.sys_prompt,
            help="The special instruction you want to give to the model.",
        )

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
                st.rerun()
        with sb2:
            st.session_state.stream = st.toggle(
                "Stream",
                value=st.session_state.stream,
                help="the output to be stream or not.",
            )

    for i in range(len(st.session_state.response_list)):
        with response_board.chat_message("user"):
            st.markdown(st.session_state.prompt_list[i])
        with response_board.chat_message("assistant"):
            st.markdown(st.session_state.response_list[i])

    prompt_borad, para_set2 = st.columns([3, 1])
    with prompt_borad:
        prompt = st.chat_input(
            "Ask your question here",
        )

    if prompt:
        ## check if the object is created correctly ##
        with response_board.chat_message("user"):
            st.markdown(prompt)
        if not isinstance(st.session_state.akasha_obj, akasha.RAG):
            st.session_state.akasha_obj = akasha.RAG(
                embeddings=st.session_state.embed,
                chunk_size=st.session_state.chunksize,
                model=st.session_state.model,
                search_type=st.session_state.search_type,
                language="ch",
                verbose=True,
                record_exp="",
                max_input_tokens=st.session_state.max_input_tokens,
                temperature=st.session_state.temperature,
                env_file=st.session_state.env_path,
            )

        ans = st.session_state.akasha_obj(
            st.session_state.chose_doc_path,
            prompt,
            embeddings=st.session_state.embed,
            chunk_size=st.session_state.chunksize,
            model=st.session_state.model,
            search_type=st.session_state.search_type,
            system_prompt=st.session_state.sys_prompt,
            max_input_tokens=st.session_state.max_input_tokens,
            temperature=st.session_state.temperature,
            keep_logs=True,
            env_file=st.session_state.env_path,
            stream=st.session_state.stream,
        )

        with response_board.chat_message("assistant"):
            if st.session_state.stream:
                ans = st.write_stream(ans)
            else:
                st.markdown(ans)

        st.session_state.prompt_list.append(prompt)
        st.session_state.response_list.append(ans)
        timesp = st.session_state.akasha_obj.timestamp_list[-1]
        if timesp in st.session_state.akasha_obj.logs:
            st.session_state.logs[timesp] = st.session_state.akasha_obj.logs[timesp]
