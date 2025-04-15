import streamlit as st
import akasha
import sys

sys.path.append("../")


def websearch_page():
    """implement get response ui"""

    if "prompt_list" not in st.session_state:
        st.session_state.prompt_list = []

    st.title("Web Search")
    response_board, para_set = st.columns([3, 1])

    with para_set:
        st.session_state.sys_prompt = st.text_area(
            "System Prompt",
            st.session_state.sys_prompt,
            help="The special instruction you want to give to the model.",
        )

        s_eng, s_num = st.columns([1, 1])

        with s_eng:
            st.session_state.search_engine = st.selectbox(
                "Search Engine",
                ["wiki", "tavily", "serper", "brave"],
                help="The search engine you want to use.",
            )

        with s_num:
            st.session_state.search_num = st.number_input(
                "Search Number",
                min_value=1,
                max_value=20,
                value=st.session_state.search_num,
                help="The maximum number of search result.",
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
        if not isinstance(st.session_state.akasha_obj, akasha.websearch):
            st.session_state.akasha_obj = akasha.websearch(
                model=st.session_state.model,
                language="ch",
                verbose=True,
                search_engine=st.session_state.search_engine,
                search_num=st.session_state.search_num,
                max_input_tokens=st.session_state.max_input_tokens,
                temperature=st.session_state.temperature,
                env_file=st.session_state.env_path,
            )

        ans = st.session_state.akasha_obj(
            prompt,
            model=st.session_state.model,
            system_prompt=st.session_state.sys_prompt,
            max_input_tokens=st.session_state.max_input_tokens,
            temperature=st.session_state.temperature,
            keep_logs=True,
            env_file=st.session_state.env_path,
            stream=st.session_state.stream,
            search_engine=st.session_state.search_engine,
            search_num=st.session_state.search_num,
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
