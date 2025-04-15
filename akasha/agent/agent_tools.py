from typing import Callable, Union, List
from langchain.tools import BaseTool
from akasha.helper.base import extract_json
from .base import create_tool
import json


def websearch_tool(
    search_engine: str = "wiki", language: str = "ch", env_file: str = ""
) -> BaseTool:
    """return the tool to use search engine to search information for user

    Args:
        search_engine (str, optional): the search_engine options, includes "wiki", "serper" "tavily" and "brave". Defaults to "wiki".
        language (str, optional): the language options, includes "ch" and "en". Defaults to "ch".
        env_file (str, optional): the .env file save the api keys. Defaults to "".

    Returns:
        _type_: _description_
    """
    from akasha.helper.web_engine import load_docs_from_webengine

    def _websearch_tool(prompt: str, search_num: int = 5):
        docs = load_docs_from_webengine(
            prompt, search_engine, search_num, language, env_file
        )

        ret = "\n\n".join([do.page_content for do in docs])
        return ret

    ret_tool = create_tool(
        tool_name="web_search_tool",
        tool_description="This is the tool to use prompt to search information from web engine, the params \
        includes prompt, search_num(default 5)",
        func=_websearch_tool,
    )

    return ret_tool


def calculate_tool():
    def _cal_tool(expr: str):
        return eval(expr)

    ret_tool = create_tool(
        tool_name="calculate_tool",
        tool_description="This is the tool to calculate the math expression, the only one parameter is expr",
        func=_cal_tool,
    )

    return ret_tool


def saveJSON_tool() -> BaseTool:
    """return the json save tool that can save the content into json file.

    Returns:
        _type_: _description_
    """
    return create_tool(
        tool_name="json_tool",
        tool_description="This is the tool to save the content into json file, the input contains file_path and content.",
        func=_jsonSaveTool,
    )


def _jsonSaveTool(file_path: str = "default.json", content: str = None):
    """save content into json file"""
    if content:
        try:
            # change content from string to json
            try:
                content = extract_json(content)
            except Exception:
                print(content)
            import json

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=4, ensure_ascii=False)
            return f"Success create {file_path}"
        except Exception as e:
            print("content: ", content)
            return f"{e}, Cannot save file_path {file_path}, save file as default.json"
    else:
        return "Error: content is empty"


def rag_tool(
    embeddings: Union[str, Callable],
    chunk_size: int = 1000,
    max_input_tokens: int = 3000,
    env_file: str = "",
):
    from akasha.utils.db import process_db
    from akasha.helper.handle_objects import handle_embeddings
    from akasha.utils.search.retrievers.base import get_retrivers
    from akasha.utils.search.search_doc import search_docs

    embeddings_obj = handle_embeddings(embeddings, env_file=env_file)

    def _rag_tool(
        data_source: Union[List[str], str], prompt: str, search_type: str = "knn"
    ):
        # Ensure data_source is a list
        if isinstance(data_source, str):
            try:
                # Attempt to load as JSON
                data_source = json.loads(data_source)
                if not isinstance(data_source, list):
                    # If it's not a list, treat it as a single path string
                    data_source = [data_source]
            except json.JSONDecodeError:
                # If JSON decoding fails, treat it as a single path string
                data_source = [data_source]
        elif not isinstance(data_source, list):
            raise ValueError(
                f"data_source should be a list of strings or a single string, instead, we got {data_source}."
            )

        db = process_db(
            data_source=data_source,
            embeddings=embeddings_obj,
            chunk_size=chunk_size,
            env_file=env_file,
        )

        retrivers_list = get_retrivers(
            db,
            embeddings_obj,
            0.0,
            search_type,
            env_file,
        )

        docs, doc_length, doc_tokens = search_docs(
            retrivers_list,
            prompt,
            max_input_tokens=max_input_tokens,
            search_type=search_type,
        )

        return "\n\n".join([do.page_content for do in docs])

    ret_tool = create_tool(
        tool_name="rag_tool",
        tool_description="This is the tool to use prompt to search information from db, the params \
        includes data_source(str or list, the path of data source), prompt, search_type(default 'knn')",
        func=_rag_tool,
    )

    return ret_tool
