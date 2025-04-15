from typing import List
from langchain.schema import Document
from dotenv import dotenv_values
import os
from akasha.utils.prompts.format import websearch_language_dict, websearch_country_dict


def load_docs_from_webengine(
    prompt: str,
    search_engine: str = "wiki",
    search_num: int = 5,
    language: str = "ch",
    env_file: str = "",
) -> List[Document]:
    """get the search results based on the prompt and search engine
    search_engine include : wiki, serper, brave, tavily

    """

    search_engine = search_engine.lower()

    if search_engine == "wiki":
        api_key = ""
    else:
        api_key = _get_search_api_key(search_engine, env_file)

    if search_engine == "wiki":
        from langchain_community.document_loaders import WikipediaLoader

        docs = WikipediaLoader(
            query=prompt,
            load_max_docs=search_num,
            lang=websearch_language_dict[language][search_engine],
        ).load()

    elif search_engine == "serper":
        from langchain_community.utilities import GoogleSerperAPIWrapper

        google_serper = GoogleSerperAPIWrapper(
            serper_api_key=api_key,
            gl=websearch_country_dict[language][search_engine],
            hl=websearch_language_dict[language][search_engine],
            k=search_num,
        )
        search_res = google_serper.run(prompt)

        docs = [Document(page_content=search_res)]

    elif search_engine == "brave":
        from langchain_community.document_loaders import BraveSearchLoader

        loader = BraveSearchLoader(
            query=prompt,
            api_key=api_key,
            search_kwargs={
                "count": search_num,
                "country": "all",
                "search_lang": websearch_language_dict[language][search_engine],
            },
        )
        docs = loader.load()

    elif search_engine == "tavily":
        from tavily import TavilyClient

        loader = TavilyClient(api_key=api_key)

        results = loader.search(prompt, "advanced", max_results=search_num)
        docs = [
            Document(page_content=doc_dict["title"] + "\n" + doc_dict["content"])
            for doc_dict in results["results"]
        ]

    else:
        raise ValueError(f"search_engine {search_engine} is not supported")
    return docs


def _get_search_api_key(search_engine: str = "wiki", env_file: str = "") -> str:
    """get the search api key based on the search engine"""
    if env_file == "" or not os.path.exists(env_file):
        if search_engine == "serper":
            return os.environ["SERPER_API_KEY"]
        elif search_engine == "brave":
            return os.environ["BRAVE_API_KEY"]
        elif search_engine == "tavily":
            return os.environ["TAVILY_API_KEY"]
        else:
            return ""

    else:
        env_dict = dotenv_values(env_file)

        if search_engine == "serper":
            return env_dict["SERPER_API_KEY"]
        elif search_engine == "brave":
            return env_dict["BRAVE_API_KEY"]
        elif search_engine == "tavily":
            return env_dict["TAVILY_API_KEY"]
        else:
            return ""
