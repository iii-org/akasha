import requests
import os
### use akasha api in terminal to start the api server ###
### use -p port -h host to specify the port and host ###
HOST = os.getenv("API_HOST", "http://127.0.0.1")
PORT = os.getenv("API_PORT", "8000")
urls = {
    "summary": f"{HOST}:{PORT}/summary",
    "rag": f"{HOST}:{PORT}/RAG",
    "ask": f"{HOST}:{PORT}/ask",
    "websearch": f"{HOST}:{PORT}/websearch",
}

env_config = {
    "AZURE_API_KEY": "your azure key",
    "AZURE_API_BASE": "your azure base",
    "OPENAI_API_KEY": "your openai key",
    "SERPER_API_KEY": "your serper key",
    "BRAVE_API_KEY": "your brave key",
    "ANTHROPIC_API_KEY": "your anthropic key",
    "GEMINI_API_KEY": "your gemini key",
}

ask_data = {
    "prompt": "太陽能電池技術?",
    "info": "太陽能電池技術5塊錢",
    "model": "openai:gpt-3.5-turbo",
    "system_prompt": "",
    "temperature": 0.0,
    "env_config": env_config
}

rag_data = {
    "data_source": "docs/mic/",
    "prompt": "工業4.0",
    "chunk_size": 1000,
    "model": "openai:gpt-3.5-turbo",
    "embedding_model": "openai:text-embedding-ada-002",
    "threshold": 0.1,
    "search_type": 'auto',
    "system_prompt": "",
    "max_input_tokens": 3000,
    "temperature": 0.0,
    "env_config": env_config
}
summary_data = {
    "content": "docs/2.pdf",
    "model": "openai:gpt-3.5-turbo",
    "summary_type": "reduce_map",
    "summary_len": 500,
    "system_prompt": "用中文做500字摘要",
    "env_config": env_config
}

websearch_data = {
    "prompt": "太陽能電池技術?",
    "model": "openai:gpt-3.5-turbo",
    "system_prompt": "",
    "temperature": 0.0,
    "env_config": env_config,
    "search_engine": "serper",
    "search_num": 5
}

ask_response = requests.post(urls["ask"], json=ask_data).json()
print(ask_response)
# rag_response = requests.post(urls["rag"], json=rag_data).json()
# print(rag_response)

# sum_response = requests.post(
#     urls["summary"],
#     json=summary_data,
# ).json()

# print(sum_response)

# websearch_response = requests.post(urls["websearch"],
#                                    json=websearch_data).json()
# print(websearch_response)
