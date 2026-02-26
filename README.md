# akasha

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pypi package](https://img.shields.io/pypi/v/akasha-terminal)](https://pypi.org/project/akasha-terminal/)
[![downloads](https://img.shields.io/pypi/dm/akasha-terminal)](https://pypi.org/project/akasha-terminal/)
[![python version : 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/release/python-3109/)
![GitLab CI](https://img.shields.io/badge/gitlab%20ci-%23181717.svg?style=for-the-badge&logo=gitlab&logoColor=white)
akasha simplifies document-based Question Answering (QA) and Retrieval Augmented Generation(RAG) by harnessing the power of Large Language Models to accurately answer your queries while searching through your provided documents.

With akasha, you have the flexibility to choose from a variety of language models, embedding models, and search types. Adjusting these parameters is straightforward, allowing you to optimize your approach and discover the most effective methods for obtaining accurate answers from Large Language Models.

For the chinese manual, please visit [manual](https://tea9297.github.io/akasha/)

## Quick Start (Local Development)

If you are developing in this repository (instead of only installing from PyPI), use editable install:

```bash
cd akasha
uv venv --python 3.10
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install -e .
```

Set at least one model API key:

```bash
export OPENAI_API_KEY="your_key"

# or
export GEMINI_API_KEY="your_key"
```

Run a quick example:

```bash
python examples/ex_rag.py
python examples/ex_agent.py
```

# Change log

- 1.0

    1. bug fixes
    2. added a lightweight installation option (API-call-only mode)
    3. upgraded LangChain to 1.2

- 0.9.14

    1. function calling
    2. MCP agent support

# Installation

We recommend using Python 3.10 to run our akasha package. You can use uv or Anaconda to create virtual environment.

### Standard Installation

```shell!
###create environment
$ uv venv --python 3.10

###install akasha
$ uv pip install akasha-terminal
```

### Lightweight Installation (API-call-only, v1.0+)

```bash
### create environment
uv venv --python 3.10
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1

### install lightweight mode (API-call-only)
uv pip install "akasha-terminal[light]"
```

If you are developing in this repository:

```bash
uv pip install -e .
```

or only install the lightweight (API-call-only) extras:

```bash
uv pip install -e ".[light]"
```

## API Keys

### OPENAI

If you want to use openai models or embeddings, go to [openai](https://platform.openai.com/account/api-keys) to get the API key.
You can either save **OPENAI_API_KEY=your api key** into **.env** file to current working directory or,
set as a environment variable, using **export** in bash or use **os.environ** in python.

```bash

# set a environment variable

export OPENAI_API_KEY="your api key"

```

### GEMINI

If you want to use Gemini models, set `GEMINI_API_KEY` in your `.env` file or export it in your shell.

`.env` example:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Shell example:

```bash
export GEMINI_API_KEY="your_gemini_api_key"
```

### AZURE OPENAI

If you want to use azure openai, go to [auzreAI](https://oai.azure.com/portal) and get you own Language API base url and key.
Also, remember to depoly all the models in [Azure OpenAI Studio](https://oai.azure.com/portal), the deployment name should be same as the model name. save **OPENAI_API_KEY=your azure key**,  **OPENAI_API_BASE=your Language API base url**, **OPENAI_API_TYPE=azure**, **OPENAI_API_VERSION=2023-05-15** into **.env** file to current working directory.
If you want to save both openai key and azure key at the same time, you can also use **AZURE_API_KEY**, **AZURE_API_BASE**, **AZURE_API_TYPE**, **AZURE_API_VERSION**

```sh

## .env file
AZURE_API_KEY={your azure key}
AZURE_API_BASE={your Language API base url}
AZURE_API_TYPE=azure
AZURE_API_VERSION=2023-05-15

```
And now we can run akasha in python

```python
#PYTHON3.10+
import akasha

# simple QA
ak = akasha.ask(model="gemini:gemini-2.5-flash")
response = ak(
    prompt="akasha 是什麼？",
    info=["https://github.com/iii-org/akasha"],
)

```

And then run a RAG example:

```python
#PYTHON3.10+
import akasha
data_source = "doc/mic"
prompt = "五軸是什麼?"
ak = akasha.RAG(model="gemini:gemini-2.5-flash")
response = ak(data_source, prompt)

```

## Some models you can use

Please note that for OpenAI models, you need to set the environment variable 'OPENAI_API_KEY,' and for most Hugging Face models, a GPU is required to run the models. However, for .gguf models, you can use a CPU to run them.

```python
openai_model = "openai:gpt-3.5-turbo"  # need environment variable "OPENAI_API_KEY" or "AZURE_API_KEY"
openai4_model = "openai:gpt-4"  # need environment variable "OPENAI_API_KEY" or "AZURE_API_KEY"
gemini_flash_model = "gemini:gemini-2.5-flash" # need environment variable "GEMINI_API_KEY"
huggingface_model = "hf:meta-llama/Llama-2-7b-chat-hf" #need environment variable "HUGGINGFACEHUB_API_TOKEN" to download meta-llama model
quantized_ch_llama_model = "gptq:FlagAlpha/Llama2-Chinese-13b-Chat-4bit"
taiwan_llama_gptq = "gptq:weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ"
mistral = "hf:Mistral-7B-Instruct-v0.2" 
mediatek_Breeze = "hf:MediaTek-Research/Breeze-7B-Instruct-64k-v0.1"

### If you want to use llama-cpp to run model on cpu, you can download gguf version of models 

### from https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF  and the name behind "llama-gpu:" or "llama-cpu:"

### from https://huggingface.co/TheBloke/CodeUp-Llama-2-13B-Chat-HF-GGUF

### is the path of the downloaded .gguf file
llama_cpp_model = "llama-cpp:model/llama-2-13b-chat-hf.Q5_K_S.gguf"  
llama_cpp_model = "llama-cpp:model/llama-2-7b-chat.Q5_K_S.gguf"
llama_cpp_chinese_alpaca = "llama-cpp:model/chinese-alpaca-2-7b.Q5_K_S.gguf"
chatglm_model = "chatglm:THUDM/chatglm2-6b"

```

## Some embeddings you can use

Please noted that each embedding model has different window size, texts that over the max seq length will be truncated and won't be represent
in embedding model.

Rerank_base and rerank_large are not embedding models; instead, they compare the query to each chunk of the documents and return scores that represent the similarity. As a result, they offer higher accuracy compared to embedding models but may be slower.

```python
openai_emd = "openai:text-embedding-ada-002"  # need environment variable "OPENAI_API_KEY"  # 8192 max seq length
huggingface_emd = "hf:all-MiniLM-L6-v2" 
text2vec_ch_emd = "hf:shibing624/text2vec-base-chinese"   # 128 max seq length 
text2vec_mul_emd = "hf:shibing624/text2vec-base-multilingual"  # 256 max seq length
text2vec_ch_para_emd = "hf:shibing624/text2vec-base-chinese-paraphrase" # perform better for long text, 256 max seq length
bge_en_emd = "hf:BAAI/bge-base-en-v1.5"  # 512 max seq length
bge_ch_emd = "hf:BAAI/bge-base-zh-v1.5"  # 512 max seq length

rerank_base = "rerank:BAAI/bge-reranker-base"    # 512 max seq length
rerank_large = "rerank:BAAI/bge-reranker-large"  # 512 max seq length

```

## File Summarization

To create a summary of a text file in various formats like `.pdf`, `.txt`, or `.docx`, you can use the `Summary` class. For example, the following code uses the `map_reduce` method to generate a summary.

There are two summary types, `map_reduce` and `refine`. `map_reduce` summarizes each chunk first, then produces a final summary from all chunk summaries. `refine` summarizes chunk-by-chunk and uses the previous summary as context for the next chunk, which often improves consistency.

```python
import akasha

summ = akasha.summary(
    model="gemini:gemini-2.5-flash",
    sum_type="map_reduce",
    chunk_size=1000,
    sum_len=1000,
    language="en",
    keep_logs=True,
    verbose=True,
    max_input_tokens=8000,
)

# Content can be a URL, file, or plain text
ret = summ(content=["https://github.com/iii-org/akasha"])
```

## agent

By implementing an agent, you empower the LLM with the capability to utilize tools more effectively to accomplish tasks. You can allocate tools for tasks such as file editing, conducting Google searches, and enlisting the LLM's assistance in task execution, rather than solely relying on it to respond your questions.

### Use Built-in Tools

```python
import akasha.agent.agent_tools as at

# Use built-in web search and JSON save tools
tool_list = [at.websearch_tool(search_engine="brave"), at.saveJSON_tool()]

agent = akasha.agents(
    tools=tool_list,
    model="gemini:gemini-2.5-flash",
    temperature=1.0,
    max_input_tokens=8000,
    verbose=True,
    keep_logs=True,
)

# Ask a question and let the agent use tools to answer
response = agent("Search for Industry 4.0 on the web and save the result to iii.json")
print(response)

# Save logs
agent.save_logs("logs.json")
```

### Define and Use a Custom Tool

```python
import akasha
from datetime import datetime

# Define a tool to get today's date
def today_f():
    now = datetime.now()
    return "today's date: " + str(now.strftime("%Y-%m-%d %H:%M:%S"))

# Create the tool
today_tool = akasha.create_tool(
    "This is the tool to get today's date, the tool doesn't have any input parameter.",
    today_f,
    "today_date_tool",
)

# Create an agent with the tool
agent = akasha.agents(
    tools=[today_tool],
    model="gemini:gemini-2.5-flash",
    temperature=1.0,
    verbose=True,
    keep_logs=True,
)

# Ask a question and let the agent use the tool
response = agent("What is today's date?")
print(response)

# Save logs
agent.save_logs("logs.json")
```

### Use Tools from MCP Servers

```python
import asyncio
import akasha
from langchain_mcp_adapters.client import MultiServerMCPClient

MODEL = "gemini:gemini-2.5-flash"

# Define MCP server connection info
connection_info = {
    "math": {
        "command": "python",
        "args": ["cal_server.py"],
        "transport": "stdio",
    },
    "weather": {
        "url": "http://localhost:8000/sse",
        "transport": "sse",
    },
}
prompt = "tell me the weather in Taipei"

# Use MCP tools
agent = akasha.agents(
    model=MODEL,
    temperature=1.0,
    verbose=True,
    keep_logs=True,
)
response = agent.mcp_agent(connection_info, prompt)
agent.save_logs("logs_agent.json")

```
