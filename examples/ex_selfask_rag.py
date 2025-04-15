import akasha

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
DEFAULT_EMBED = "openai:text-embedding-ada-002"
DEFAULT_MAX_INPUT_TOKENS = 3000
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_SEARCH_TYPE = "auto"
PROMPT = "akasha是甚麼?"

#### create a RAG object and call it ###
ak = akasha.RAG(
    embeddings="openai:text-embedding-3-small",
    model="openai:gpt-4o",
    max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
    keep_logs=True,
    verbose=True,
)
### selfask_RAG is a function that will first use llm to separate user question into several prompts,
### then use RAG to search for the answer. ###
### use data source as reference and search similar document to answer the query ###
### data_source can be a list of local files, directories, or urls ###
### files includes pdf, docx, txt, md, csv, pptx files ###
res = ak.selfask_RAG(
    data_source=["docs/mic", "https://github.com/iii-org/akasha"],
    prompt=PROMPT,
)

# save the logs or turn verbose on to see the details
ak.save_logs("logs.json")

### you can set stream to True to get the response in stream ###
st = ak.selfask_RAG("docs/mic", PROMPT, stream=True)

for s in st:
    print(s)
