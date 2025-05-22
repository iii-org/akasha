import akasha.helper as ah
from akasha.utils.prompts.gen_prompt import format_sys_prompt
from pydantic import BaseModel

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
DEFAULT_EMBED = "openai:text-embedding-ada-002"
DEFAULT_MAX_INPUT_TOKENS = 3000
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_SEARCH_TYPE = "auto"
SYS_PROMPT = "you are a helpful assistant to answer user question"
PROMPT = "akasha是甚麼?"
PROMPT2 = "工業4.0是甚麼?"
PROMPT3 = """
openai_model = "openai:gpt-3.5-turbo"  # need environment variable "OPENAI_API_KEY"
gemini_model="gemini:gemini-1.5-flash" # need environment variable "GEMINI_API_KEY"
anthropic_model = "anthropic:claude-3-5-sonnet-20241022" # need environment variable "ANTHROPIC_API_KEY"
huggingface_model = "hf:meta-llama/Llama-2-7b-chat-hf" #need environment variable "HUGGINGFACEHUB_API_TOKEN" to download meta-llama model
qwen_model = "hf:Qwen/Qwen2.5-7B-Instruct"
quantized_ch_llama_model = "hf:FlagAlpha/Llama2-Chinese-13b-Chat-4bit"
taiwan_llama_gptq = "hf:weiren119/Taiwan-LLaMa-v1.0-4bits-GPTQ"
mistral = "hf:Mistral-7B-Instruct-v0.2" 
mediatek_Breeze = "hf:MediaTek-Research/Breeze-7B-Instruct-64k-v0.1"
"""
TEMPERATURE = 1.0

### create a model object and call it ###
model_obj = ah.handle_model(
    DEFAULT_MODEL,
    verbose=True,
    temperature=TEMPERATURE,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
)

# (option) format the prompt#
prod_sys_prompt = format_sys_prompt(SYS_PROMPT, PROMPT, "chat_gpt", DEFAULT_MODEL)

# call the model #
ret = ah.call_model(model_obj, prod_sys_prompt)

# call the model in parallel #
ret2 = ah.call_batch_model(model_obj, [PROMPT, PROMPT2])

# call the model in stream #
st = ah.call_stream_model(model_obj, PROMPT)

for s in st:
    print(s)


# restrict the model to output JSON format response
# you can use pydantic to define the JSON format keys
class Model_Type(BaseModel):
    model_type: str
    model_name: str


json_response = ah.call_JSON_formatter(model_obj, PROMPT3, keys=Model_Type)
print(json_response)
# response: [{'model_type': 'openai', 'model_name': 'gpt-3.5-turbo'}, {'model_type': 'gemini', 'model_name': 'gemini-1.5-flash'}, {'model_type': 'anthropic', 'model_name': 'claude-3-5-sonnet-20241022'},...

#
#
#
#
#
#
### create an embedding object and embed the query ###
emb_obj = ah.handle_embeddings(DEFAULT_EMBED, verbose=True)
embed_val = emb_obj.embed_query(PROMPT)

### get the name(string) of the embedding/model object ###
embed_name = ah.handle_model_type(emb_obj)
