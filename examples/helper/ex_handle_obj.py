import akasha.helper as ah
from akasha.utils.prompts.gen_prompt import format_sys_prompt

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
DEFAULT_EMBED = "openai:text-embedding-ada-002"
DEFAULT_MAX_INPUT_TOKENS = 3000
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_SEARCH_TYPE = "auto"
SYS_PROMPT = "you are a helpful assistant to answer user question"
PROMPT = "akasha是甚麼?"
PROMPT2 = "工業4.0是甚麼?"
TEMPERATURE = 1.0

### create a model object and call it ###
model_obj = ah.handle_model(DEFAULT_MODEL,
                            verbose=True,
                            temperature=TEMPERATURE,
                            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS)

#(option) format the prompt#
prod_sys_prompt = format_sys_prompt(SYS_PROMPT, PROMPT, "chat_gpt",
                                    DEFAULT_MODEL)

# call the model #
ret = ah.call_model(model_obj, prod_sys_prompt)

# call the model in parallel #
ret2 = ah.call_batch_model(model_obj, [PROMPT, PROMPT2])

# call the model in stream #
st = ah.call_stream_model(model_obj, PROMPT)

for s in st:
    print(s)
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
