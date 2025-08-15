import akasha

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
DEFAULT_MAX_INPUT_TOKENS = 3000
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_SEARCH_TYPE = "auto"
PROMPT = "akasha是甚麼?"

#### create a ask object and call it ###
ak = akasha.ask(
    model="openai:gpt-4o", max_input_tokens=8000, keep_logs=True, verbose=True
)

### use info as reference, could be empty, but not recommand to be large, since llm will use all of the content to answer the question ###
### info can be a list of local files, string, directories, or urls ###
### files includes pdf, docx, txt, md, csv, pptx files ###
res = ak(
    prompt=PROMPT,
    info=["https://github.com/iii-org/akasha"],
)

### use vision to ask question about image ###
### image_path can be a local file or an url ###
### please noted that most of the models can not process high quality images (e.g. 3mb) ###
res = ak.vision(
    prompt="這張圖片是什麼?",
    image_path="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
)

# save the logs or turn verbose on to see the details
ak.save_logs("logs_ask.json")

### you can set stream to True to get the response in stream ###
st = ak(PROMPT, "https://github.com/iii-org/akasha", stream=True)
full_response = ""
for s in st:
    full_response += s
