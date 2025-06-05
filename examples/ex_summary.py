import akasha

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
DEFAULT_MAX_INPUT_TOKENS = 3000
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CHUNK_SIZE = 1000

#### create a summary object and call it ###
### sum_type is the summarization method, could be "map_reduce" or "refine" ###
### sum_len is the length of the final summary you suggest llm should be ###
### it will first split the content into chunks based on chunk_size, and then summarize several chunks at a time(depend on the llm window size/max_input_tokens),
### and finally merge them together. ###
### language is the language of the content, could be "en" or "zh" ###

summ = akasha.summary(
    "openai:gpt-4o",
    sum_type="map_reduce",
    chunk_size=DEFAULT_CHUNK_SIZE,
    sum_len=1000,
    language="en",
    keep_logs=True,
    verbose=True,
    max_input_tokens=8000,
)

### use llm to summarize content,  ###
### info can be a list of local files, string, directories, or urls ###
### files includes pdf, docx, txt, md, csv, pptx files ###
ret = summ(content=["https://github.com/iii-org/akasha"])

# save the logs or turn verbose on to see the details
summ.save_logs("sumlog.json")
