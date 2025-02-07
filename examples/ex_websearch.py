import akasha

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
DEFAULT_MAX_INPUT_TOKENS = 3000
DEFAULT_MAX_OUTPUT_TOKENS = 1024
PROMPT = "工業4.0是什麼?"

#### create a websearch object and call it, it will use the prompt to search in the web, and based on the information
# to answer the question.  ###
### language will determine the language and region of search engine is used. "en" or "ch" ###
### search_engine is the search api to use, include "wiki", "serper" and "brave" ###
### to use "brave" api, you need to apply the api key (https://brave.com/search/api/) and set the environment
# variable BRAVE_API_KEY in .env file ###
### to use google "serper" api, you need to apply the api key (https://serper.dev/) and set the environment
# variable SERPER_API_KEY in .env file ###
### search_num is the number of search results to return ###
wb = akasha.websearch(
    model="openai:gpt-4o",
    language="ch",
    search_engine="serper",
    search_num=5,
    verbose=True,
    keep_logs=True,
)

### use search api to find the information and use llm to answer the user prompt ###
res = wb(PROMPT)

# save the logs or turn verbose on to see the details
wb.save_logs("wb.json")

### same as ask, you can set stream to True to get the response in stream ###
st = wb(PROMPT, stream=True)

for s in st:
    print(s)
