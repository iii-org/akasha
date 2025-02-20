import akasha.helper as ah

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
TEXT = "工業4.0是甚麼?"

### compute the tokens of the text by the model ###
tokens = ah.myTokenizer.compute_tokens(TEXT, DEFAULT_MODEL)

### compute the length of the text by jieba ###
doc_length = ah.get_doc_length(TEXT)

### translate simplified chinese to traditional chinese ###
ret = ah.sim_to_trad(TEXT)
