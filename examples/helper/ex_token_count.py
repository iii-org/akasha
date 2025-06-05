import akasha.helper as ah

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
TEXT = "工業4.0是甚麼?"
TEXT2 = """工業4.0是甚麼? 
{
  "title": "工業4.0",
  "description": "工業4.0是一場製造業的數位轉型，融合了物聯網、人工智慧與自動化技術，提升生產效率與靈活性。" 
}
"""
### compute the tokens of the text by the model ###
tokens = ah.myTokenizer.compute_tokens(TEXT, DEFAULT_MODEL)

### compute the length of the text by jieba ###
doc_length = ah.get_doc_length(TEXT)

### translate simplified chinese to traditional chinese ###
ret = ah.sim_to_trad(TEXT)

### get the json format dictionary from the text ###
json_str = ah.extract_json(TEXT2)
