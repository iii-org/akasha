import click


import akasha as ak


@click.group()
def akasha():
    pass




@click.command()
@click.option('--doc_path', '-d', help='document directory path, parse all .txt, .pdf, .docx files in the directory', required=True)
@click.option('--prompt','-p', help='prompt you want to ask to llm', required=True)
@click.option('--embeddings','-e', default="openai:text-embedding-ada-002", help='embeddings for storing the documents')
@click.option('--chunk_size', '-c', default = 1000, help='chunk size for storing the documents')
@click.option('--model', '-m', default="openai:gpt-3.5-turbo",help='llm model for generating the response')
@click.option('--wwww', '-k', default = 2, help='select topK relevant documents')
@click.option('--threshold', '-t', default=0.2, help='threshold score for selecting the relevant documents')
@click.option('--language', '-l', default='ch', help='language for the documents, default is \'ch\' for chinese')
@click.option('--search_type', '-s', default='merge', help='search type for the documents, include merge, svm, mmr, tfidf')
@click.option('--record_exp', '-r', default="", help='input the experiment name if you want to record the experiment using aiido')
@click.option('--system_prompt', '-sys', default="", help='system prompt for the llm model')
def get_response(doc_path:str, prompt:str, embeddings:str, chunk_size:int, model:str, wwww:int, threshold:float,\
                 language:str, search_type:str, record_exp:str, system_prompt:str):
    
    res = ak.get_response(doc_path, prompt, embeddings, chunk_size\
                 , model, False, wwww, threshold,\
                 language , search_type, False, record_exp, \
                 system_prompt)
    
    print(res)


@click.command()
@click.option('--doc_path', '-d', help='document directory path, parse all .txt, .pdf, .docx files in the directory', required=True)
@click.option('--prompt','-p',multiple=True, help='prompt you want to ask to llm, if you want to ask multiple questions, use -p multiple times', required=True)
@click.option('--embeddings','-e', default="openai:text-embedding-ada-002", help='embeddings for storing the documents')
@click.option('--chunk_size', '-c', default = 1000, help='chunk size for storing the documents')
@click.option('--model', '-m', default="openai:gpt-3.5-turbo",help='llm model for generating the response')
@click.option('--topK', '-k', default=2, help='select topK relevant documents')
@click.option('--threshold', '-t', default=0.2, help='threshold score for selecting the relevant documents')
@click.option('--language', '-l', default='ch', help='language for the documents, default is \'ch\' for chinese')
@click.option('--search_type', '-s', default='merge', help='search type for the documents, include merge, svm, mmr, tfidf')
@click.option('--record_exp', '-r', default="", help='input the experiment name if you want to record the experiment using aiido')
@click.option('--system_prompt', '-sys', default="", help='system prompt for the llm model')
def chain_of_thought(doc_path:str, prompt, embeddings:str, chunk_size:int, model:str, topK:int, threshold:float,\
                 language:str, search_type:str, record_exp:str, system_prompt:str):
    
     res = ak.chain_of_thought(doc_path, prompt, embeddings, chunk_size\
                 , model, False, topK, threshold,\
                 language , search_type, False, record_exp, \
                 system_prompt)
     
     print(res)
     
     
     
akasha.add_command(get_response)
akasha.add_command(chain_of_thought)






if __name__ == '__main__':
    akasha()