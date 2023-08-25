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
@click.option('--topk', '-k', default = 2, help='select topK relevant documents')
@click.option('--threshold', '-t', default=0.2, help='threshold score for selecting the relevant documents')
@click.option('--language', '-l', default='ch', help='language for the documents, default is \'ch\' for chinese')
@click.option('--search_type', '-s', default='merge', help='search type for the documents, include merge, svm, mmr, tfidf')
@click.option('--record_exp', '-r', default="", help='input the experiment name if you want to record the experiment using aiido')
@click.option('--system_prompt', '-sys', default="", help='system prompt for the llm model')
def get_response(doc_path:str, prompt:str, embeddings:str, chunk_size:int, model:str, topk:int, threshold:float,\
                 language:str, search_type:str, record_exp:str, system_prompt:str):

    res = ak.get_response(doc_path, prompt, embeddings, chunk_size\
                 , model, False, topk, threshold,\
                 language , search_type, False, record_exp, \
                 system_prompt)
    
    print(res)


@click.command()
@click.option('--doc_path', '-d', help='document directory path, parse all .txt, .pdf, .docx files in the directory', required=True)
@click.option('--prompt','-p',multiple=True, help='prompt you want to ask to llm, if you want to ask multiple questions, use -p multiple times', required=True)
@click.option('--embeddings','-e', default="openai:text-embedding-ada-002", help='embeddings for storing the documents')
@click.option('--chunk_size', '-c', default = 1000, help='chunk size for storing the documents')
@click.option('--model', '-m', default="openai:gpt-3.5-turbo",help='llm model for generating the response')
@click.option('--topk', '-k', default=2, help='select topK relevant documents')
@click.option('--threshold', '-t', default=0.2, help='threshold score for selecting the relevant documents')
@click.option('--language', '-l', default='ch', help='language for the documents, default is \'ch\' for chinese')
@click.option('--search_type', '-s', default='merge', help='search type for the documents, include merge, svm, mmr, tfidf')
@click.option('--record_exp', '-r', default="", help='input the experiment name if you want to record the experiment using aiido')
@click.option('--system_prompt', '-sys', default="", help='system prompt for the llm model')
def chain_of_thought(doc_path:str, prompt, embeddings:str, chunk_size:int, model:str, topk:int, threshold:float,\
                 language:str, search_type:str, record_exp:str, system_prompt:str):
    
     res = ak.chain_of_thought(doc_path, prompt, embeddings, chunk_size\
                 , model, False, topk, threshold,\
                 language , search_type, False, record_exp, \
                 system_prompt)
     
     print(res)






@click.command()
@click.option('--q_file', '-q', help='question set file path, each line is a question', required=True)
@click.option('--doc_path', '-d', help='document directory path, parse all .txt, .pdf, .docx files in the directory', required=True)
@click.option('--embeddings','-e', default="openai:text-embedding-ada-002", help='embeddings for storing the documents')
@click.option('--chunk_size', '-c', default = 1000, help='chunk size for storing the documents')
@click.option('--model', '-m', default="openai:gpt-3.5-turbo",help='llm model for generating the response')
@click.option('--topk', '-k', default=2, help='select topK relevant documents')
@click.option('--threshold', '-t', default=0.2, help='threshold score for selecting the relevant documents')
@click.option('--language', '-l', default='ch', help='language for the documents, default is \'ch\' for chinese')
@click.option('--search_type', '-s', default='merge', help='search type for the documents, include merge, svm, mmr, tfidf')
@click.option('--record_exp', '-r', default="", help='input the experiment name if you want to record the experiment using aiido')
def test_performance(q_file:str, doc_path:str, embeddings:str, chunk_size:int, model:str, topk:int, threshold:float,\
                 language:str, search_type:str, record_exp:str):


    cor_rate, tokens = ak.test_performance(q_file, doc_path, embeddings, chunk_size, model, topk, threshold,\
                 language , search_type, False, record_exp) 
    print("correct rate: ", cor_rate)
    print("total tokens: ", tokens)
    




@click.command()
@click.option('--doc_path', '-d', help='document directory path, parse all .txt, .pdf, .docx files in the directory', required=True)
@click.option('question_num', '-qn', default=10, help='number of questions you want to generate')
@click.option('--embeddings','-e', default="openai:text-embedding-ada-002", help='embeddings for storing the documents')
@click.option('--chunk_size', '-c', default = 1000, help='chunk size for storing the documents')
@click.option('--topk', '-k', default=2, help='select topK relevant documents')
@click.option('--threshold', '-t', default=0.2, help='threshold score for selecting the relevant documents')
@click.option('--language', '-l', default='ch', help='language for the documents, default is \'ch\' for chinese')
@click.option('--search_type', '-s', default='merge', help='search type for the documents, include merge, svm, mmr, tfidf')
@click.option('--record_exp', '-r', default="", help='input the experiment name if you want to record the experiment using aiido')
def auto_create_questionset(doc_path:str, question_num:int , embeddings:str , chunk_size:int\
                 , topk:int, threshold:float,\
                 language:str, search_type:str , record_exp:str):

    model = "openai:gpt-3.5-turbo"
    
    ak.auto_create_questionset(doc_path, question_num, embeddings, chunk_size\
                 , model, False, topk, threshold, language, search_type, record_exp) 





@click.command()
@click.option('--question_path', '-qp', help='document directory path, parse all .txt, .pdf, .docx files in the directory', required=True)
@click.option('--doc_path', '-d', help='document directory path, parse all .txt, .pdf, .docx files in the directory', required=True)
@click.option('--embeddings','-e', default="openai:text-embedding-ada-002", help='embeddings for storing the documents')
@click.option('--chunk_size', '-c', default = 1000, help='chunk size for storing the documents')
@click.option('--model', '-m', default="openai:gpt-3.5-turbo",help='llm model for generating the response')
@click.option('--topk', '-k', default=2, help='select topK relevant documents')
@click.option('--threshold', '-t', default=0.2, help='threshold score for selecting the relevant documents')
@click.option('--language', '-l', default='ch', help='language for the documents, default is \'ch\' for chinese')
@click.option('--search_type', '-s', default='merge', help='search type for the documents, include merge, svm, mmr, tfidf')
@click.option('--record_exp', '-r', default="", help='input the experiment name if you want to record the experiment using aiido')
def auto_evaluation(question_path:str, doc_path:str, embeddings:str, chunk_size:int\
                 , model:str, topK:int, threshold:float,\
                    language:str , search_type:str , record_exp:str):
    
    
    
    avg_bert, avg_rouge = ak.auto_evaluation(question_path, doc_path, embeddings, chunk_size\
                 , model, False, topK, threshold,\
                 language, search_type, record_exp)

    print("avg bert score: ", avg_bert)
    print("average rouge score: ", avg_rouge)









akasha.add_command(get_response)
akasha.add_command(chain_of_thought)
akasha.add_command(test_performance)
akasha.add_command(auto_create_questionset)
akasha.add_command(auto_evaluation)



if __name__ == '__main__':
    akasha()