import click

import akasha as ak
import akasha.eval.eval as eval
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
@click.option('--max_token', '-mt', default=3000, help='max token for the llm model input')
def get_response(doc_path:str, prompt:str, embeddings:str, chunk_size:int, model:str, topk:int, threshold:float,\
                 language:str, search_type:str, record_exp:str, system_prompt:str, max_token:int):

    res = ak.get_response(doc_path, prompt, embeddings, chunk_size\
                 , model, False, topk, threshold,\
                 language , search_type, False, record_exp, \
                 system_prompt, max_token)
    
    print(res)









@click.command()
@click.option('--doc_path', '-d', help='document directory path, parse all .txt, .pdf, .docx files in the directory', required=True)
@click.option('--embeddings','-e', default="openai:text-embedding-ada-002", help='embeddings for storing the documents')
@click.option('--chunk_size', '-c', default = 1000, help='chunk size for storing the documents')
@click.option('--model', '-m', default="openai:gpt-3.5-turbo",help='llm model for generating the response')
@click.option('--topk', '-k', default = 2, help='select topK relevant documents')
@click.option('--threshold', '-t', default=0.2, help='threshold score for selecting the relevant documents')
@click.option('--language', '-l', default='ch', help='language for the documents, default is \'ch\' for chinese')
@click.option('--search_type', '-s', default='merge', help='search type for the documents, include merge, svm, mmr, tfidf')
@click.option('--system_prompt', '-sys', default="", help='system prompt for the llm model')
@click.option('--max_token', '-mt', default=3000, help='max token for the llm model input')
def keep_responsing(doc_path:str, embeddings:str, chunk_size:int, model:str, topk:int, threshold:float,\
                 language:str, search_type:str, system_prompt:str, max_token:int):

    import akasha.helper as helper
    import akasha.search as search
    from langchain.chains.question_answering import load_qa_chain
    logs = ["\n\n-----------------keep_response----------------------\n"]
    embeddings_name = embeddings
    embeddings = helper.handle_embeddings(embeddings, logs, False)
    model = helper.handle_model(model, logs, False)
    
    

    db = helper.create_chromadb(doc_path, logs, False, embeddings, embeddings_name, chunk_size)

    if db is None:
        info = "document path not exist\n"
        print(info)
        logs.append(info)
        helper.save_logs(logs)
        return ""


    user_input = click.prompt("Please input your question(type \"exit()\" to quit) ")
    while user_input != "exit()":
        
        docs, tokens = search.get_docs(db, embeddings, user_input, topk, threshold, language, search_type, False,\
                        logs, model, False, max_token)
        if docs is None:
            docs = []
        
        
        
        chain = load_qa_chain(llm=model, chain_type="stuff",verbose=False)

        logs.append("\n\ndocuments: \n\n" + ''.join([doc.page_content for doc in docs]))
        
        res = chain.run(input_documents=docs, question=system_prompt + user_input)
        res =  helper.sim_to_trad(res)
        response = res.split("Finished chain.")
        
        
        logs.append("\n\nresponse:\n\n"+ response[-1])
        
        
        print("Response: ",res)
        print("\n\n")
        user_input = click.prompt("Please input your question(type \"exit()\" to quit) ")












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
@click.option('--max_token', '-mt', default=3000, help='max token for the llm model input')
def chain_of_thought(doc_path:str, prompt, embeddings:str, chunk_size:int, model:str, topk:int, threshold:float,\
                 language:str, search_type:str, record_exp:str, system_prompt:str, max_token:int):
    
     res = ak.chain_of_thought(doc_path, prompt, embeddings, chunk_size\
                 , model, False, topk, threshold,\
                 language , search_type, False, record_exp, \
                 system_prompt, max_token)
     for r in res:
        print(r)





@click.command()
@click.option('--doc_path', '-d', help='document directory path, parse all .txt, .pdf, .docx files in the directory', required=True)
@click.option('-question_num', '-qn', default=10, help='number of questions you want to generate')
@click.option('-question_type','--qt', default="essay", help='question type, include single_choice, essay')
@click.option('--embeddings','-e', default="openai:text-embedding-ada-002", help='embeddings for storing the documents')
@click.option('--chunk_size', '-c', default = 1000, help='chunk size for storing the documents')
@click.option('--topk', '-k', default=2, help='select topK relevant documents')
@click.option('--threshold', '-t', default=0.2, help='threshold score for selecting the relevant documents')
@click.option('--language', '-l', default='ch', help='language for the documents, default is \'ch\' for chinese')
@click.option('--search_type', '-s', default='merge', help='search type for the documents, include merge, svm, mmr, tfidf')
@click.option('--record_exp', '-r', default="", help='input the experiment name if you want to record the experiment using aiido')
def auto_create_questionset(doc_path:str, question_num:int , question_type:str, embeddings:str , chunk_size:int\
                 , topk:int, threshold:float,\
                 language:str, search_type:str , record_exp:str):

    model = "openai:gpt-3.5-turbo"
    
    eval.auto_create_questionset(doc_path, question_num, question_type, embeddings, chunk_size\
                 , model, False, topk, threshold, language, search_type, record_exp) 





@click.command()
@click.option('--question_path', '-qp', help='document directory path, parse all .txt, .pdf, .docx files in the directory', required=True)
@click.option('--doc_path', '-d', help='document directory path, parse all .txt, .pdf, .docx files in the directory', required=True)
@click.option('--question_type', '-qt', default="essay", help='question type, include single_choice, essay')
@click.option('--embeddings','-e', default="openai:text-embedding-ada-002", help='embeddings for storing the documents')
@click.option('--chunk_size', '-c', default = 1000, help='chunk size for storing the documents')
@click.option('--model', '-m', default="openai:gpt-3.5-turbo",help='llm model for generating the response')
@click.option('--topk', '-k', default=2, help='select topK relevant documents')
@click.option('--threshold', '-t', default=0.2, help='threshold score for selecting the relevant documents')
@click.option('--language', '-l', default='ch', help='language for the documents, default is \'ch\' for chinese')
@click.option('--search_type', '-s', default='merge', help='search type for the documents, include merge, svm, mmr, tfidf')
@click.option('--record_exp', '-r', default="", help='input the experiment name if you want to record the experiment using aiido')
@click.option('--max_token', '-mt', default=3000, help='max token for the llm model input')
def auto_evaluation(question_path:str, doc_path:str, question_type:str, embeddings:str, chunk_size:int\
                 , model:str, topk:int, threshold:float,\
                    language:str , search_type:str , record_exp:str, max_token:int):
    
    
    if question_type.lower() == "single_choice":
        cor_rate, tokens = eval.auto_evaluation(question_path, doc_path, question_type, embeddings, chunk_size\
                 , model, False, topk, threshold,\
                 language, search_type, record_exp, max_token)
        
        print("correct rate: ", cor_rate)
        print("total tokens: ", tokens)
    
    else:
        avg_bert, avg_rouge, avg_llm, tokens = eval.auto_evaluation(question_path, doc_path, question_type, embeddings, chunk_size\
                 , model, False, topk, threshold,\
                 language, search_type, record_exp, max_token)

        print("avg bert score: ", avg_bert)
        print("average rouge score: ", avg_rouge)
        print("avg llm score: ", avg_llm)
        print("total tokens: ", tokens)







akasha.add_command(keep_responsing)
akasha.add_command(get_response)
akasha.add_command(chain_of_thought)
akasha.add_command(auto_create_questionset)
akasha.add_command(auto_evaluation)



if __name__ == '__main__':
    akasha()