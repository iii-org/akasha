import click
import uvicorn
import akasha as ak


@click.group()
def akasha():
    pass


@click.command()
@click.option(
    "--data_source",
    "-d",
    help="document directory path, or url, parse all .txt, .pdf, .docx files in the directory",
    required=True,
)
@click.option("--prompt", "-p", help="prompt you want to ask to llm", required=True)
@click.option(
    "--embeddings",
    "-e",
    default="openai:text-embedding-ada-002",
    help="embeddings for storing the documents",
)
@click.option(
    "--chunk_size", "-c", default=1000, help="chunk size for storing the documents"
)
@click.option(
    "--model",
    "-m",
    default="openai:gpt-3.5-turbo",
    help="llm model for generating the response",
)
@click.option(
    "--language",
    "-l",
    default="ch",
    help="language for the documents, default is 'ch' for chinese",
)
@click.option(
    "--search_type",
    "-s",
    default="auto",
    help="search type for the documents, include auto, knn, svm, bm25",
)
@click.option(
    "--record_exp",
    "-r",
    default="",
    help="input the experiment name if you want to record the experiment using aiido",
)
@click.option(
    "--system_prompt", "-sys", default="", help="system prompt for the llm model"
)
@click.option(
    "--max_input_tokens", "-md", default=3000, help="max token for the llm model input"
)
def rag(
    data_source: str,
    prompt: str,
    embeddings: str,
    chunk_size: int,
    model: str,
    language: str,
    search_type: str,
    record_exp: str,
    system_prompt: str,
    max_input_tokens: int,
):
    gr = ak.RAG(
        verbose=False,
        embeddings=embeddings,
        chunk_size=chunk_size,
        model=model,
        language=language,
        search_type=search_type,
        record_exp=record_exp,
        system_prompt=system_prompt,
        max_input_tokens=max_input_tokens,
    )
    res = gr(data_source, prompt)

    print(res)

    del gr


@click.command()
@click.option(
    "--data_source",
    "-d",
    help="document directory path, or urls, parse all .txt, .pdf, .docx files in the directory",
    required=True,
)
@click.option(
    "--embeddings",
    "-e",
    default="openai:text-embedding-ada-002",
    help="embeddings for storing the documents",
)
@click.option(
    "--chunk_size", "-c", default=1000, help="chunk size for storing the documents"
)
@click.option(
    "--model",
    "-m",
    default="openai:gpt-3.5-turbo",
    help="llm model for generating the response",
)
@click.option(
    "--language",
    "-l",
    default="ch",
    help="language for the documents, default is 'ch' for chinese",
)
@click.option(
    "--search_type",
    "-s",
    default="merge",
    help="search type for the documents, include merge, svm, mmr, tfidf",
)
@click.option(
    "--system_prompt", "-sys", default="", help="system prompt for the llm model"
)
@click.option(
    "--max_input_tokens", "-md", default=3000, help="max token for the llm model input"
)
def keep_rag(
    data_source: str,
    embeddings: str,
    chunk_size: int,
    model: str,
    language: str,
    search_type: str,
    system_prompt: str,
    max_input_tokens: int,
):
    import akasha.helper as helper
    from langchain.chains.question_answering import load_qa_chain
    import akasha.utils.db as dd
    from akasha.utils.search.retrievers.base import get_retrivers
    from akasha.utils.search.search_doc import search_docs

    embeddings = helper.handle_embeddings(embeddings, False)
    model_name = model
    model = helper.handle_model(model, False)

    db, ign = dd.process_db(data_source, embeddings, chunk_size)

    if db is None:
        info = "document path not exist\n"
        print(info)
        return ""

    user_input = click.prompt('Please input your question(type "exit()" to quit) ')
    retrivers_list = get_retrivers(db, embeddings, 0.0, search_type)

    while user_input != "exit()":
        docs, docs_len, tokens = search_docs(
            retrivers_list,
            user_input,
            model_name,
            max_input_tokens,
            search_type,
            language,
        )
        if docs is None:
            docs = []

        chain = load_qa_chain(llm=model, chain_type="stuff", verbose=False)

        res = chain.run(input_documents=docs, question=system_prompt + user_input)
        res = helper.sim_to_trad(res)

        print("Response: ", res)
        print("\n\n")
        user_input = click.prompt('Please input your question(type "exit()" to quit) ')

    del db, model, embeddings


@click.command()
@click.option(
    "--data_source",
    "-d",
    help="document directory path, parse all .txt, .pdf, .docx files in the directory",
    required=True,
)
@click.option(
    "-question_num", "-qn", default=10, help="number of questions you want to generate"
)
@click.option(
    "-question_type",
    "--qt",
    default="essay",
    help="question type, include single_choice, essay",
)
@click.option(
    "--embeddings",
    "-e",
    default="openai:text-embedding-ada-002",
    help="embeddings for storing the documents",
)
@click.option(
    "--chunk_size", "-c", default=1000, help="chunk size for storing the documents"
)
@click.option(
    "--language",
    "-l",
    default="ch",
    help="language for the documents, default is 'ch' for chinese",
)
@click.option(
    "--search_type",
    "-s",
    default="merge",
    help="search type for the documents, include merge, svm, mmr, tfidf",
)
@click.option(
    "--record_exp",
    "-r",
    default="",
    help="input the experiment name if you want to record the experiment using aiido",
)
def create_questionset(
    data_source: str,
    question_num: int,
    question_type: str,
    embeddings: str,
    chunk_size: int,
    language: str,
    search_type: str,
    record_exp: str,
):
    model = "openai:gpt-3.5-turbo"
    eva = ak.eval(
        model=model,
        embeddings=embeddings,
    )
    eva.create_questionset(
        data_source,
        question_num,
        question_type=question_type,
        chunk_size=chunk_size,
        verbose=False,
        language=language,
        search_type=search_type,
        record_exp=record_exp,
    )

    del eva


@click.command()
@click.option(
    "--question_path",
    "-qp",
    help="document directory path, parse all .txt, .pdf, .docx files in the directory",
    required=True,
)
@click.option(
    "--data_source",
    "-d",
    help="document directory path, parse all .txt, .pdf, .docx files in the directory",
    required=True,
)
@click.option(
    "--question_type",
    "-qt",
    default="essay",
    help="question type, include single_choice, essay",
)
@click.option(
    "--embeddings",
    "-e",
    default="openai:text-embedding-ada-002",
    help="embeddings for storing the documents",
)
@click.option(
    "--chunk_size", "-c", default=1000, help="chunk size for storing the documents"
)
@click.option(
    "--model",
    "-m",
    default="openai:gpt-3.5-turbo",
    help="llm model for generating the response",
)
@click.option("--topk", "-k", default=2, help="select topK relevant documents")
@click.option(
    "--language",
    "-l",
    default="ch",
    help="language for the documents, default is 'ch' for chinese",
)
@click.option(
    "--search_type",
    "-s",
    default="merge",
    help="search type for the documents, include merge, svm, mmr, tfidf",
)
@click.option(
    "--record_exp",
    "-r",
    default="",
    help="input the experiment name if you want to record the experiment using aiido",
)
@click.option(
    "--max_input_tokens", "-md", default=3000, help="max token for the llm model input"
)
def evaluation(
    question_path: str,
    data_source: str,
    question_type: str,
    embeddings: str,
    chunk_size: int,
    model: str,
    language: str,
    search_type: str,
    record_exp: str,
    max_input_tokens: int,
):
    eva = ak.eval(model=model, embeddings=embeddings)

    if question_type.lower() == "single_choice":
        cor_rate, tokens = eva.evaluation(
            question_path,
            data_source,
            question_type=question_type,
            chunk_size=chunk_size,
            verbose=False,
            language=language,
            search_type=search_type,
            record_exp=record_exp,
            max_input_tokens=max_input_tokens,
        )

        print("correct rate: ", cor_rate)
        print("total tokens: ", tokens)

    else:
        avg_bert, avg_rouge, avg_llm, tokens = eva.evaluation(
            question_path,
            data_source,
            question_type=question_type,
            chunk_size=chunk_size,
            verbose=False,
            language=language,
            search_type=search_type,
            record_exp=record_exp,
            max_input_tokens=max_input_tokens,
        )

        print("avg bert score: ", avg_bert)
        print("average rouge score: ", avg_rouge)
        print("avg llm score: ", avg_llm)
        print("total tokens: ", tokens)


@click.command("toy", short_help="simple toy for akasha")
def ui():
    import os
    import sys
    import site
    from streamlit import config as _config
    from streamlit.web import cli as stcli

    # make a folder `docs/Default`
    if not os.path.exists("docs") or not os.path.exists(
        os.path.join("docs", "Default")
    ):
        os.makedirs(os.path.join(".", "docs", "Default"))
    else:
        pass

    if not os.path.exists("docs"):
        os.makedirs(os.path.join(".", "docs", "Default"))
    else:
        pass

    # make a folder `model`
    if not os.path.exists("model"):
        os.makedirs("model")
    else:
        pass

    # find the location of ui.py
    site_packages_dirs = site.getsitepackages()
    for dir in site_packages_dirs:
        if dir.endswith("site-packages"):
            target_dir = dir
            break
        else:
            target_dir = "."

    # run streamlit web service by ui.py
    _config.set_option("server.headless", True)

    ui_py_file = os.path.join(target_dir, "akasha", "ui.py")
    # streamlit.web.bootstrap.run(ui_py_file, "", [], [])
    sys.argv = ["streamlit", "run", ui_py_file]
    sys.exit(stcli.main())


@click.command("api", short_help="simple api for akasha")
@click.option(
    "--workers",
    "-w",
    default=2,
    help="Number of workers to use",
)
@click.option(
    "--host",
    "-h",
    default="0.0.0.0",
    help="Host to run the FastAPI server on",
)
@click.option(
    "--port",
    "-p",
    default=8000,
    help="Port to run the FastAPI server on",
)
def start_fastapi(workers: int, host: str, port: str):
    uvicorn.run("akasha.api:app", host=host, port=port, workers=workers)


akasha.add_command(keep_rag)
akasha.add_command(rag)
akasha.add_command(create_questionset)
akasha.add_command(evaluation)
akasha.add_command(ui)
akasha.add_command(start_fastapi)

if __name__ == "__main__":
    akasha()
