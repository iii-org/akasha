import akasha
import akasha.helper as ah
import akasha.utils.db as adb
from akasha.utils.search.retrievers.base import get_retrivers
from akasha.utils.search.search_doc import retri_docs

emb_name = "openai:text-embedding-ada-002"
emb_obj = ah.handle_embeddings(emb_name)
search_type = "auto"
query = "五軸是甚麼?"
model_name = "openai:gpt-3.5-turbo"
max_input_tokens = 3000

# 1. get dbs object
db, _ = adb.process_db(
    data_source="docs/mic", verbose=False, embeddings=emb_obj, chunk_size=1000
)

# 2. get retriver list
retriver_list = get_retrivers(db=db, embeddings=emb_obj, search_type=search_type)


### use search by yourself  ###
docs, doc_length, doc_tokens = akasha.search.search_docs(
    retriver_list,
    query,
    model=model_name,
    max_input_tokens=max_input_tokens,
    search_type=search_type,
    language="ch",
)

print(docs[0].page_content)  # docs is list of Documents
print(doc_length)  # integer
print(doc_tokens)  # integer


### get the search method sorting and score ###
# it can only be used in single search method (knn, svm, mmr, tfidf, bm25)
single_retriver = get_retrivers(db=db, embeddings=emb_obj, search_type="knn")[0]


# 3. get sorted list of Documents and scores by similarity
### this method is only for single search method, not for 'merge' and 'auto' ###
docs, scores = single_retriver.get_relevant_documents_and_scores(query)


print(docs[0].page_content)  # docs is list of Document


### use retri docs to get the sorting list of documents by similarity ###
# 3. get sorted list of Documents by similarity
docs = retri_docs(
    db,
    retriver_list,
    query,
    search_type=search_type,
    topK=100,
)

print(docs[0].page_content)  # docs is list of Document
