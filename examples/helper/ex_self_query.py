import akasha.utils.db as adb
import akasha.helper as ah
from akasha.utils.search.retrievers.base import get_retrivers
import akasha

#### self query method helps you to filter the documents in the db by metadata ####
#### it will use the metadata and the prompt to generate a query ####
#### and then use the query to filter the documents in the db ####


##### 1. load db from data source and add metadata to db ######
data_source = ["docs/pns_query_small"]
embed_name = "openai:text-embedding-3-small"
chunk_size = 1000
db, ignore_files = adb.process_db(
    data_source=data_source, embeddings=embed_name, chunk_size=chunk_size, verbose=True
)


### you should create your own metadata function to add metadata for every text chunks in the db ###
### in this example, we use the source(file name) of the text chunk to map text chunks and metadatas ###
def add_metadata(db: adb.dbs):
    """this function is used to add metadata to the old_db object.

    Args:
        old_db (adb.dbs):

    Returns:
        (adb.dbs):
    """
    import json
    from pathlib import Path

    for metadata in db.metadatas:
        file_path = metadata["source"]  # source is the file path
        try:
            with Path(file_path).open("r", encoding="utf-8") as file:
                dictionary = json.load(file)

            # dictionary = helper.extract_json(text)
            metadata["課別"] = dictionary["課別"]
            metadata["業務擔當"] = dictionary["業務擔當"]
            ddate = dictionary["拜訪日期"]
            metadata["拜訪年"] = int(ddate.split("-")[0])
            metadata["拜訪月"] = int(ddate.split("-")[1])
            metadata["產品"] = dictionary["產品"]
            metadata["公司名稱"] = dictionary["公司名稱"]
            metadata["大分類"] = dictionary["大分類"]
            metadata["中分類"] = dictionary["中分類"]
        except Exception as e:
            print(f"JSONDecodeError: {e}")
    return


add_metadata(db)
adb.update_db(db, data_source, embed_name, chunk_size=chunk_size)


##### 2. use self-query to filter docs and use query for similarity search ######

db, ignore_files = adb.process_db(
    data_source=data_source, embeddings=embed_name, chunk_size=chunk_size, verbose=True
)
prompt = "A公司在2024年對電動車的銷售量為多少?"
search_type = "knn"
model_obj = ah.handle_model("openai:gpt-4o", True)


## each metadata attribute should include name, description and type(integer, float, string) ##
metadata_field_info = [
    {"name": "拜訪年", "description": "此訪談紀錄的拜訪年份", "type": "integer"},
    {"name": "拜訪月", "description": "此訪談紀錄的拜訪月份", "type": "integer"},
    {"name": "業務擔當", "description": "業務的名稱", "type": "string"},
    {"name": "中分類", "description": "訪談產品的中等分類", "type": "string"},
    {"name": "公司名稱", "description": "訪談對象的公司名稱", "type": "string"},
    {"name": "大分類", "description": "訪談產品的大分類", "type": "string"},
    {"name": "產品", "description": "訪談的產品名稱/型號", "type": "string"},
    {"name": "課別", "description": "公司部門的課別名稱或代號", "type": "string"},
]

document_content_description = "業務與客戶的訪談紀錄"

####################


### use self-query to filter docs
new_dbs, query, matched_fields = ah.self_query(
    prompt, model_obj, db, metadata_field_info, document_content_description
)

### option1   use knn similarity search to sort docs from filtered docs

retriver = get_retrivers(new_dbs, embed_name, threshold=0.0, search_type=search_type)[0]

docs, scores = retriver.get_relevant_documents_and_scores(query)
print(docs)

### option2    use new_dbs(filtered docs) to run other akasha functions


ak = akasha.RAG(
    model=model_obj,
    embeddings=embed_name,
)
resposne = ak(data_source=new_dbs, prompt=prompt)
