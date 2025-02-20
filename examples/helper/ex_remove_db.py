import akasha.utils.db as adb

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
DEFAULT_EMBED = "openai:text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 1000

data_source = ["docs/mic", "docs/1.pdf", "https://github.com/iii-org/akasha"]
db, ignore_files = adb.process_db(data_source=data_source, embeddings = DEFAULT_EMBED, chunk_size=DEFAULT_CHUNK_SIZE,\
    verbose=True)

### you can use extrace_db functions to extract the certain file or documents from the chromadb ###
# extract the data by the file name
extracted_db = adb.extract_db_by_file(db, ["docs/1.pdf"])

# extract the data by keywords, it the document contains the keyword, it will be extracted
extracted_db = adb.extract_db_by_keyword(db, ["工業4.0"])

# extract the data by the ids, this will extract the id[0] and id[1] to become new extracted_db
id_list = db.get_ids()
extracted_db = adb.extract_db_by_ids(db, [id_list[0], id_list[1]])

# pop the data by the ids, this will pop out id[0] and id[1] from db
id_list = db.get_ids()
adb.pop_db_by_ids(db, [id_list[0], id_list[1]])
#
#
#
#
#
#### if you want to remove the chromadb, or remove the certain file from the chromadb, you can use the following functions ####
delete_num = adb.delete_documents_by_directory("docs/mic", DEFAULT_EMBED,
                                               DEFAULT_CHUNK_SIZE)
delete_num = adb.delete_documents_by_file("docs/1.pdf", DEFAULT_EMBED,
                                          DEFAULT_CHUNK_SIZE)
