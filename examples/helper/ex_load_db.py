import akasha.utils.db as adb

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
DEFAULT_EMBED = "openai:text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 1000

### you can use the process_db function to create chromadb of datasource and load it to dbs object(db) ###
### if the saved file and parameters are the same, the function will load the existing chromadb ###
## data_source can be file name, directory name, or url ##
data_source = ["docs/mic", "docs/1.pdf", "https://github.com/iii-org/akasha"]
db, ignore_files = adb.process_db(data_source=data_source, embeddings = DEFAULT_EMBED, chunk_size=DEFAULT_CHUNK_SIZE,\
    verbose=True)

### dbs object is a class that stores all information of the chromadb ###
db.get_docs()
db.get_embeds()
db.get_metadatas
db.get_ids()

### for each string in data_source, you can use get_storage_directory to get the storage directory of the chromadb ###
embed_type, embed_name = DEFAULT_EMBED.split(":")
chromadb_mic_dir = adb.get_storage_directory("docs/mic", DEFAULT_CHUNK_SIZE,
                                             embed_type, embed_name)

### after you created the chromadb, you can also load it by chroma_name ###
chroma_list = [chromadb_mic_dir]
db, ignore_files = adb.load_db_by_chroma_name(chroma_name_list=chroma_list)
#
#
#
#
### if you don't want to create chromadb, you can directly load the documents files and get the
### list of Document object(page_content=str, metadata=dict) ###
### info can be string text, file name, directory name, or url ###
docs = adb.load_docs_from_info(info=data_source, verbose=True)

print(docs[0].page_content)
