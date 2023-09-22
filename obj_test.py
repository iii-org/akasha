import akasha
import akasha.eval as eval
import akasha.summary as summary


query1 = "五軸是甚麼?"
query2 = ["西門子自有工廠如何朝工業4.0 發展","詳細解釋「工業4.0 成熟度指數」發展路徑的六個成熟度","根據西門子自有工廠朝工業4.0發展，探討其各項工業4.0的成熟度指標"]


## used for custom search type, use query_embeds and docs_embeds to determine relevant documents.
# you can add your own variables using log to record, for example log['dd'] = "miao"
def cust(query_embeds, docs_embeds, k:int, relevancy_threshold:float, log:dict):
    
    from scipy.spatial.distance import euclidean
    import numpy as np
    distance = [[euclidean(query_embeds, docs_embeds[idx]),idx] for idx in range(len(docs_embeds))]
    distance = sorted(distance, key=lambda x: x[0])
    
    log['dd'] = "miao"
    return  [idx for dist,idx in distance[:k] if (1 - dist) >= relevancy_threshold]




qa = akasha.Doc_QA(verbose=False, search_type="svm")

qa.get_response(doc_path="./../doc/mic/", prompt = query1, chunk_size = 500, record_exp="", search_type=cust,\
    max_token=3000, system_prompt="請你在回答前面加上喵")

print(qa.response)

### you can use qa.logs to get the log of each run, logs is a dictonary, for each run, the key is timestamp and the log of that run is the value. 
# use qa.timestamp_list to get the timestamp of each run, so you can get the log of each run by using it as key.
# timestamp_list = qa.timestamp_list
# print(qa.logs[timestamp_list[-1]])
# print(qa.logs[timestamp_list[-1]]['dd'])   # the variable you add to log in cust function
