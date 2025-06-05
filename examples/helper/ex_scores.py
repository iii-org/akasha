import akasha.helper as ah

DEFAULT_MODEL = "openai:gpt-4o"

rf1 = """工業4.0是指智慧與互聯的生產系統，旨在感知、預測物理世界並與之互動，以便即時做出支持生產的決策。這一概念最早於2011年在漢諾威工業博覽會上提出，並被
納入德國、美國、中國等國家的高科技產業發展策略中。工業4.0的核心在於利用物聯網、人工智慧、大數據等技術，將生產從自動化提升到智慧化，實現更高效的生
產管理和決策。"""

rf2 = """工業 4.0 是一種智慧與互聯的生產系統，旨在感知、預測物理世界並與之互動，以便即時做出支持生產的決策。  
簡單來說，它利用像是物聯網、AI、大數據和雲端等技術，讓製造業更加智能化和自動化。
"""

### use bert to get the similarity score ###
bert_score = ah.get_bert_score(rf1, rf2)

### use rouge to get the similarity score ###
rouge_score = ah.get_rouge_score(rf1, rf2)

### use llm to get the similarity score ###
llm_score = ah.get_llm_score(rf1, rf2, DEFAULT_MODEL)
