# Graph Report - akasha  (2026-07-02)

## Corpus Check
- 119 files · ~133,707 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1086 nodes · 2161 edges · 89 communities (85 shown, 4 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8131cc0a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_evaluation.py|evaluation.py]]
- [[_COMMUNITY_agents|agents]]
- [[_COMMUNITY_tests|tests.md]]
- [[_COMMUNITY_gen_prompt.py|gen_prompt.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_dbs|dbs]]
- [[_COMMUNITY_self_query_filter.py|self_query_filter.py]]
- [[_COMMUNITY_atman|atman]]
- [[_COMMUNITY_handle_objects.py|handle_objects.py]]
- [[_COMMUNITY_model_eval.py|model_eval.py]]
- [[_COMMUNITY_.__call__|.__call__]]
- [[_COMMUNITY_.__call__|.__call__]]
- [[_COMMUNITY_anthropic_model|anthropic_model]]
- [[_COMMUNITY_gemini_model|gemini_model]]
- [[_COMMUNITY_AzureOpenAIClient|AzureOpenAIClient]]
- [[_COMMUNITY_remote_model|remote_model]]
- [[_COMMUNITY_create_db.py|create_db.py]]
- [[_COMMUNITY_gptq|gptq]]
- [[_COMMUNITY_ask.py|ask.py]]
- [[_COMMUNITY_custom_embed|custom_embed]]
- [[_COMMUNITY_ui.py|ui.py]]
- [[_COMMUNITY_api.py|api.py]]
- [[_COMMUNITY_hf_model|hf_model]]
- [[_COMMUNITY_self_ask.py|self_ask.py]]
- [[_COMMUNITY_Installation|Installation]]
- [[_COMMUNITY_Akasha Upgrade Plan Light Version Support|Akasha Upgrade Plan: Light Version Support]]
- [[_COMMUNITY_delete_documents_by_file|delete_documents_by_file]]
- [[_COMMUNITY_file_loader.py|file_loader.py]]
- [[_COMMUNITY_configure_logging|configure_logging]]
- [[_COMMUNITY_search_docs|search_docs]]
- [[_COMMUNITY_LlamaCPP|LlamaCPP]]
- [[_COMMUNITY_search_doc.py|search_doc.py]]
- [[_COMMUNITY_.__call__|.__call__]]
- [[_COMMUNITY_basic_llm|basic_llm]]
- [[_COMMUNITY_gemini_embed|gemini_embed]]
- [[_COMMUNITY_myFAISSRetriever|myFAISSRetriever]]
- [[_COMMUNITY_myKNNRetriever|myKNNRetriever]]
- [[_COMMUNITY_myTFIDFRetriever|myTFIDFRetriever]]
- [[_COMMUNITY_handle_embeddings_and_name|handle_embeddings_and_name]]
- [[_COMMUNITY_MemoryManager|MemoryManager]]
- [[_COMMUNITY_db_structure.py|db_structure.py]]
- [[_COMMUNITY_LLM|LLM]]
- [[_COMMUNITY_mySVMRetriever|mySVMRetriever]]
- [[_COMMUNITY_BaseRetriever|BaseRetriever]]
- [[_COMMUNITY_test_api_stability.py|test_api_stability.py]]
- [[_COMMUNITY_get_retrivers|get_retrivers]]
- [[_COMMUNITY_aiido_upload|aiido_upload]]
- [[_COMMUNITY_customRetriever|customRetriever]]
- [[_COMMUNITY_myMMRRetriever|myMMRRetriever]]
- [[_COMMUNITY_myRerankRetriever|myRerankRetriever]]
- [[_COMMUNITY_test_openai_rag.py|test_openai_rag.py]]
- [[_COMMUNITY_test_light_restrictions.py|test_light_restrictions.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_load_docs_from_webengine|load_docs_from_webengine]]
- [[_COMMUNITY_test_agents_observation_normalization.py|test_agents_observation_normalization.py]]
- [[_COMMUNITY_anthropic_rag|anthropic_rag]]
- [[_COMMUNITY_gemini_rag|gemini_rag]]
- [[_COMMUNITY__generate_single_choice_question|_generate_single_choice_question]]
- [[_COMMUNITY_test_live_model_final_action_aliases|test_live_model_final_action_aliases]]
- [[_COMMUNITY_pytest_configure|pytest_configure]]
- [[_COMMUNITY_hello memory|hello memory.md]]
- [[_COMMUNITY_akasha_terminal|akasha_terminal]]

## God Nodes (most connected - your core abstractions)
1. `dbs` - 58 edges
2. `format_sys_prompt()` - 35 edges
3. `call_model()` - 33 edges
4. `get_doc_length()` - 26 edges
5. `handle_model()` - 24 edges
6. `create_directory_db()` - 22 edges
7. `process_db()` - 21 edges
8. `get_retrivers()` - 21 edges
9. `Model_Eval` - 20 edges
10. `separate_name()` - 20 edges

## Surprising Connections (you probably didn't know these)
- `base_line()` --calls--> `RAG`  [EXTRACTED]
  tests/test_akasha.py → akasha/RAG/rag.py
- `test_RAG()` --references--> `RAG`  [EXTRACTED]
  tests/test_akasha.py → akasha/RAG/rag.py
- `base_line()` --calls--> `eval`  [EXTRACTED]
  tests/test_eval.py → akasha/eval/evaluation.py
- `test_Model_Eval()` --references--> `eval`  [EXTRACTED]
  tests/test_eval.py → akasha/eval/evaluation.py
- `add_metadata()` --references--> `dbs`  [EXTRACTED]
  examples/helper/ex_self_query.py → akasha/utils/db/db_structure.py

## Import Cycles
- None detected.

## Communities (89 total, 4 thin omitted)

### Community 0 - "evaluation.py"
Cohesion: 0.05
Nodes (44): check_essay_system_prompt(), check_sum_type(), find_same_category(), get_non_repeat_rand_int(), get_question_from_file(), get_source_files(), load questions from file and save the questions into lists.     a question list, iterate the category dictionary and check if any category has more than cate_thr (+36 more)

### Community 1 - "agents"
Cohesion: 0.06
Nodes (37): calculate_tool(), _jsonSaveTool(), BaseTool, rag_tool(), return the tool to use search engine to search information for user      Args:, return the json save tool that can save the content into json file.      Retur, save content into json file, saveJSON_tool() (+29 more)

### Community 2 - "tests.md"
Cohesion: 0.05
Nodes (40): 1.1 RAG 核心, 1.1 遠端模型與基礎問答, 1.2 摘要功能 (Summary), 1.2 核心邏輯與數據處理, 1.3 代理人功能 (Agents), 1.3 輕量檢索器, 1.4 評估功能 (Eval), 1.5 影像處理與多模態 (Vision & Multimodal) (+32 more)

### Community 3 - "gen_prompt.py"
Cohesion: 0.09
Nodes (29): merge_history_and_prompt(), merge system prompt, history messages, and prompt based on the prompt format typ, reference docs after calling the rag function, will return the reference file na, compare_question_prompt(), decide_auto_prompt_format_type(), default_get_reference_prompt(), format_category_prompt(), format_chat_gemini_prompt() (+21 more)

### Community 4 - "__init__.py"
Cohesion: 0.13
Nodes (31): extract_json(), extract_multiple_json(), Any, Extract JSON data from text generated by LLMs.     Handles both individual dict, Extract multiple JSON objects from text generated by LLMs.      Args:, convert simplified chinese to traditional chinese      Args:         **text (, sim_to_trad(), handle_model_and_name() (+23 more)

### Community 5 - "dbs"
Cohesion: 0.13
Nodes (19): dbs, extract_db_by_file(), extract_db_by_ids(), extract_db_by_keyword(), pop_db_by_ids(), pop undesired data from  dbs based on ids      Args:         db (dbs): dbs ob, extract db from dbs based on keyword_list      Args:         db (dbs): dbs ob, extract db from dbs based on ids      Args:         db (dbs): dbs object (+11 more)

### Community 6 - "self_query_filter.py"
Cohesion: 0.10
Nodes (27): check_metadata_info(), DocumentCP, filter_docs(), find_subset(), generate_query_constructor(), generate_query_filter(), handle_attr(), Any (+19 more)

### Community 7 - "atman"
Cohesion: 0.08
Nodes (15): BaseLanguageModel, Embeddings, Path, RAG, input the documents directory path and question, will first store the documents, input the documents directory path and question, will first store the documents, class for implement search db based on user prompt and generate response from ll, initials of Doc_QA class          Args:             embeddings (_type_, optio (+7 more)

### Community 8 - "handle_objects.py"
Cohesion: 0.11
Nodes (23): separate type:name by ':'      Args:         **name (str)**: string with form, separate_name(), _get_env_var(), _handle_azure_env(), handle_client(), handle_embeddings(), handle_model(), handle_model_type() (+15 more)

### Community 9 - "model_eval.py"
Cohesion: 0.11
Nodes (23): extract_result(), get_torch(), generate fact resposne from the question, can evaluate with reference answer, generate summary resposne from the question, can evaluate with reference answer, separate the question type and call different function to generate response, to prevent the output of llm format is not what we want, try to extract the answ, get_bert_pack(), get_bert_score() (+15 more)

### Community 10 - ".__call__"
Cohesion: 0.10
Nodes (18): _calculate_approx_sum_times(), _calculate_per_summary_chunks(), _get_text(), Document, Path, _summary_          Args:             tot_time (float): _description_, Summarize each chunk and merge them until the combined chunks are smaller than t, refine summary summarizing a chunk at a time and using the previous summary as a (+10 more)

### Community 11 - ".__call__"
Cohesion: 0.12
Nodes (13): ask, Document, Path, add to logs for function if keep_logs is True, add to logs for ask function if keep_logs is True, add to logs for vision function if keep_logs is True, the function to ask model with prompt and info documents,         the info can, ask model with image and prompt, image_path can be list of image path or url (+5 more)

### Community 12 - "anthropic_model"
Cohesion: 0.11
Nodes (12): anthropic_model, Any, run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use, define custom model, input func and temperature          Args:             **, caculate the token count          Args:             prompt (Union[list,str]): (+4 more)

### Community 13 - "gemini_model"
Cohesion: 0.11
Nodes (12): gemini_model, Any, BaseModel, Path, run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use, define custom model, input func and temperature          Args:             ** (+4 more)

### Community 14 - "AzureOpenAIClient"
Cohesion: 0.11
Nodes (13): _async_get_completion(), AzureOpenAIClient, Any, BaseModel, Path, run llm and get the stream generator          Args:             prompt (str):, generate image based on the user prompt.\n         Args:             prompt (s, generate image based on the user prompt.\n         Args:             prompt (s (+5 more)

### Community 15 - "remote_model"
Cohesion: 0.12
Nodes (12): get_stop_list(), handle_url(), run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use, get number of tokens in the text          Args:             **text (str)**: i, get stop list      Args:         stop (Optional[List[str]]): stop list, define custom model, input func and temperature          Args:             **, return llm type          Returns:             str: llm type (+4 more)

### Community 16 - "create_db.py"
Cohesion: 0.21
Nodes (20): create_chromadb_from_file(), create_directory_db(), create_single_file_db(), create_webpage_db(), _is_doc_built(), _is_url_built(), Chroma, Document (+12 more)

### Community 17 - "gptq"
Cohesion: 0.10
Nodes (8): gptq, peft_Llama2, get number of tokens in the text          Args:             **text (str)**: i, define initials and _call function for llama2 peft model      Args:         L, get number of tokens in the text          Args:             **text (str)**: i, define initials and _call function for gptq model      Args:         LLM (_ty, get number of tokens in the text          Args:             **text (str)**: i, TaiwanLLaMaGPTQ

### Community 18 - "ask.py"
Cohesion: 0.25
Nodes (11): _summary_          Args:             tot_time (float): _description_, handle_metrics(), handle_params(), handle_table(), Document, save running parameters into dictionary in order to parse to aiido      Args:, save running metrics into dictionary in order to parse to aiido      Args:, save running results into dictionary in order to parse to aiido      Args: (+3 more)

### Community 19 - "custom_embed"
Cohesion: 0.11
Nodes (12): custom_embed, custom_model, Any, run llm and get the response          Args:             **prompt (str)**: use, get number of tokens in the text          Args:             **text (str)**: i, Initialize the sentence_transformer., Compute doc embeddings using a HuggingFace transformer model.          Args:, Compute query embeddings using a HuggingFace transformer model.          Args: (+4 more)

### Community 20 - "ui.py"
Cohesion: 0.12
Nodes (10): parse all model files(gguf) and directory in the model folder, set the arguments for the model, set_model_dir(), setting_page(), upload documents files to docs folder, upload_page(), implement get response ui, websearch_page() (+2 more)

### Community 21 - "api.py"
Cohesion: 0.24
Nodes (17): ask(), clean(), ConsultModel, ConsultModelReturn, InfoModel, load_env(), RAG(), run RAG for document search, load openai config if needed.      Args: (+9 more)

### Community 22 - "hf_model"
Cohesion: 0.14
Nodes (8): Shared constants for the Akasha package., get_stop_list(), hf_model, run llm and get the response          Args:             **prompt (str)**: use, get stop list      Args:         stop (Optional[List[str]]): stop list, define custom model, input func and temperature          Args:             **, return llm type          Returns:             str: llm type, init vision model          Args:             model_name (str): model name

### Community 23 - "self_ask.py"
Cohesion: 0.16
Nodes (15): get_inter_info(), format the follow up question and answer to a string      Args:         inter, implement the self ask rag function, first get the follow up questions by user p, self_ask_f(), default_self_ask_prompt(), JSON_formatter(), JSON_formatter_dict(), JSON_formatter_list() (+7 more)

### Community 24 - "Installation"
Cohesion: 0.11
Nodes (17): agent, akasha, API Keys, AZURE OPENAI, Change log, Define and Use a Custom Tool, File Summarization, GEMINI (+9 more)

### Community 25 - "Akasha Upgrade Plan: Light Version Support"
Cohesion: 0.12
Nodes (15): 1. Objective, 2. Proposed Syntax, 3. Light vs. Full Comparison, 4. Why Rerank & BERTScore are moved to `[full]`?, 5. Is RAG still functional in `[light]`?, 6. Dependency Reorganization, 7. User Experience: Error & Warning Handling, 8. Implementation Steps (+7 more)

### Community 26 - "delete_documents_by_file"
Cohesion: 0.20
Nodes (14): decide_embedding_type(), get_embedding_type_and_name(), Embeddings, check the embedding type and return the type:name      Args:         embeddin, get the type and name of the embeddings, _delete_docs_built_time(), delete_documents_by_directory(), delete_documents_by_file() (+6 more)

### Community 27 - "file_loader.py"
Cohesion: 0.20
Nodes (11): get_text_from_url(), detect_encoding(), Path, get_load_file_list(), load_directory(), load_url(), Document, Path (+3 more)

### Community 28 - "configure_logging"
Cohesion: 0.29
Nodes (10): _AkashaConsoleFilter, _AkashaOnlyFilter, configure_logging(), _is_akasha_record(), LogRecord, _get_console_handler(), _get_file_handler(), test_keep_logs_bool_creates_default_path() (+2 more)

### Community 29 - "search_docs"
Cohesion: 0.15
Nodes (7): _merge_docs(), BaseLanguageModel, Document, merge different search types documents, if total len of documents too large,, search docs based on given search_type, default is merge, which contain 'mmr', ', search_docs(), keep_rag()

### Community 30 - "LlamaCPP"
Cohesion: 0.18
Nodes (7): get_stop_list(), LlamaCPP, run llm and get the response          Args:             **prompt (str)**: use, get stop list      Args:         stop (Optional[List[str]]): stop list, define custom model, input func and temperature          Args:             **, Cleanup function to be called on exit., return llm type          Returns:             str: llm type

### Community 31 - "search_doc.py"
Cohesion: 0.24
Nodes (9): get_relevant_doc_auto(), get_relevant_doc_auto_rerank(), try every solution to get  to search relevant documents.      Args:         *, try every solution to get  to search relevant documents.      Args:         *, rerank(), rerank_reduce(), BaseRetriever, search docs based on given search_type, default is merge, which contain 'mmr', ' (+1 more)

### Community 32 - ".__call__"
Cohesion: 0.23
Nodes (6): return list of texts that do not exceed the left_token_len      Args:, _retri_max_texts(), _summary_          Args:             tot_time (float): _description_, search the prompt in the web and based on the results to answer the question., websearch class will search the user prompt in the web and based on the results, websearch

### Community 33 - "basic_llm"
Cohesion: 0.17
Nodes (7): basic_llm, change other arguments if user use **kwargs to change them., add pre-process log to self.logs          Args:             timestamp (str):, add post-process log to self.logs          Args:             timestamp (str):, save logs into json or txt file          Args:             file_name (str, op, basic class for akasha, implement _set_model, _change_variables, _check_db, add_, handle_language()

### Community 34 - "gemini_embed"
Cohesion: 0.18
Nodes (8): check_format_prompt(), convert_vision_prompt(), gemini_embed, convert the vision prompt to the correct format, check and format the prompt to fit the correct gemini format, gemini embedding models., Compute doc embeddings using a HuggingFace transformer model.          Args:, Compute query embeddings using a HuggingFace transformer model.          Args:

### Community 35 - "myFAISSRetriever"
Cohesion: 0.26
Nodes (7): myFAISSRetriever, Any, Document, Embeddings, KNNRetriever, implement FAISS search to find relevant documents          Args:, implement FAISS search to find relevant documents          Args:

### Community 36 - "myKNNRetriever"
Cohesion: 0.26
Nodes (7): myKNNRetriever, Any, Document, Embeddings, KNNRetriever, implement k-means search to find relevant documents          Args:, implement k-means search to find relevant documents          Args:

### Community 37 - "myTFIDFRetriever"
Cohesion: 0.35
Nodes (5): myTFIDFRetriever, Any, Document, CallbackManagerForRetrieverRun, TFIDFRetriever

### Community 38 - "handle_embeddings_and_name"
Cohesion: 0.29
Nodes (9): get_mac_address(), get_text_md5(), handle_embeddings_and_name(), get the embeddings object and embed name      Args:         embed (_type_, op, add_chunks(), Embeddings, Path, add chunk(s) to the chromadb collection of the given file_name      Args: (+1 more)

### Community 39 - "MemoryManager"
Cohesion: 0.22
Nodes (7): MemoryManager, Uses an LLM to extract key information from a conversation turn., Uses an LLM to determine a suitable topic for the memory., The main pipeline to process a conversation turn and save it to memory., Manages the creation, storage, and retrieval of long-term semantic memory., default_categorize_memory_prompt(), Returns the system prompt for categorizing a piece of memory.

### Community 40 - "db_structure.py"
Cohesion: 0.31
Nodes (9): get_storage_directory(), is_url(), Path, _sanitize_path_string(), _get_file_dir_docs(), load_docs_from_info(), Document, Path (+1 more)

### Community 41 - "LLM"
Cohesion: 0.18
Nodes (6): chatGLM, define chatglm model and the tokenizer          Args:             **model_nam, return llm type          Returns:             str: llm type, run llm and get the response          Args:             **prompt (str)**: use, get number of tokens in the text          Args:             **text (str)**: i, LLM

### Community 42 - "mySVMRetriever"
Cohesion: 0.29
Nodes (6): mySVMRetriever, Any, Document, Embeddings, implement svm to find relevant documents          Args:             **query (, SVMRetriever

### Community 43 - "BaseRetriever"
Cohesion: 0.38
Nodes (5): myBM25Retriever, Any, Document, implement bm25 to find relevant documents          Args:             **query, BaseRetriever

### Community 44 - "test_api_stability.py"
Cohesion: 0.20
Nodes (6): 測試 Memory 記憶功能 (Light 版通常是 API 呼叫 + 摘要保存), [Light] RAG 煙霧測試：使用遠端 API 進行文件檢索與回答, 測試 Agent 讀取 JSON 並處理資訊的能力, test_agent_json_processing(), test_memory_stability(), test_rag_smoke()

### Community 45 - "get_retrivers"
Cohesion: 0.22
Nodes (7): Searches the vector store for memories relevant to the query., get_retrivers(), BaseRetriever, Embeddings, get the retrivers based on given search_type, default is auto, which contain, 's, add_metadata(), this function is used to add metadata to the old_db object.      Args:

### Community 46 - "aiido_upload"
Cohesion: 0.31
Nodes (7): handle_score_table(), add each response's bert and rouge score into table dictionary      Args:, aiido_upload(), log_params_and_metrics(), mlflow_init(), Equivalent to mlflow.log_params then mlflow.log_metrics      Parameters     -, upload params_metrics, table to mlflow server for tracking.      Args:

### Community 47 - "customRetriever"
Cohesion: 0.39
Nodes (4): customRetriever, Document, Embeddings, implement using custom function to find relevant documents, the custom function

### Community 48 - "myMMRRetriever"
Cohesion: 0.39
Nodes (4): myMMRRetriever, Document, Embeddings, implement using custom function to find relevant documents, the custom function

### Community 49 - "myRerankRetriever"
Cohesion: 0.42
Nodes (4): myRerankRetriever, Any, Document, implement rerank to find relevant documents          Args:             **quer

### Community 50 - "test_openai_rag.py"
Cohesion: 0.25
Nodes (8): api_rag(), Test a simple RAG call using OpenAI.     This should work without torch/transfo, Test that importing akasha and running a basic OpenAI task     does not pull in, Test that trying to use a local model without torch/transformers     raises a c, Fixture for RAG using OpenAI (default).     Ensure OPENAI_API_KEY is set in env, test_graceful_failure_local_model(), test_no_torch_imported(), test_openai_rag_call()

### Community 51 - "test_light_restrictions.py"
Cohesion: 0.29
Nodes (7): importlib_reload(), Test that search_type='rerank' shows a warning when torch is missing., Test that get_bert_score raises ImportError when bert_score is missing., Test that hf_model raises ImportError when torch is missing., test_bert_score_missing(), test_rerank_warning_when_torch_missing(), test_torch_missing_hf_model()

### Community 52 - "__init__.py"
Cohesion: 0.38
Nodes (5): edit_image(), gen_image(), Path, the image generate image call Model to generate image based on the user prompt.\, the image generate image call Model to generate image based on the user prompt.\

### Community 54 - "load_docs_from_webengine"
Cohesion: 0.40
Nodes (5): _get_search_api_key(), load_docs_from_webengine(), Document, get the search results based on the prompt and search engine     search_engine, get the search api key based on the search engine

### Community 55 - "test_agents_observation_normalization.py"
Cohesion: 0.60
Nodes (3): _DummyModel, _fake_basic_llm_init(), test_run_agent_retri_observation_accepts_non_string_outputs()

### Community 56 - "anthropic_rag"
Cohesion: 0.50
Nodes (4): anthropic_rag(), Test a simple RAG call using Anthropic., Fixture for RAG using Anthropic.     Ensure ANTHROPIC_API_KEY is set in environ, test_anthropic_rag_call()

### Community 57 - "gemini_rag"
Cohesion: 0.50
Nodes (4): gemini_rag(), Test a simple RAG call using Gemini., Fixture for RAG using Gemini.     Ensure GEMINI_API_KEY is set in environment., test_gemini_rag_call()

### Community 58 - "_generate_single_choice_question"
Cohesion: 0.50
Nodes (4): _generate_single_choice_question(), Based on gernerated question and answer, generate wrong answers for single choic, format_wrong_answer(), prompt for generate wrong answers to create single choice question      Args:

## Knowledge Gaps
- **57 isolated node(s):** `akasha_terminal`, `Quick Start (Local Development)`, `Change log`, `Standard Installation`, `Lightweight Installation (API-call-only, v1.0+)` (+52 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `dbs` connect `dbs` to `evaluation.py`, `basic_llm`, `gen_prompt.py`, `myFAISSRetriever`, `myKNNRetriever`, `self_query_filter.py`, `MemoryManager`, `handle_objects.py`, `atman`, `db_structure.py`, `handle_embeddings_and_name`, `mySVMRetriever`, `get_retrivers`, `customRetriever`, `myMMRRetriever`, `ask.py`, `test_light_restrictions.py`, `base.py`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Why does `handle_model()` connect `handle_objects.py` to `__init__.py`, `LLM`, `anthropic_model`, `gemini_model`, `AzureOpenAIClient`, `remote_model`, `gptq`, `ask.py`, `custom_embed`, `hf_model`, `LlamaCPP`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `AzureOpenAIClient` connect `AzureOpenAIClient` to `handle_objects.py`, `LLM`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `dbs` (e.g. with `eval` and `MemoryManager`) actually correct?**
  _`dbs` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `class for implement search db based on user prompt and generate response from ll`, `initials of Doc_QA class          Args:             embeddings (_type_, optio`, `input the documents directory path and question, will first store the documents` to the rest of the system?**
  _344 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `evaluation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.052083333333333336 - nodes in this community are weakly interconnected._
- **Should `agents` be split into smaller, more focused modules?**
  _Cohesion score 0.05519480519480519 - nodes in this community are weakly interconnected._