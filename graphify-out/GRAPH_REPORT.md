# Graph Report - akasha-repo  (2026-07-30)

## Corpus Check
- 193 files · ~157,943 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1827 nodes · 3255 edges · 152 communities (143 shown, 9 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 113 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b6e18213`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evaluation.py
- agents
- tests.md
- gen_prompt.py
- __init__.py
- dbs
- self_query_filter.py
- atman
- handle_objects.py
- model_eval.py
- .__call__
- .__call__
- anthropic_model
- gemini_model
- AzureOpenAIClient
- remote_model
- create_db.py
- gptq
- ask.py
- custom_embed
- ui.py
- api.py
- hf_model
- self_ask.py
- Installation
- Akasha Upgrade Plan: Light Version Support
- delete_documents_by_file
- file_loader.py
- configure_logging
- search_docs
- LlamaCPP
- search_doc.py
- .__call__
- basic_llm
- gemini_embed
- myFAISSRetriever
- myKNNRetriever
- myTFIDFRetriever
- handle_embeddings_and_name
- MemoryManager
- db_structure.py
- LLM
- mySVMRetriever
- BaseRetriever
- test_api_stability.py
- get_retrivers
- aiido_upload
- customRetriever
- myMMRRetriever
- myRerankRetriever
- test_openai_rag.py
- test_light_restrictions.py
- __init__.py
- base.py
- load_docs_from_webengine
- test_agents_observation_normalization.py
- anthropic_rag
- gemini_rag
- _generate_single_choice_question
- test_agent.py
- test_live_model_final_action_aliases
- pytest_configure
- hello memory.md
- test_summary.py
- 個人背景.md
- 個人資料.md
- 個人資訊.md
- 地點資訊.md
- 居住地.md
- 居住地資訊.md
- 食物偏好.md
- 食物喜好.md
- ex_ask.py
- ex_eval.py
- ex_generate_img.py
- ex_long_term_memory.py
- ex_rag.py
- ex_selfask_rag.py
- akasha_terminal
- Path
- Document
- Path
- Chroma
- Embeddings
- Path
- Embeddings
- Path
- Embeddings
- Path
- Embeddings
- Path
- BaseModel
- MemoryManager
- akasha.agents 動態 Skills 使用說明
- Agent Skills 本機 Runtime 與 Script 執行修改計畫
- Thinking / Tool / Skill / Answer 回傳格式
- SkillContext
- BaseRetriever
- call_stream_events
- _Response
- ._call
- Q: 我執行 repl_app.py 然後再看輸出時，看不出是模型自己想的，還是有根據 skill 做的? 能在加上一些過程的訊息嗎?
- Q: verbose=True 應由 akasha 內部印出載入 skill 與使用工具的過程，而不是要求範例使用者解析 stream event。
- test_ask_thinking_warning.py
- conftest.py
- ResolvedSkillTools
- SKILL.md
- SKILL.md
- REFERENCE.md

## God Nodes (most connected - your core abstractions)
1. `dbs` - 69 edges
2. `DynamicSkillMiddleware` - 48 edges
3. `Skill` - 38 edges
4. `call_model()` - 34 edges
5. `format_sys_prompt()` - 32 edges
6. `get_storage_directory()` - 27 edges
7. `SkillContext` - 26 edges
8. `get_retrivers()` - 26 edges
9. `agents` - 24 edges
10. `get_doc_length()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `test_rag_check_db_rejects_missing_database()` --indirect_call--> `RAG`  [INFERRED]
  tests/unit/test_rag_input_contract.py → akasha/RAG/rag.py
- `test_rag_check_doc_path_preserves_path_objects()` --indirect_call--> `RAG`  [INFERRED]
  tests/unit/test_rag_input_contract.py → akasha/RAG/rag.py
- `test_load_skill_directory_reads_only_skill_markdown()` --calls--> `load_skill_directory()`  [EXTRACTED]
  tests/unit/test_agent_skills.py → akasha/agent/skills/loader.py
- `test_separate_docs_drops_empty_chunks()` --indirect_call--> `ask`  [INFERRED]
  tests/unit/test_ask_info.py → akasha/tools/ask.py
- `_ask_for_display()` --indirect_call--> `ask`  [INFERRED]
  tests/unit/test_ask_thinking_warning.py → akasha/tools/ask.py

## Import Cycles
- None detected.

## Communities (152 total, 9 thin omitted)

### Community 0 - "evaluation.py"
Cohesion: 0.08
Nodes (34): merge_history_and_prompt(), merge system prompt, history messages, and prompt based on the prompt format typ, from messages dict list, get pairs of user question and assistant response from, retri_history_messages(), reference docs after calling the rag function, will return the reference file na, decide_auto_prompt_format_type(), default_get_reference_prompt(), fact_question_prompt() (+26 more)

### Community 1 - "agents"
Cohesion: 0.06
Nodes (37): calculate_tool(), _jsonSaveTool(), BaseTool, rag_tool(), return the tool to use search engine to search information for user      Args:, return the json save tool that can save the content into json file.      Retur, save content into json file, saveJSON_tool() (+29 more)

### Community 2 - "tests.md"
Cohesion: 0.05
Nodes (40): 1.1 RAG 核心, 1.1 遠端模型與基礎問答, 1.2 摘要功能 (Summary), 1.2 核心邏輯與數據處理, 1.3 代理人功能 (Agents), 1.3 輕量檢索器, 1.4 評估功能 (Eval), 1.5 影像處理與多模態 (Vision & Multimodal) (+32 more)

### Community 3 - "gen_prompt.py"
Cohesion: 0.09
Nodes (38): check_essay_system_prompt(), check_sum_type(), find_same_category(), get_non_repeat_rand_int(), get_question_from_file(), get_source_files(), load questions from file and save the questions into lists.     a question list, iterate the category dictionary and check if any category has more than cate_thr (+30 more)

### Community 4 - "__init__.py"
Cohesion: 0.22
Nodes (12): get_storage_directory(), is_url(), _sanitize_path_part(), _sanitize_path_string(), PurePath, _FakeArray, _FakeChroma, test_dbs_initializes_from_chroma_like_object_with_fallbacks() (+4 more)

### Community 5 - "dbs"
Cohesion: 0.19
Nodes (7): gemini_model, BaseModel, LLM, Path, run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use

### Community 6 - "self_query_filter.py"
Cohesion: 0.10
Nodes (27): check_metadata_info(), DocumentCP, filter_docs(), find_subset(), generate_query_constructor(), generate_query_filter(), handle_attr(), Any (+19 more)

### Community 7 - "atman"
Cohesion: 0.06
Nodes (26): BaseLanguageModel, Embeddings, Path, RAG, input the documents directory path and question, will first store the documents, input the documents directory path and question, will first store the documents, class for implement search db based on user prompt and generate response from ll, initials of Doc_QA class          Args:             embeddings (_type_, optio (+18 more)

### Community 8 - "handle_objects.py"
Cohesion: 0.05
Nodes (37): 10. Logs 與可觀測性, 11. 測試規劃, 12. 實作階段, 13. 不在 Phase 2 處理, 14.1 決策後的模型, 14. 待確認決策, 15.1 公開使用方式, 15.2 內部責任邊界 (+29 more)

### Community 9 - "model_eval.py"
Cohesion: 0.10
Nodes (23): get_torch(), generate fact resposne from the question, can evaluate with reference answer, generate summary resposne from the question, can evaluate with reference answer, separate the question type and call different function to generate response, get_bert_pack(), get_bert_score(), get_llm_score(), get_rouge_score() (+15 more)

### Community 10 - ".__call__"
Cohesion: 0.10
Nodes (13): anthropic_model, Any, LLM, run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use, define custom model, input func and temperature          Args:             ** (+5 more)

### Community 11 - ".__call__"
Cohesion: 0.11
Nodes (13): get_stop_list(), handle_url(), LLM, run llm and get the response          Args:             **prompt (str)**: use, run llm and get the response          Args:             **prompt (str)**: use, get number of tokens in the text          Args:             **text (str)**: i, get stop list      Args:         stop (Optional[List[str]]): stop list, define custom model, input func and temperature          Args:             ** (+5 more)

### Community 12 - "anthropic_model"
Cohesion: 0.16
Nodes (18): get_text_from_url(), detect_encoding(), Path, get_load_file_list(), load_directory(), load_file(), load_url(), Document (+10 more)

### Community 13 - "gemini_model"
Cohesion: 0.11
Nodes (11): dbs, extract_db_by_ids(), extract_db_by_keyword(), pop_db_by_ids(), pop undesired data from  dbs based on ids      Args:         db (dbs): dbs ob, extract db from dbs based on keyword_list      Args:         db (dbs): dbs ob, extract db from dbs based on ids      Args:         db (dbs): dbs object, add_metadata() (+3 more)

### Community 14 - "AzureOpenAIClient"
Cohesion: 0.10
Nodes (14): _async_get_completion(), AzureOpenAIClient, Any, BaseModel, LLM, Path, run llm and get the stream generator          Args:             prompt (str):, generate image based on the user prompt.\n         Args:             prompt (s (+6 more)

### Community 15 - "remote_model"
Cohesion: 0.09
Nodes (29): agents, _as_message_list(), _count_tokens(), _extract_messages(), _json_safe(), _last_answer(), _message_dump(), _message_text() (+21 more)

### Community 16 - "create_db.py"
Cohesion: 0.24
Nodes (17): ask(), clean(), ConsultModel, ConsultModelReturn, InfoModel, load_env(), BaseModel, RAG() (+9 more)

### Community 17 - "gptq"
Cohesion: 0.10
Nodes (14): custom_embed, custom_model, Any, BaseModel, Embeddings, LLM, run llm and get the response          Args:             **prompt (str)**: use, get number of tokens in the text          Args:             **text (str)**: i (+6 more)

### Community 18 - "ask.py"
Cohesion: 0.10
Nodes (12): handle_model(), ThinkingBudget, create model client used in document QA, default if openai "gpt-3.5-turbo", gptq, peft_Llama2, LLM, get number of tokens in the text          Args:             **text (str)**: i, define initials and _call function for llama2 peft model      Args:         L (+4 more)

### Community 19 - "custom_embed"
Cohesion: 0.07
Nodes (28): Akasha Package Upgrade Log, Chroma compatibility helper, Code Changes, Dependency Changes, Fix, Follow-up Suggestions, Import chain hardening, Known Constraints (+20 more)

### Community 20 - "ui.py"
Cohesion: 0.20
Nodes (5): Compute the number of tokens in a given text using either huggingface or OpenAI, Load a tokenizer from local path or huggingface model hub.          Args:, Save the tokenizer to local path.          Args:             name (str): The, Compute the number of tokens in a given text using huggingface tokenizer., Compute the number of tokens in a given text using OpenAI tiktoken.          A

### Community 21 - "api.py"
Cohesion: 0.11
Nodes (15): ask, Document, Path, ThinkingBudget, add to logs for function if keep_logs is True, add to logs for ask function if keep_logs is True, add to logs for vision function if keep_logs is True, the function to ask model with prompt and info documents,         the info can (+7 more)

### Community 22 - "hf_model"
Cohesion: 0.07
Nodes (27): 1. 執行 unit tests, 2. 只執行 thinking adapter 測試, 3. 執行 API contract 與 response normalization 測試, 4. 執行 logging 回歸測試, 5. 執行 embedding provider smoke, 6. 執行 OpenAI 分階段 RAG pipeline, 7. 執行 Gemini 完整 RAG pipeline, 8. 執行 MCP smoke tests (+19 more)

### Community 23 - "self_ask.py"
Cohesion: 0.24
Nodes (6): eval, BaseLanguageModel, Embeddings, initials of Model_Eval class          Args:             **embeddings (str, op, base_line(), test_Model_Eval()

### Community 24 - "Installation"
Cohesion: 0.13
Nodes (9): Shared constants for the Akasha package., get_stop_list(), hf_model, LLM, run llm and get the response          Args:             **prompt (str)**: use, get stop list      Args:         stop (Optional[List[str]]): stop list, define custom model, input func and temperature          Args:             **, return llm type          Returns:             str: llm type (+1 more)

### Community 25 - "Akasha Upgrade Plan: Light Version Support"
Cohesion: 0.19
Nodes (16): get_chroma_components(), Compatibility helpers for optional Chroma dependencies., extract_db_by_file(), extract db from dbs based on file_name_list      Args:         db (dbs): dbs, _display_db_num(), load_db_by_chroma_name(), load_directory_db(), load_files_db() (+8 more)

### Community 26 - "delete_documents_by_file"
Cohesion: 0.12
Nodes (10): parse all model files(gguf) and directory in the model folder, set the arguments for the model, set_model_dir(), setting_page(), upload documents files to docs folder, upload_page(), implement get response ui, websearch_page() (+2 more)

### Community 27 - "file_loader.py"
Cohesion: 0.13
Nodes (19): call_model(), call llm model and return the response      Args:         model (BaseLanguage, get_inter_info(), format the follow up question and answer to a string      Args:         inter, implement the self ask rag function, first get the follow up questions by user p, self_ask_f(), default_self_ask_prompt(), JSON_formatter() (+11 more)

### Community 28 - "configure_logging"
Cohesion: 0.14
Nodes (16): _AkashaConsoleFilter, _AkashaOnlyFilter, configure_logging(), _is_akasha_record(), Prevent fallback/root handlers from duplicating agent trace records., _SuppressAkashaTraceFilter, LogRecord, _get_console_handler() (+8 more)

### Community 29 - "search_docs"
Cohesion: 0.18
Nodes (21): create_chromadb_from_file(), create_directory_db(), create_single_file_db(), create_webpage_db(), _get_recursive_character_text_splitter(), _is_doc_built(), _is_url_built(), Any (+13 more)

### Community 30 - "LlamaCPP"
Cohesion: 0.29
Nodes (5): Stage-by-stage live RAG checks for the Chroma-to-answer pipeline., Release RAG-owned references and run collection before pytest exits., A built Chroma store can be opened again and read independently., test_rag_stage_1_reload_chroma(), test_rag_stage_4_cleanup_is_executable()

### Community 31 - "search_doc.py"
Cohesion: 0.13
Nodes (27): BaseTool, Create and validate skill tools without replacing existing tools., resolve_skill_tools(), BaseTool, Safe, explicit configuration exposed to a skill tool factory., An explicit allowlist of factories available to skill definitions., SkillToolContext, ToolRegistry (+19 more)

### Community 32 - ".__call__"
Cohesion: 0.10
Nodes (20): 1. `akasha.ask()`, 2. `akasha.agents()`, 3. `akasha.RAG()`, 4. `akasha.summary()`, 5. `MemoryManager`, 6. `akasha.agent.*` / 內部但對外可感知的行為, 7. 共通高風險參數, 8. 測試優先順序 (+12 more)

### Community 33 - "basic_llm"
Cohesion: 0.18
Nodes (6): basic_llm, change other arguments if user use **kwargs to change them., add pre-process log to self.logs          Args:             timestamp (str):, add post-process log to self.logs          Args:             timestamp (str):, save logs into json or txt file          Args:             file_name (str, op, basic class for akasha, implement _set_model, _change_variables, _check_db, add_

### Community 34 - "gemini_embed"
Cohesion: 0.10
Nodes (20): agent, akasha, ANTHROPIC, API Keys, AZURE OPENAI, Change log, Define and Use a Custom Tool, Editable Install Commands (+12 more)

### Community 35 - "myFAISSRetriever"
Cohesion: 0.09
Nodes (14): _generate_single_choice_question(), Model_Eval, BaseLanguageModel, Embeddings, class for implement evaluation of llm model, include auto_create_questionset and, add post-process log to self.logs          Args:             timestamp (str):, save questions and ref answers into txt file, and save the path of question set, parse the question and answer from the llm response, and save it into question a (+6 more)

### Community 36 - "myKNNRetriever"
Cohesion: 0.20
Nodes (20): _assert_json_safe(), _load_test_env(), Real provider contract smoke tests.  These tests intentionally cross the provi, A real provider stream must stay an iterable of text chunks., The LangChain agent adapter must also receive a final AI message., Agent streaming must normalize real provider chunks into events., Gemini's native thinking path must preserve answer/thinking separation., Gemini agents must expose native thinking events without corrupting answer. (+12 more)

### Community 37 - "myTFIDFRetriever"
Cohesion: 0.35
Nodes (5): myTFIDFRetriever, Any, Document, CallbackManagerForRetrieverRun, TFIDFRetriever

### Community 38 - "handle_embeddings_and_name"
Cohesion: 0.12
Nodes (15): 1. Objective, 2. Proposed Syntax, 3. Light vs. Full Comparison, 4. Why Rerank & BERTScore are moved to `[full]`?, 5. Is RAG still functional in `[light]`?, 6. Dependency Reorganization, 7. User Experience: Error & Warning Handling, 8. Implementation Steps (+7 more)

### Community 39 - "MemoryManager"
Cohesion: 0.20
Nodes (17): convert simplified chinese to traditional chinese      Args:         **text (, sim_to_trad(), handle_model_and_name(), get the model object and model name      Args:         model (_type_, optiona, call_batch_model(), call_image_model(), call_stream_model(), call_translator() (+9 more)

### Community 40 - "db_structure.py"
Cohesion: 0.11
Nodes (18): public API 測試矩陣對照表, `tests/test_agent.py`, `tests/test_akasha.py`, `tests/test_api_stability.py`, `tests/test_live_gemini_agent.py`, `tests/test_summary.py`, `tests/unit/test_agents_core.py`, `tests/unit/test_agents_observation_normalization.py` (+10 more)

### Community 41 - "LLM"
Cohesion: 0.16
Nodes (8): get_stop_list(), LlamaCPP, LLM, run llm and get the response          Args:             **prompt (str)**: use, get stop list      Args:         stop (Optional[List[str]]): stop list, define custom model, input func and temperature          Args:             **, Cleanup function to be called on exit., return llm type          Returns:             str: llm type

### Community 42 - "mySVMRetriever"
Cohesion: 0.06
Nodes (29): Searches the vector store for memories relevant to the query., get_relevant_doc_auto_rerank(), try every solution to get  to search relevant documents.      Args:         *, get_retrivers(), BaseRetriever, Embeddings, get the retrivers based on given search_type, default is auto, which contain, 's, myRerankRetriever (+21 more)

### Community 43 - "BaseRetriever"
Cohesion: 0.13
Nodes (14): ThinkingBudget, change model, temperature if user use **kwargs to change them., _summary_          Args:             model (str, optional): _description_. De, _normalize_openai_base_url(), LangChain ChatModel factory used by the public Akasha model selectors., Convert a full OpenAI endpoint URL into an SDK base URL., normalize_thinking_budget(), normalize_thinking_level() (+6 more)

### Community 44 - "test_api_stability.py"
Cohesion: 0.17
Nodes (17): build_chat_model(), Any, ThinkingBudget, Build a LangChain 1.3+ chat model for a supported provider., test_anthropic_does_not_send_temperature(), test_anthropic_thinking_reserves_tokens_for_reasoning(), test_azure_environment_requires_key_and_base_url(), test_azure_openai_compatible_endpoint_uses_dedicated_environment() (+9 more)

### Community 45 - "get_retrivers"
Cohesion: 0.23
Nodes (8): myFAISSRetriever, Any, BaseRetriever, Document, Embeddings, KNNRetriever, implement FAISS search to find relevant documents          Args:, implement FAISS search to find relevant documents          Args:

### Community 46 - "aiido_upload"
Cohesion: 0.23
Nodes (8): myKNNRetriever, Any, BaseRetriever, Document, Embeddings, KNNRetriever, implement k-means search to find relevant documents          Args:, implement k-means search to find relevant documents          Args:

### Community 47 - "customRetriever"
Cohesion: 0.29
Nodes (13): _assert_json_safe(), _configured(), _load_env(), _rag(), Opt-in RAG smoke tests covering the public file/path contract., OpenAI RAG must ingest one file and return a serializable contract., Gemini RAG must ingest a directory and retrieve the relevant file., A Windows runner must accept an absolute Path object as file input. (+5 more)

### Community 48 - "myMMRRetriever"
Cohesion: 0.14
Nodes (10): 測試 Memory 記憶功能 (light 版使用遠端 embedding + Chroma), 驗證 Windows absolute path 可搭配遠端 embedding 與 Chroma 正常運作。, 驗證 MemoryManager 可使用 Windows absolute path 作為持久化目錄。, [Light] RAG 煙霧測試：使用遠端 API 進行文件檢索與回答, 測試原生 tool calling Agent 讀取 JSON 並處理資訊的能力, test_agent_native_tool_calling(), test_memory_stability(), test_rag_smoke() (+2 more)

### Community 49 - "myRerankRetriever"
Cohesion: 0.15
Nodes (13): 2026-07 測試實際準備清單, B. 第一批要測的模型清單, C. Embedding model 清單, D. Python 與套件環境, E. 測試資料夾與檔案, F. Vision 實際需要的東西, G. Tool-calling 實際需要的東西, H. MCP 實際需要的東西 (+5 more)

### Community 50 - "test_openai_rag.py"
Cohesion: 0.29
Nodes (6): Compute the number of tokens in a given text using Google Vertex AI., calculate_token(), check_format_prompt(), convert_vision_prompt(), convert the vision prompt to the correct format, check and format the prompt to fit the correct gemini format

### Community 51 - "test_light_restrictions.py"
Cohesion: 0.26
Nodes (11): get_embedding_type_and_name(), Embeddings, get the type and name of the embeddings, _delete_docs_built_time(), delete_documents_by_directory(), delete_documents_by_file(), Embeddings, Path (+3 more)

### Community 52 - "__init__.py"
Cohesion: 0.29
Nodes (5): gemini_embed, Embeddings, gemini embedding models., Compute doc embeddings using a HuggingFace transformer model.          Args:, Compute query embeddings using a HuggingFace transformer model.          Args:

### Community 53 - "base.py"
Cohesion: 0.20
Nodes (9): Akasha Testing Guide, Coverage Policy, Marker 約定, 升級前後建議流程, 建議指令, 注意事項, 測試分層, 目標 (+1 more)

### Community 54 - "load_docs_from_webengine"
Cohesion: 0.14
Nodes (7): _AvailableSkill, DynamicSkillMiddleware, Any, Path, Expose load_skill first, then route loaded skill tools and resources., ModelRequest, ToolCallRequest

### Community 55 - "test_agents_observation_normalization.py"
Cohesion: 0.09
Nodes (26): myTokenizer, Initialize a Tokenizer object.          Args:             model_id (str): The, this class is for computing the number of tokens in a given text using different, _summary_          Args:             tot_time (float): _description_, return list of texts that do not exceed the left_token_len      Args:, _retri_max_texts(), _summary_          Args:             tot_time (float): _description_, _summary_          Args:             tot_time (float): _description_ (+18 more)

### Community 56 - "anthropic_rag"
Cohesion: 0.27
Nodes (11): MultiServerMCPClient, _client(), _discover_tools(), _load_test_env(), MCP discovery -> tool invocation -> real agent event/log contracts., Every manifest model must select MCP and record its tool call., MCP agents use ainvoke and reject the sync stream facade explicitly., The local stdio server exposes stable tools and callable schemas. (+3 more)

### Community 57 - "gemini_rag"
Cohesion: 0.15
Nodes (20): Public skill APIs for Akasha agents., _build_skill(), load_skill_directory(), Any, Path, Load Agent Skills from SKILL.md files with YAML frontmatter., Load a skill directory containing standard SKILL.md frontmatter., _read_skill_document() (+12 more)

### Community 58 - "_generate_single_choice_question"
Cohesion: 0.18
Nodes (6): chatGLM, LLM, define chatglm model and the tokenizer          Args:             **model_nam, return llm type          Returns:             str: llm type, run llm and get the response          Args:             **prompt (str)**: use, get number of tokens in the text          Args:             **text (str)**: i

### Community 59 - "test_agent.py"
Cohesion: 0.22
Nodes (5): Any, run llm and get the response          Args:             **prompt (str)**: use, define custom model, input func and temperature          Args:             **, Initialize the sentence_transformer., run llm and get the stream generator          Args:             prompt (str):

### Community 60 - "test_live_model_final_action_aliases"
Cohesion: 0.10
Nodes (18): _calculate_approx_sum_times(), _calculate_per_summary_chunks(), _get_recursive_character_text_splitter(), _get_text(), Document, Path, Summarize each chunk and merge them until the combined chunks are smaller than t, refine summary summarizing a chunk at a time and using the previous summary as a (+10 more)

### Community 61 - "pytest_configure"
Cohesion: 0.33
Nodes (5): myBM25Retriever, Any, BaseRetriever, Document, implement bm25 to find relevant documents          Args:             **query

### Community 62 - "hello memory.md"
Cohesion: 0.44
Nodes (8): executable_lines(), load_module(), load_test_modules(), main(), normalize(), Path, run_selected_tests(), tracer()

### Community 63 - "test_summary.py"
Cohesion: 0.33
Nodes (5): customRetriever, BaseRetriever, Document, Embeddings, implement using custom function to find relevant documents, the custom function

### Community 65 - "個人背景.md"
Cohesion: 0.25
Nodes (8): api_rag(), Test a simple RAG call using OpenAI.     This should work in light mode with re, Test that importing akasha and running a basic OpenAI task     does not pull in, Test that trying to use a local model without torch/transformers     raises a c, Fixture for RAG using OpenAI (default).     Ensure OPENAI_API_KEY is set in env, test_graceful_failure_local_model(), test_no_torch_imported(), test_openai_rag_call()

### Community 66 - "個人資料.md"
Cohesion: 0.33
Nodes (5): myMMRRetriever, BaseRetriever, Document, Embeddings, implement using custom function to find relevant documents, the custom function

### Community 67 - "個人資訊.md"
Cohesion: 0.14
Nodes (19): decide_embedding_type(), separate type:name by ':'      Args:         **name (str)**: string with form, check the embedding type and return the type:name      Args:         embeddin, separate_name(), handle_embeddings_and_name(), handle_model_type(), BaseLanguageModel, Embeddings (+11 more)

### Community 68 - "地點資訊.md"
Cohesion: 0.16
Nodes (12): load_skill_metadata(), Load only standard SKILL.md metadata; defer instructions., A named set of instructions and optionally registered tools.      Phase 1 uses `, Skill, Path, An allowlisted, in-process skill registry.      A registry is intentionally expl, SkillRegistry, test_load_skill_directory_reads_only_skill_markdown() (+4 more)

### Community 69 - "居住地.md"
Cohesion: 0.25
Nodes (7): mcp_add(), mcp_get_weather(), mcp_lookup_version(), Deterministic stdio MCP server used by the MCP smoke tests., Add two integers and return the sum., Return deterministic weather for a test city., Return a deterministic package version for contract testing.

### Community 70 - "居住地資訊.md"
Cohesion: 0.48
Nodes (5): _FakeDB, test_get_retrievers_builds_expected_retriever_types(), test_get_retrievers_raises_on_unknown_search_type(), test_get_retrievers_supports_custom_callable(), test_get_retrievers_warns_when_rerank_support_is_missing()

### Community 71 - "食物偏好.md"
Cohesion: 0.20
Nodes (10): extract_multiple_json(), Any, Extract multiple JSON objects from text generated by LLMs.      Args:, basemodel_keys_list(), call_JSON_formatter(), keys_to_basemodel_class(), BaseModel, use LLM to transfer texts into JSON format      Args:         model_obj (Base (+2 more)

### Community 72 - "食物喜好.md"
Cohesion: 0.53
Nodes (5): _env_file(), Opt-in live test for asking about Akasha from web-provided context.  The test, Gemini should identify Akasha as a flexible LLM QA/RAG tool., _require_gemini_key(), test_gemini_ask_answers_from_web_info()

### Community 74 - "ex_ask.py"
Cohesion: 0.11
Nodes (18): 10. 測試計畫, 11. 實作階段, 12. 暫不處理, 13. 待確認決策, 17. Resource Loading, 1. 目標, 2. 第一版範圍, 3. 公開 API (+10 more)

### Community 75 - "ex_eval.py"
Cohesion: 0.50
Nodes (3): Live contract test for ``ask(..., info=[url, url])``., The public callable API accepts two URL references in ``info``., test_gemini_ask_with_two_url_info_items()

### Community 76 - "ex_generate_img.py"
Cohesion: 0.25
Nodes (7): Design decisions, Implementation, LangChain 1.3+ 原生 Agent 升級說明, Streaming contract, Summary, Testing and acceptance, Thinking provider mapping

### Community 77 - "ex_long_term_memory.py"
Cohesion: 0.36
Nodes (4): _FakeRetriever, test_retri_docs_uses_auto_and_deduplicates(), test_search_docs_merges_multiple_retrievers(), test_search_docs_uses_auto_helpers()

### Community 90 - "Path"
Cohesion: 0.29
Nodes (5): akasha-repo 測試補強計畫, 可直接照做的檢查清單, 目的, 真實 smoke 的建議規模, 驗收標準

### Community 92 - "Document"
Cohesion: 0.29
Nodes (7): agents, ask, MemoryManager, RAG, summary, 依功能切分, 建議的測試矩陣

### Community 93 - "Path"
Cohesion: 0.29
Nodes (7): Capability manifest 不等於 skip 清單, Embedding model 是獨立的測試對象, Live smoke 成本與診斷, MCP 是 agent tool integration 的獨立路徑, RAG 準備與驗收, 本次決議與後續建置規則, 統一 stream event contract

### Community 95 - "Chroma"
Cohesion: 0.62
Nodes (6): _env_file(), Opt-in live tests for the LangChain-native Gemini Agent path.  These tests int, _require_key(), test_live_gemini_agent_returns_final_answer(), test_live_gemini_agent_streams_thinking_and_answer_events(), test_live_gemini_ignores_budget_when_thinking_disabled()

### Community 97 - "Embeddings"
Cohesion: 0.26
Nodes (7): mySVMRetriever, Any, BaseRetriever, Document, Embeddings, implement svm to find relevant documents          Args:             **query (, SVMRetriever

### Community 98 - "Path"
Cohesion: 0.53
Nodes (5): main(), _parse_array(), Path, _split_dependencies(), _write_requirements()

### Community 101 - "Embeddings"
Cohesion: 0.50
Nodes (4): anthropic_rag(), Test a simple RAG call using Anthropic., Fixture for RAG using Anthropic.     Ensure ANTHROPIC_API_KEY is set in environ, test_anthropic_rag_call()

### Community 102 - "Path"
Cohesion: 0.50
Nodes (4): gemini_rag(), Test a simple RAG call using Gemini., Fixture for RAG using Gemini.     Ensure GEMINI_API_KEY is set in environment., test_gemini_rag_call()

### Community 105 - "Embeddings"
Cohesion: 0.50
Nodes (4): 1. mock 只留給純邏輯, 2. public API 要有真實 smoke, 3. 測「關鍵組合」，不要全排列, 核心原則

### Community 106 - "Path"
Cohesion: 0.50
Nodes (4): 1. 先保留 unit，補 smoke, 2. 再收斂 mock, 3. 統一測試入口, 建議實作方式

### Community 109 - "Embeddings"
Cohesion: 0.50
Nodes (4): A. Unit tests, B. Contract / smoke tests, C. Upgrade tests, 測試分層建議

### Community 110 - "Path"
Cohesion: 0.50
Nodes (4): P0, P1, P2, 建議的落地順序

### Community 113 - "BaseModel"
Cohesion: 0.67
Nodes (3): A. API keys 與 provider 帳號, CI secrets, 本機 .env

### Community 115 - "MemoryManager"
Cohesion: 0.14
Nodes (9): get_mac_address(), get_text_md5(), MemoryManager, Uses an LLM to extract key information from a conversation turn., Uses an LLM to determine a suitable topic for the memory., The main pipeline to process a conversation turn and save it to memory., Manages the creation, storage, and retrieval of long-term semantic memory., default_categorize_memory_prompt() (+1 more)

### Community 116 - "akasha.agents 動態 Skills 使用說明"
Cohesion: 0.12
Nodes (16): akasha.agents 動態 Skills 使用說明, Logs, Phase 2b：動態載入流程, Skill Directory 結構, SKILL.md 格式, Skill Resources, Skill Script 執行, Skill Tool Factory (+8 more)

### Community 117 - "Agent Skills 本機 Runtime 與 Script 執行修改計畫"
Cohesion: 0.14
Nodes (13): Agent Skills 本機 Runtime 與 Script 執行修改計畫, Agent 可見工具, Phase A：內部執行介面, Phase B：執行限制與錯誤處理, Phase C：Skill instructions 與 resource 整合, Phase D：測試與相容性, 不在本次範圍, 主程式執行環境 (+5 more)

### Community 118 - "Thinking / Tool / Skill / Answer 回傳格式"
Cohesion: 0.14
Nodes (13): `agents()` 完整範例, Answer, `ask()` 範例, Skill 如何辨識, Streaming event 共通格式, Thinking, Thinking / Tool / Skill / Answer 回傳格式, Tool / Skill result (+5 more)

### Community 119 - "SkillContext"
Cohesion: 0.17
Nodes (6): AgentState, BaseModel, _PythonExecuteInput, SkillAgentState, Resolved, immutable skill information used by agent middleware., SkillContext

### Community 121 - "call_stream_events"
Cohesion: 0.26
Nodes (11): call_stream_events(), content_to_text(), content_to_thinking(), normalize_chat_input(), Extract visible text from LangChain content blocks.      Reasoning/thinking bloc, Extract reasoning/thinking text from LangChain content blocks., Yield normalized answer/thinking events from a ChatModel stream., Convert legacy prompt dictionaries to LangChain chat messages. (+3 more)

### Community 122 - "_Response"
Cohesion: 0.43
Nodes (4): _Response, test_get_text_from_url_extracts_title_and_visible_text(), test_get_text_from_url_handles_request_exceptions(), test_get_webpage_last_modified_handles_present_and_missing_headers()

### Community 124 - "Q: 我執行 repl_app.py 然後再看輸出時，看不出是模型自己想的，還是有根據 skill 做的? 能在加上一些過程的訊息嗎?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: 我執行 repl_app.py 然後再看輸出時，看不出是模型自己想的，還是有根據 skill 做的? 能在加上一些過程的訊息嗎?, Source Nodes

### Community 125 - "Q: verbose=True 應由 akasha 內部印出載入 skill 與使用工具的過程，而不是要求範例使用者解析 stream event。"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: verbose=True 應由 akasha 內部印出載入 skill 與使用工具的過程，而不是要求範例使用者解析 stream event。, Source Nodes

### Community 126 - "test_ask_thinking_warning.py"
Cohesion: 0.70
Nodes (4): _ask_for_display(), test_verbose_does_not_warn_when_gemini_budget_is_set(), test_verbose_does_not_warn_when_thinking_is_disabled(), test_verbose_reports_gemini_api_default_without_thinking_budget()

### Community 127 - "conftest.py"
Cohesion: 0.50
Nodes (3): pytest_sessionfinish(), Shared pytest configuration for the repository., Remove the project-local pytest temporary directory after a run.

## Knowledge Gaps
- **273 isolated node(s):** `akasha_terminal`, `Quick Start (Local Development)`, `Change log`, `Standard Installation`, `Lightweight Installation (API-call-only, v1.0+)` (+268 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `dbs` connect `gemini_model` to `basic_llm`, `個人資料.md`, `gen_prompt.py`, `個人資訊.md`, `Embeddings`, `self_query_filter.py`, `atman`, `__init__.py`, `居住地資訊.md`, `mySVMRetriever`, `get_retrivers`, `aiido_upload`, `MemoryManager`, `test_agents_observation_normalization.py`, `self_ask.py`, `Akasha Upgrade Plan: Light Version Support`, `file_loader.py`, `test_summary.py`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._
- **Why does `Skill` connect `地點資訊.md` to `ResolvedSkillTools`, `remote_model`, `load_docs_from_webengine`, `SkillContext`, `gemini_rag`, `search_doc.py`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `basic_llm` connect `basic_llm` to `atman`, `BaseRetriever`, `gemini_model`, `remote_model`, `api.py`, `test_agents_observation_normalization.py`, `test_live_model_final_action_aliases`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `dbs` (e.g. with `eval` and `MemoryManager`) actually correct?**
  _`dbs` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `DynamicSkillMiddleware` (e.g. with `Skill` and `SkillContext`) actually correct?**
  _`DynamicSkillMiddleware` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `Skill` (e.g. with `._has_skill_references()` and `_AvailableSkill`) actually correct?**
  _`Skill` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `call_model()` (e.g. with `_generate_single_choice_question()` and `._create_compare_questionset()`) actually correct?**
  _`call_model()` has 4 INFERRED edges - model-reasoned connections that need verification._