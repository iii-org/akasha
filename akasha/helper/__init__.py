from importlib import import_module

__all__ = [
    "separate_name",
    "decide_embedding_type",
    "get_embedding_type_and_name",
    "get_doc_length",
    "get_docs_length",
    "sim_to_trad",
    "extract_json",
    "handle_model_type",
    "handle_embeddings",
    "handle_model",
    "myTokenizer",
    "detect_encoding",
    "get_mac_address",
    "get_text_md5",
    "call_model",
    "call_batch_model",
    "call_stream_model",
    "call_image_model",
    "call_translator",
    "call_JSON_formatter",
    "merge_history_and_prompt",
    "load_docs_from_webengine",
    "get_text_from_url",
    "get_llm_score",
    "get_toxic_score",
    "get_bert_score",
    "get_rouge_score",
    "self_query",
]

_LAZY_IMPORTS = {
    "separate_name": (".base", "separate_name"),
    "decide_embedding_type": (".base", "decide_embedding_type"),
    "get_embedding_type_and_name": (".base", "get_embedding_type_and_name"),
    "get_doc_length": (".base", "get_doc_length"),
    "get_docs_length": (".base", "get_docs_length"),
    "sim_to_trad": (".base", "sim_to_trad"),
    "extract_json": (".base", "extract_json"),
    "handle_model_type": (".handle_objects", "handle_model_type"),
    "handle_embeddings": (".handle_objects", "handle_embeddings"),
    "handle_model": (".handle_objects", "handle_model"),
    "myTokenizer": (".token_counter", "myTokenizer"),
    "detect_encoding": (".encoding", "detect_encoding"),
    "get_mac_address": (".encoding", "get_mac_address"),
    "get_text_md5": (".encoding", "get_text_md5"),
    "call_model": (".run_llm", "call_model"),
    "call_batch_model": (".run_llm", "call_batch_model"),
    "call_stream_model": (".run_llm", "call_stream_model"),
    "call_image_model": (".run_llm", "call_image_model"),
    "call_translator": (".run_llm", "call_translator"),
    "call_JSON_formatter": (".run_llm", "call_JSON_formatter"),
    "merge_history_and_prompt": (".preprocess_prompts", "merge_history_and_prompt"),
    "load_docs_from_webengine": (".web_engine", "load_docs_from_webengine"),
    "get_text_from_url": (".crawler", "get_text_from_url"),
    "get_llm_score": (".scores", "get_llm_score"),
    "get_toxic_score": (".scores", "get_toxic_score"),
    "get_bert_score": (".scores", "get_bert_score"),
    "get_rouge_score": (".scores", "get_rouge_score"),
    "self_query": (".self_query_filter", "self_query"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
