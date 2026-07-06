from importlib import import_module

__all__ = [
    "create_single_file_db",
    "create_directory_db",
    "create_webpage_db",
    "dbs",
    "get_storage_directory",
    "process_db",
    "load_db_by_chroma_name",
    "load_docs_from_info",
    "delete_documents_by_file",
    "delete_documents_by_directory",
    "extract_db_by_file",
    "extract_db_by_ids",
    "extract_db_by_keyword",
    "pop_db_by_ids",
    "update_db",
]

_LAZY_IMPORTS = {
    "create_single_file_db": (".create_db", "create_single_file_db"),
    "create_directory_db": (".create_db", "create_directory_db"),
    "create_webpage_db": (".create_db", "create_webpage_db"),
    "dbs": (".db_structure", "dbs"),
    "get_storage_directory": (".db_structure", "get_storage_directory"),
    "process_db": (".load_db", "process_db"),
    "load_db_by_chroma_name": (".load_db", "load_db_by_chroma_name"),
    "load_docs_from_info": (".load_docs", "load_docs_from_info"),
    "delete_documents_by_file": (".delete_db", "delete_documents_by_file"),
    "delete_documents_by_directory": (".delete_db", "delete_documents_by_directory"),
    "extract_db_by_file": (".extract_db", "extract_db_by_file"),
    "extract_db_by_ids": (".extract_db", "extract_db_by_ids"),
    "extract_db_by_keyword": (".extract_db", "extract_db_by_keyword"),
    "pop_db_by_ids": (".extract_db", "pop_db_by_ids"),
    "update_db": (".upadte_db", "update_db"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
