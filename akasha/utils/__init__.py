from importlib import import_module

__all__ = [
    "aiido_upload",
    "handle_language",
    "handle_score_table",
    "handle_metrics",
    "handle_params",
    "handle_table",
    "search_docs",
    "retri_docs",
]

_LAZY_IMPORTS = {
    "aiido_upload": (".upload", "aiido_upload"),
    "handle_language": (".prompts.format", "handle_language"),
    "handle_score_table": (".prompts.format", "handle_score_table"),
    "handle_metrics": (".prompts.format", "handle_metrics"),
    "handle_params": (".prompts.format", "handle_params"),
    "handle_table": (".prompts.format", "handle_table"),
    "search_docs": (".search.search_doc", "search_docs"),
    "retri_docs": (".search.search_doc", "retri_docs"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
