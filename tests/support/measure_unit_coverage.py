from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest
from coverage.parser import PythonParser


ROOT = Path(__file__).resolve().parents[2]

TARGETS = {
    ROOT / "akasha" / "utils" / "db" / "db_structure.py",
    ROOT / "akasha" / "utils" / "search" / "retrievers" / "base.py",
    ROOT / "akasha" / "helper" / "encoding.py",
    ROOT / "akasha" / "helper" / "crawler.py",
    ROOT / "akasha" / "helper" / "base.py",
    ROOT / "akasha" / "helper" / "preprocess_prompts.py",
    ROOT / "akasha" / "utils" / "search" / "search_doc.py",
    ROOT / "akasha" / "utils" / "logging_config.py",
    ROOT / "akasha" / "agent" / "agents.py",
}


def load_module(relative_path: str, name: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalize(path: str) -> Path:
    return Path(path).resolve()


executed: dict[Path, set[int]] = {path: set() for path in TARGETS}


def tracer(frame, event, arg):
    filename = normalize(frame.f_code.co_filename)
    if event == "call":
        try:
            filename.relative_to(ROOT)
        except ValueError:
            return None
        return tracer
    if event == "line" and filename in executed:
        executed[filename].add(frame.f_lineno)
    return tracer


def load_test_modules():
    return {
        "test_db_structure": load_module("tests/rag/input/test_db_structure.py", "test_db_structure"),
        "test_retrievers_base": load_module("tests/rag/retrieval/test_retrievers.py", "test_retrievers_base"),
        "test_helper_base": load_module("tests/rag/input/test_helpers.py", "test_helper_base"),
        "test_preprocess": load_module("tests/ask/prompt/test_preprocess.py", "test_preprocess_prompts"),
        "test_search_doc": load_module("tests/rag/retrieval/test_search_docs.py", "test_search_doc"),
        "test_encoding": load_module("tests/rag/input/test_encoding.py", "test_encoding"),
        "test_crawler": load_module("tests/ask/info/url/test_crawler.py", "test_crawler"),
        "test_agents_core": load_module("tests/agent/basic/test_core.py", "test_agents_core"),
        "test_logging": load_module("tests/observability/logging/test_logging_config.py", "test_logging_config_unit"),
    }


def run_selected_tests(modules):
    test_db_structure = modules["test_db_structure"]
    test_retrievers_base = modules["test_retrievers_base"]
    test_helper_base = modules["test_helper_base"]
    test_preprocess = modules["test_preprocess"]
    test_search_doc = modules["test_search_doc"]
    test_encoding = modules["test_encoding"]
    test_crawler = modules["test_crawler"]
    test_agents_core = modules["test_agents_core"]
    test_logging = modules["test_logging"]

    monkeypatch = pytest.MonkeyPatch()
    try:
        test_db_structure.test_dbs_defaults_to_empty_lists()
        test_db_structure.test_dbs_initializes_from_chroma_like_object_with_fallbacks()
        test_db_structure.test_merge_and_add_chromadb_keep_unique_ids()
        test_db_structure.test_get_documents_returns_langchain_documents()
        test_db_structure.test_storage_directory_handles_path_dot_and_url()
        test_db_structure.test_url_and_path_sanitization_helpers()

        test_retrievers_base.test_get_retrievers_builds_expected_retriever_types(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_retrievers_base.test_get_retrievers_supports_custom_callable(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_retrievers_base.test_get_retrievers_warns_when_rerank_support_is_missing(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_retrievers_base.test_get_retrievers_raises_on_unknown_search_type(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()

        test_helper_base.test_separate_name_and_embedding_name_helpers()
        test_helper_base.test_doc_length_helpers_and_conversion(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_helper_base.test_extract_json_and_extract_multiple_json()

        test_preprocess.test_merge_history_and_prompt_without_history(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_preprocess.test_merge_history_and_prompt_for_chat_format(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_preprocess.test_merge_history_and_prompt_for_string_format(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_preprocess.test_retri_history_messages_limits_pairs_and_tokens(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()

        test_search_doc.test_merge_docs_deduplicates_and_stops_on_token_limit(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_search_doc.test_search_docs_uses_auto_helpers(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_search_doc.test_search_docs_merges_multiple_retrievers(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_search_doc.test_retri_docs_uses_auto_and_deduplicates(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_encoding.test_detect_encoding_reads_file_prefix(Path(tmp_dir))
        test_encoding.test_md5_and_mac_address_are_stable(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()

        test_crawler.test_get_text_from_url_extracts_title_and_visible_text(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_crawler.test_get_text_from_url_handles_request_exceptions(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        test_crawler.test_get_webpage_last_modified_handles_present_and_missing_headers(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()

        test_agents_core.test_final_action_aliases_are_case_insensitive()

        test_logging.test_configure_logging_reuses_console_handler(monkeypatch)
        monkeypatch.undo(); monkeypatch = pytest.MonkeyPatch()
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_logging.test_configure_logging_replaces_file_handler_when_path_changes(Path(tmp_dir))
            logging_config = modules["test_logging"].logging_config
            root_logger = __import__("logging").getLogger()
            if logging_config._file_handler is not None:
                root_logger.removeHandler(logging_config._file_handler)
                logging_config._file_handler.close()
                logging_config._file_handler = None
                logging_config._file_path = None
    finally:
        monkeypatch.undo()


def executable_lines(path: Path) -> set[int]:
    parser = PythonParser(filename=str(path))
    parser.parse_source()
    return set(parser.statements)


def main() -> int:
    modules = load_test_modules()
    sys.settrace(tracer)
    try:
        run_selected_tests(modules)
    finally:
        sys.settrace(None)

    totals = {"statements": 0, "executed": 0}
    for path in sorted(TARGETS):
        statements = executable_lines(path)
        hit = statements & executed[path]
        totals["statements"] += len(statements)
        totals["executed"] += len(hit)
        percent = (len(hit) / len(statements) * 100) if statements else 100.0
        print(f"{path.relative_to(ROOT)}: {len(hit)}/{len(statements)} = {percent:.2f}%")

    overall = totals["executed"] / totals["statements"] * 100 if totals["statements"] else 100.0
    print(f"TOTAL: {totals['executed']}/{totals['statements']} = {overall:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
