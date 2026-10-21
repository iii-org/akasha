"""Public feature entry points load only their selected dependencies."""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit


def probe(source):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr


def test_name_helpers_and_english_length_do_not_load_chinese_resources():
    probe("""
        import sys
        from akasha.helper import separate_name, get_doc_length
        assert separate_name('gemini:model:version') == ('gemini', 'model:version')
        assert get_doc_length('en', 'one two three') == 3
        assert 'jieba3' not in sys.modules
        assert 'opencc' not in sys.modules
    """)


def test_plain_info_preserves_documents_without_loading_file_or_web_handlers():
    probe("""
        import sys
        from langchain_core.documents import Document
        from akasha.utils.db import load_docs_from_info
        original = Document(page_content='existing', metadata={'source': 'test'})
        docs = load_docs_from_info(['', 'plain info', original])
        assert [doc.page_content for doc in docs] == ['plain info', 'existing']
        assert docs[1] is original
        assert 'akasha.utils.db.file_loader' not in sys.modules
        assert 'akasha.helper.crawler' not in sys.modules
    """)


def test_ask_can_be_constructed_without_database_or_web_features():
    probe("""
        import sys
        import os
        os.environ['GEMINI_API_KEY'] = 'test'
        import akasha
        qa = akasha.ask(model='gemini:gemini-2.5-flash', language='en',
                        env_file='', max_output_tokens=128)
        assert callable(qa)
        assert 'akasha.utils.db.load_db' not in sys.modules
        assert 'akasha.helper.crawler' not in sys.modules
        assert 'jieba3' not in sys.modules
        assert 'opencc' not in sys.modules
    """)


def test_create_tool_can_invoke_without_loading_agent_or_skills_runtime():
    probe("""
        import sys
        import akasha
        def add_one(value: int) -> int:
            return value + 1
        tool = akasha.create_tool('Add one.', add_one)
        assert tool.invoke({'value': 2}) == 3
        assert 'langchain.agents' not in sys.modules
        assert 'akasha.agent.skills.middleware' not in sys.modules
        assert 'langchain_experimental' not in sys.modules
    """)


def test_skill_registry_keeps_identity_without_loading_middleware():
    probe("""
        import sys
        import akasha
        from akasha.agent.skills import SkillRegistry, default_registry
        assert akasha.SkillRegistry is SkillRegistry
        assert akasha.default_registry is default_registry
        assert list(SkillRegistry()) == []
        assert 'langchain.agents' not in sys.modules
        assert 'akasha.agent.skills.middleware' not in sys.modules
        assert 'langchain_experimental' not in sys.modules
    """)


def test_agent_without_skills_does_not_load_skills_middleware():
    probe("""
        import sys
        import akasha
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        agent = akasha.agents(model=FakeListChatModel(responses=['LOCAL_AGENT_OK']),
                               language='en', keep_logs=False)
        assert agent('Reply briefly.') == 'LOCAL_AGENT_OK'
        assert 'akasha.agent.skills.middleware' not in sys.modules
        assert 'langchain_experimental' not in sys.modules
    """)


def test_skill_middleware_does_not_load_repl_until_execution():
    probe("""
        import sys
        import akasha
        from akasha.agent.skills import DynamicSkillMiddleware
        middleware = DynamicSkillMiddleware(akasha.Skill(name='writing', instructions='Be concise.'))
        assert middleware.available_context.names == ['writing']
        assert 'langchain_experimental' not in sys.modules
    """)


def test_bm25_retrieval_does_not_load_other_retriever_strategies():
    probe("""
        import sys
        from akasha.utils.db.db_structure import dbs
        from akasha.utils.search.retrievers.base import get_retrivers
        db = dbs()
        db.docs = ['apples fruit', 'cars road', 'apples garden']
        db.metadatas = [{'id': 1}, {'id': 2}, {'id': 3}]
        retrievers = get_retrivers(db, 'unused', search_type='bm25')
        assert len(retrievers) == 1
        docs, scores = retrievers[0].get_relevant_documents_and_scores('apples')
        assert len(docs) == 3
        assert len(scores) == 3
        for name in ('retri_knn', 'retri_svm', 'retri_mmr', 'retri_tfidf', 'retri_rerank'):
            assert 'akasha.utils.search.retrievers.' + name not in sys.modules
    """)


def test_llm_scoring_does_not_load_rouge_or_chinese_tokenizer():
    probe("""
        import sys
        from akasha.helper import get_llm_score
        from langchain_core.language_models.fake import FakeListLLM
        assert get_llm_score('answer', 'reference', FakeListLLM(responses=['0.8'])) == 0.8
        assert 'rouge' not in sys.modules
        assert 'rouge_chinese' not in sys.modules
        assert 'jieba3' not in sys.modules
    """)


def test_api_cleanup_does_not_attempt_to_import_unused_torch():
    probe("""
        import builtins
        import akasha.api
        original = builtins.__import__
        attempts = []
        def tracked_import(name, *args, **kwargs):
            if name == 'torch':
                attempts.append(name)
                raise ImportError('torch is not in use')
            return original(name, *args, **kwargs)
        builtins.__import__ = tracked_import
        akasha.api.clean()
        assert attempts == []
    """)


def test_image_exports_do_not_load_clients_or_preview_library():
    probe("""
        import sys
        import akasha
        assert callable(akasha.gen_image)
        assert callable(akasha.edit_image)
        assert 'akasha.helper.handle_objects' not in sys.modules
        assert 'PIL.Image' not in sys.modules
    """)


def test_api_cleanup_still_releases_already_loaded_cuda_resources():
    probe("""
        import sys
        from types import SimpleNamespace
        import akasha.api
        calls = []
        sys.modules['torch'] = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True,
            ipc_collect=lambda: calls.append('ipc'),
            empty_cache=lambda: calls.append('cache'),
        ))
        akasha.api.clean()
        assert calls == ['ipc', 'cache']
    """)


def test_chinese_conversion_is_shared_across_concurrent_first_use():
    probe("""
        from concurrent.futures import ThreadPoolExecutor
        import opencc
        from akasha.helper import sim_to_trad
        original = opencc.OpenCC
        calls = []
        def create(config):
            calls.append(config)
            return original(config)
        opencc.OpenCC = create
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(sim_to_trad, ['汉语'] * 16))
        assert results == ['漢語'] * 16
        assert calls == ['s2twp']
    """)


def test_chinese_conversion_can_retry_after_initialization_failure():
    probe("""
        import opencc
        from akasha.helper import sim_to_trad
        original = opencc.OpenCC
        def fail(config):
            raise RuntimeError('resource unavailable')
        opencc.OpenCC = fail
        try:
            sim_to_trad('汉语')
        except RuntimeError:
            pass
        else:
            raise AssertionError('initialization failure was hidden')
        opencc.OpenCC = original
        assert sim_to_trad('汉语') == '漢語'
    """)


def test_english_rouge_preserves_score_without_chinese_resources():
    probe("""
        import sys
        from akasha.helper import get_rouge_score
        assert get_rouge_score('one two three', 'one two three', language='en') == 1.0
        # Some rouge-chinese wheels also provide the rouge namespace. Check
        # the actual heavyweight resources, not that upstream import alias.
        assert 'jieba3' not in sys.modules
        assert 'opencc' not in sys.modules
        assert 'akasha.helper.chinese' not in sys.modules
    """)


def test_url_info_preserves_title_and_url_metadata():
    probe("""
        from types import SimpleNamespace
        import requests
        from akasha.utils.db import load_docs_from_info
        requests.get = lambda *args, **kwargs: SimpleNamespace(
            content=b'<html><title>Example title</title><body><p>Example text</p></body></html>',
            raise_for_status=lambda: None,
        )
        doc, = load_docs_from_info('https://example.com/info')
        assert 'Example text' in doc.page_content
        assert doc.metadata == {'title': 'Example title', 'url': 'https://example.com/info'}
    """)


def test_public_agent_class_is_stable_after_direct_implementation_import():
    probe("""
        import importlib
        import akasha
        implementation = importlib.import_module('akasha.agent.agents')
        from akasha.agent import agents
        assert agents is implementation.agents
        assert akasha.agents is implementation.agents
        assert agents is akasha.agents
    """)


def test_selected_rouge_reports_missing_dependency_instead_of_zero_score():
    probe("""
        import builtins
        from akasha.helper import get_rouge_score
        original = builtins.__import__
        def block_rouge(name, *args, **kwargs):
            if name == 'rouge':
                raise ImportError('ROUGE unavailable')
            return original(name, *args, **kwargs)
        builtins.__import__ = block_rouge
        try:
            get_rouge_score('one two', 'one two', language='en')
        except ImportError:
            pass
        else:
            raise AssertionError('Missing dependency was hidden as a score')
    """)
