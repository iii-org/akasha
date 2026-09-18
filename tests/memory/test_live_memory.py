"""Live persistence and retrieval contract for MemoryManager."""

import pytest

import akasha
from tests.support.live import load_test_env, require_keys


pytestmark = [
    pytest.mark.live,
    pytest.mark.contract,
    pytest.mark.requires_api,
    pytest.mark.smoke,
]


def test_memory_persists_and_retrieves_a_grounded_fact(tmp_path):
    require_keys("GEMINI_API_KEY")
    env_file = load_test_env()
    settings = {
        "memory_name": "live-contract",
        "memory_dirname": str(tmp_path),
        "model": "gemini:gemini-2.5-flash",
        "embeddings": "gemini:gemini-embedding-001",
        "env_file": env_file,
    }
    manager = akasha.MemoryManager(**settings)
    manager.add_memory(
        "What is the memory verification code?",
        "The memory verification code is MEMORY-7319.",
        language="en",
    )

    first_results = manager.search_memory("memory verification code", top_k=2)
    reloaded = akasha.MemoryManager(**settings)
    reloaded_results = reloaded.search_memory("MEMORY-7319", top_k=2)

    assert any("MEMORY-7319" in item for item in first_results)
    assert any("MEMORY-7319" in item for item in reloaded_results)
    assert any("MEMORY-7319" in item for item in reloaded.show_memory())
