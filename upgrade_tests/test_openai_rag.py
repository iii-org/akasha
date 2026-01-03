import pytest
import akasha
import os
import sys

@pytest.fixture
def api_rag():
    """
    Fixture for RAG using OpenAI (default).
    Ensure OPENAI_API_KEY is set in environment.
    """
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY not found in environment")
    
    return akasha.RAG(
        model="openai:gpt-3.5-turbo",
        embeddings="openai:text-embedding-3-small",
        verbose=True
    )

def test_openai_rag_call(api_rag):
    """
    Test a simple RAG call using OpenAI.
    This should work without torch/transformers once refactored.
    """
    # Create a dummy data source: a small text file
    test_file = "test_doc.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("The capital of France is Paris. The Eiffel Tower is in Paris.")
    
    try:
        response = api_rag(
            data_source=test_file,
            prompt="What is the capital of France?"
        )
        assert "Paris" in response
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def test_no_torch_imported():
    """
    Test that importing akasha and running a basic OpenAI task
    does not pull in torch if we are in 'light' mode.
    """
    # This test is a placeholder for the logic that will verify torch is not loaded.
    # We can check sys.modules after a fresh import in a subprocess.
    pass

def test_graceful_failure_local_model():
    """
    Test that trying to use a local model without torch/transformers
    raises a clear ImportError.
    """
    # Once refactored, this should raise ImportError if torch is not installed.
    # For now, it might still try to import torch.
    pass
