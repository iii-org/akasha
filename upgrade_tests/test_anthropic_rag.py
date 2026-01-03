import pytest
import akasha
import os

@pytest.fixture
def anthropic_rag():
    """
    Fixture for RAG using Anthropic.
    Ensure ANTHROPIC_API_KEY is set in environment.
    """
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY not found in environment")
    
    return akasha.RAG(
        model="anthropic:claude-3-5-sonnet",
        verbose=True
    )

def test_anthropic_rag_call(anthropic_rag):
    """
    Test a simple RAG call using Anthropic.
    """
    test_file = "test_doc_anthropic.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("DeepMind is a research lab owned by Google.")
    
    try:
        response = anthropic_rag(
            data_source=test_file,
            prompt="Who owns DeepMind?"
        )
        assert "Google" in response
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
