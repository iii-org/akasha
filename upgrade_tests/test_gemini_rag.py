import pytest
import akasha
import os

@pytest.fixture
def gemini_rag():
    """
    Fixture for RAG using Gemini.
    Ensure GEMINI_API_KEY is set in environment.
    """
    if "GEMINI_API_KEY" not in os.environ:
        pytest.skip("GEMINI_API_KEY not found in environment")
    
    return akasha.RAG(
        model="gemini:gemini-2.5-flash",
        embeddings="gemini:text-embedding-004",
        verbose=True
    )

def test_gemini_rag_call(gemini_rag):
    """
    Test a simple RAG call using Gemini.
    """
    test_file = "test_doc_gemini.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("The capital of Taiwan is Taipei. 101 is a famous building in Taipei.")
    
    try:
        response = gemini_rag(
            data_source=test_file,
            prompt="Taipei 101在哪裡?"
        )
        assert "台北" in response or "Taipei" in response
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
