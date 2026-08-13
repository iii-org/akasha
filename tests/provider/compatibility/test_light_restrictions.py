import pytest
import sys
from unittest.mock import patch, MagicMock


def test_torch_missing_hf_model():
    """Test that hf_model raises ImportError when torch is missing."""
    with patch.dict(sys.modules, {'torch': None, 'transformers': None}):
        import akasha.helper.handle_objects as handle_objects
        importlib_reload(handle_objects) # Ensure fresh state if possible
        
        with pytest.raises(ImportError) as excinfo:
            handle_objects.handle_model("hf:any-model")
        
        assert "Feature requiring 'torch/transformers' is not installed" in str(excinfo.value)
        assert "pip install akasha-terminal[full]" in str(excinfo.value)

def test_rerank_warning_when_torch_missing():
    """Test that search_type='rerank' shows a warning when torch is missing."""
    with patch.dict(sys.modules, {'torch': None}):
        from akasha.utils.search.retrievers.base import get_retrivers
        from akasha.utils.db.db_structure import dbs
        
        mock_db = MagicMock(spec=dbs)
        mock_db.get_Documents.return_value = []
        
        # This should print a warning but not crash, returning whatever retrievers it could find
        # Since we only ask for rerank and it fails, it might raise ValueError later if list is empty
        with patch('builtins.print') as mock_print:
            with pytest.raises(ValueError): # No retrievers found because rerank was skipped
                get_retrivers(
                    mock_db,
                    "gemini:gemini-embedding-001",
                    search_type="rerank",
                )
            
            # Check if warning was printed
            warning_called = any("pip install akasha-terminal[full]" in args[0] for args, _ in mock_print.call_args_list)
            assert warning_called

def test_bert_score_missing():
    """Test that get_bert_score raises ImportError when bert_score is missing."""
    with patch.dict(sys.modules, {'bert_score': None}):
        import akasha.helper.scores as scores
        
        with pytest.raises(ImportError) as excinfo:
            scores.get_bert_score("cand", "ref")
        
        assert "Feature requiring 'bert-score' is not installed" in str(excinfo.value)


def test_chromadb_missing_raises_clear_error():
    """Test that Chroma-backed features point users to a light/base install."""
    with patch.dict(sys.modules, {"langchain_chroma": None, "chromadb": None}):
        import akasha.utils.db.chroma_compat as chroma_compat

        importlib_reload(chroma_compat)

        with pytest.raises(ImportError) as excinfo:
            chroma_compat.get_chroma_components()

        assert "Feature requiring 'chromadb/langchain-chroma' is not installed" in str(
            excinfo.value
        )
        assert "pip install akasha-terminal[light]" in str(excinfo.value)

def importlib_reload(module):
    import importlib
    importlib.reload(module)

if __name__ == "__main__":
    pytest.main([__file__])
