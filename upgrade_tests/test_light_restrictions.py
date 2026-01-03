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
                get_retrivers(mock_db, "openai:embeddings", search_type="rerank")
            
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

def importlib_reload(module):
    import importlib
    importlib.reload(module)

if __name__ == "__main__":
    pytest.main([__file__])
