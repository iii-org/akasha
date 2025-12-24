import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.messages.ai import AIMessage
from akasha.helper.run_llm import call_model, call_batch_model, call_stream_model, call_image_model


class MockModel:
    """Mock LLM model for testing"""
    def __init__(self, responses=None, empty_responses=0):
        self._llm_type = "mock"
        self.responses = responses or ["test response"]
        self.empty_responses = empty_responses
        self.call_count = 0
        
    def _call(self, input_text, verbose=False):
        self.call_count += 1
        if self.call_count <= self.empty_responses:
            return ""
        return self.responses[min(self.call_count - self.empty_responses - 1, len(self.responses) - 1)]
    
    def invoke(self, input_text, verbose=False):
        return self._call(input_text, verbose)


def test_call_model_respects_max_retries_parameter():
    """Test that call_model respects the max_retries parameter"""
    # Create a mock that returns empty response first 2 times, then a valid response
    mock_model = MockModel(responses=["valid response"], empty_responses=2)
    
    # Test with max_retries=1 (should fail after 1 attempt)
    with pytest.raises(Exception, match="LLM response is empty after max retries"):
        call_model(mock_model, "test input", verbose=False, max_retries=1)
    
    # Reset the mock
    mock_model = MockModel(responses=["valid response"], empty_responses=2)
    
    # Test with max_retries=3 (should succeed on 3rd attempt)
    result = call_model(mock_model, "test input", verbose=False, max_retries=3)
    assert result == "valid response"
    assert mock_model.call_count == 3


def test_call_model_default_max_retries():
    """Test that call_model uses default max_retries of 3"""
    mock_model = MockModel(responses=["valid response"], empty_responses=2)
    
    # Should succeed with default max_retries=3
    result = call_model(mock_model, "test input", verbose=False)
    assert result == "valid response"
    assert mock_model.call_count == 3


def test_call_model_no_retries_needed():
    """Test that call_model doesn't retry when response is valid"""
    mock_model = MockModel(responses=["valid response"], empty_responses=0)
    
    result = call_model(mock_model, "test input", verbose=False, max_retries=5)
    assert result == "valid response"
    assert mock_model.call_count == 1


def test_call_model_custom_high_retries():
    """Test that call_model can use high retry counts when specified"""
    mock_model = MockModel(responses=["valid response"], empty_responses=10)
    
    # Should succeed with max_retries=15
    result = call_model(mock_model, "test input", verbose=False, max_retries=15)
    assert result == "valid response"
    assert mock_model.call_count == 11  # 10 empty + 1 valid


def test_batch_model_respects_max_retries():
    """Test that call_batch_model respects the max_retries parameter"""
    mock_model = MockModel(responses=["response1"], empty_responses=2)
    mock_model.batch = Mock(side_effect=[
        [""],  # First call returns empty
        [""],  # Second call returns empty
        [AIMessage(content="valid response")],  # Third call returns valid
    ])
    
    result = call_batch_model(mock_model, ["test input"], verbose=False, max_retries=3)
    assert len(result) == 1
    assert result[0] == "valid response"
    assert mock_model.batch.call_count == 3


def test_stream_model_respects_max_retries():
    """Test that call_stream_model respects the max_retries parameter"""
    mock_model = MockModel(responses=["valid response"], empty_responses=0)
    
    # Create a generator that yields the response
    def mock_stream(input_text, verbose=False):
        yield "valid"
        yield " response"
    
    mock_model.stream = mock_stream
    
    result = list(call_stream_model(mock_model, "test input", verbose=False, max_retries=3))
    assert len(result) > 0
    assert "".join(result) == "valid response"


def test_image_model_respects_max_retries():
    """Test that call_image_model respects the max_retries parameter"""
    mock_model = MockModel(responses=["image url"], empty_responses=0)
    mock_model._llm_type = "openai"
    
    result = call_image_model(mock_model, "test prompt", verbose=False, max_retries=3)
    assert result == "image url"
