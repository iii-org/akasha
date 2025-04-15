import akasha
import pytest


@pytest.mark.summary
def test_Summary():
    summ = akasha.summary(
        "openai:gpt-3.5-turbo",
        sum_type="map_reduce",
        sum_len=1000,
        language="en",
        keep_logs=True,
        max_input_tokens=3000,
        chunk_size=501,
        chunk_overlap=41,
    )

    assert summ.verbose is False
    assert summ.chunk_size == 501
    assert summ.chunk_overlap == 41
    assert summ.max_input_tokens == 3000

    text = summ(
        content=["https://github.com/iii-org/akasha"],
        sum_type="map_reduce",
        sum_len=300,
    )

    assert isinstance(text, str)

    return
