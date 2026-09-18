import akasha
import pytest
from tests.support.live import load_test_env, require_keys


@pytest.mark.summary
@pytest.mark.live
@pytest.mark.requires_api
@pytest.mark.smoke
def test_summary():
    require_keys("GEMINI_API_KEY")
    summ = akasha.summary(
        "gemini:gemini-2.5-flash",
        sum_type="map_reduce",
        sum_len=1000,
        language="en",
        keep_logs=True,
        max_input_tokens=3000,
        chunk_size=501,
        chunk_overlap=41,
        env_file=load_test_env(),
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
