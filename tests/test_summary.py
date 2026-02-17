import akasha
import pytest
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / "tests" / "test_upgrade" / ".env"


@pytest.mark.summary
def test_Summary():
    summ = akasha.summary(
        "gemini:gemini-2.5-flash",
        sum_type="map_reduce",
        sum_len=1000,
        language="en",
        keep_logs=True,
        max_input_tokens=3000,
        chunk_size=501,
        chunk_overlap=41,
        env_file=str(ENV_FILE),
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
