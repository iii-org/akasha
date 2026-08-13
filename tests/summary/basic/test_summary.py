import akasha
import pytest
import os
from pathlib import Path
from dotenv import load_dotenv
from tests.support.paths import TEST_ENV_FILE

ENV_FILE = TEST_ENV_FILE
load_dotenv(ENV_FILE, override=True)

if not ENV_FILE.exists() or not os.getenv("GEMINI_API_KEY"):
    pytest.skip("GEMINI_API_KEY is required for summary integration tests", allow_module_level=True)


@pytest.mark.summary
@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.smoke
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
