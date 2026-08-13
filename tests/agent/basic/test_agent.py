import pytest
import akasha
import os
from pathlib import Path
from dotenv import load_dotenv
from tests.support.paths import TEST_ENV_FILE

ENV_FILE = TEST_ENV_FILE
load_dotenv(ENV_FILE, override=True)


def today_f():
    from datetime import datetime

    now = datetime.now()

    return "today's date: " + str(now.strftime("%Y-%m-%d %H:%M:%S"))


today_tool = akasha.create_tool(
    "This is the tool to get today's date, the tool don't have any input parameter.",
    today_f,
    "today_date_tool",
)


@pytest.mark.agent
@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.smoke
@pytest.mark.skipif(
    not ENV_FILE.exists() or not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY is required for this live agent test",
)
def test_agent():
    agent = akasha.agents(
        model="gemini:gemini-2.5-flash",
        tools=[today_tool],
        temperature=1.0,
        verbose=True,
        keep_logs=True,
        env_file=str(ENV_FILE),
    )

    res = agent("今天幾月幾號?")

    assert isinstance(agent.tool_name_str, str)
    assert isinstance(res, str)

    return
