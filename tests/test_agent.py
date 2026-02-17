import pytest
import akasha
from pathlib import Path
from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parents[1] / "tests" / "test_upgrade" / ".env"
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
