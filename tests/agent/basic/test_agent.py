import pytest
import akasha
from tests.support.live import load_test_env, require_keys


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
@pytest.mark.live
@pytest.mark.requires_api
@pytest.mark.smoke
def test_agent():
    require_keys("GEMINI_API_KEY")
    agent = akasha.agents(
        model="gemini:gemini-2.5-flash",
        tools=[today_tool],
        temperature=1.0,
        verbose=True,
        keep_logs=True,
        env_file=load_test_env(),
    )

    res = agent("今天幾月幾號?")

    assert isinstance(agent.tool_name_str, str)
    assert isinstance(res, str)

    return
