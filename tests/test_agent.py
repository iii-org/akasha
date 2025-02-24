import pytest
import akasha


def today_f():
    from datetime import datetime
    now = datetime.now()

    return f"today's date: " + str(now.strftime("%Y-%m-%d %H:%M:%S"))


today_tool = akasha.create_tool(
    "today_date_tool",
    "This is the tool to get today's date, the tool don't have any input parameter.",
    today_f)


@pytest.mark.agent
def test_agent():

    agent = akasha.agents(tools=[today_tool],
                          temperature=1.0,
                          verbose=True,
                          keep_logs=True)

    res = agent("今天幾月幾號?")

    assert type(agent.tool_name_str) == str
    assert type(res) == str

    return
