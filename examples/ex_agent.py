import akasha

MODEL = "openai:gpt-4o"

### you can use agents to let llm has ability to call tool###


### first, we define a function to retrieve today's date ###
def today_f():
    from datetime import datetime
    now = datetime.now()

    return f"today's date: " + str(now.strftime("%Y-%m-%d %H:%M:%S"))


# then we create a tool to call the function, add function name and description
# so llm can recognize the tool
today_tool = akasha.create_tool(
    "today_date_tool",
    "This is the tool to get today's date, the tool don't have any input parameter.",
    today_f)

### create an agent with the tool, so llm can use tool to answer the question correctly ###
agent = akasha.agents(tools=[today_tool],
                      model=MODEL,
                      temperature=1.0,
                      verbose=True,
                      keep_logs=True)

agent("今天幾月幾號?")

agent.save_logs("logs.json")

### akasha have some built-in tools, you can use them directly ###
import akasha.agent.agent_tools as at
# agent_tools.rag_tool  # agent_tools.calculate_tool # agent_tools.saveJSON_tool
tool_list = [at.websearch_tool(search_engine="brave"), at.saveJSON_tool()]

agnt = akasha.agents(tools=tool_list,
                     model="openai:gpt-4o",
                     temperature=1.0,
                     max_input_tokens=8000,
                     verbose=True,
                     keep_logs=True)

agnt("用網頁搜尋工業4.0，並將資訊存成json檔iii.json")
agnt.save_logs("logs.json")
