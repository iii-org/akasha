import akasha

MODEL = "openai:gpt-4o"

### you can use agents to let llm has ability to call tool###


### first, we define a function to retrieve today's date ###
def today_f():
    from datetime import datetime

    now = datetime.now()

    return "today's date: " + str(now.strftime("%Y-%m-%d %H:%M:%S"))


# then we create a tool to call the function, add function name and description
# so llm can recognize the tool
today_tool = akasha.create_tool(
    "today_date_tool",
    "This is the tool to get today's date, the tool don't have any input parameter.",
    today_f,
)

### create an agent with the tool, so llm can use tool to answer the question correctly ###
agent = akasha.agents(
    tools=[today_tool], model=MODEL, temperature=1.0, verbose=True, keep_logs=True
)

agent("今天幾月幾號?")

agent.save_logs("logs.json")

### akasha have some built-in tools, you can use them directly ###
import akasha.agent.agent_tools as at  # noqa: E402

# agent_tools.rag_tool  # agent_tools.calculate_tool # agent_tools.saveJSON_tool
tool_list = [at.websearch_tool(search_engine="brave"), at.saveJSON_tool()]

agnt = akasha.agents(
    tools=tool_list,
    model=MODEL,
    temperature=1.0,
    max_input_tokens=8000,
    verbose=True,
    keep_logs=True,
)

agnt("用網頁搜尋工業4.0，並將資訊存成json檔iii.json")
agnt.save_logs("logs.json")

#
#
#
#
#
#
#
###  MCP（Model Context Protocol） ###
# akasha agent can be the client of MCP server, and you can use akasha agent to call the tools in MCP server
# ### for example, first, we write a cal_server.py to create add tool and multiple tool ###

# cal_server.py
from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")

### also, we can create a weather server to get the weather(using sse) ###

# weather_server.py
from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    location = location.lower()
    if location == "taipei" or location == "台北":
        return "It's always rainning in Taipei"
    elif location == "kaohsiung" or location == "高雄":
        return "It's always sunny in Kaohsiung"
    elif location == "new york" or location == "紐約":
        return "It's always sunny in New York"
    else:
        return f"It's cloudy in {location}"


if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)

### run the weather_server.py using "pyhon weather_server.py" in command line to activate the server
### then we can use akasha agent to call the tools in MCP server ###

import asyncio  # noqa: E402
import akasha  # noqa: E402
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402

MODEL = "openai:gpt-4o"

prompt = """89*37=?"""
prompt = """tell me the weather in Taipei"""
connection_info = {
    "math": {
        "command": "python",
        # the first arg is the path of your python file
        "args": ["cal_server.py"],
        "transport": "stdio",
    },
    "weather": {
        # make sure you start your weather server with correct port
        "url": "http://localhost:8000/sse",
        "transport": "sse",
    },
}


## use MultiServerMCPClient to connect to multiple MCP servers and get the tools
async def call_agents(prompt: str):
    async with MultiServerMCPClient(connection_info) as client:
        tools = client.get_tools()
        agent = akasha.agents(
            tools=tools,
            model=MODEL,
            temperature=1.0,
            verbose=True,
            keep_logs=True,
        )

        # Use the agent asynchronously
        response = await agent.acall(prompt)
        agent.save_logs("logs_agent.json")
        return response


# Run the main function
asyncio.run(call_agents(prompt))

# Model: openai:gpt-4o, Temperature: 1.0
# Tool:  add, multiply, get_weather
# Prompt format type: auto, Max input tokens: 3000
# Thought: I need to check the current weather for Taipei. ....
