# Build an agent and use tools

Use `akasha.agents()` when a model needs to decide which tool to call as part of a task.

## Minimal agent example

```python
import akasha

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[],
    stream=False,
)

answer = agent("Explain when a tool-calling agent is useful.")
print(answer)
```

Replace the empty `tools` list with the tools your application allows. Keep the list explicit: a tool should have a clear purpose and a safe input boundary.

## Skills and MCP

Skills can provide instructions and allowlisted tools from a Skill directory. MCP tools can be normalized and passed to `akasha.agents(tools=...)`.

!!! warning
    Treat tools as application capabilities. Validate inputs, restrict filesystem or network access, and do not expose secrets through tool arguments.

For incremental output, combine this tutorial with [Streaming events](streaming.md).
