# `agents`

`agents` creates a tool-calling agent. The agent can decide when to call the tools that you explicitly provide.

## Create an agent

```python
import akasha

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[],
    skills=None,
    stream=False,
)
```

Common constructor options:

| Option | Meaning |
| --- | --- |
| `tools` | A tool or list of tools the agent is allowed to call. |
| `skills` | Skill paths or Skill objects that extend the agent. |
| `model` | Provider and model alias. |
| `max_round` | Maximum agent/tool interaction rounds. |
| `max_past_observation` | Maximum prior observations retained for the agent. |
| `stream` | Return streaming events instead of one final result. |
| `thinking` | Enable supported thinking/reasoning output. |
| `max_resource_bytes` | Limit resources read by Skill tooling. |

## Call an agent

```python
answer = agent("Explain when a tool-calling agent is useful.")
print(answer)
```

The non-streaming call returns the final answer. The asynchronous equivalent is:

```python
answer = await agent.acall("Run this task asynchronously.")
```

When `stream=True`, the call yields event dictionaries. See [Streaming events](../tutorials/streaming.md) for event handling.

!!! warning
    Tools are application capabilities. Validate their inputs and restrict filesystem, network, and credential access.
