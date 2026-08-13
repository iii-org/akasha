# Streaming events

Streaming lets an application render output while an agent is running.

```python
import akasha

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[],
    stream=True,
    thinking=True,
)

for event in agent("Explain the difference between a vector store and an embedding model."):
    if event["type"] == "thinking":
        print("[thinking]", event["data"])
    elif event["type"] == "tool":
        print("[tool]", event["data"])
    elif event["type"] == "answer":
        print(event["data"], end="", flush=True)
```

| Event type | Meaning |
| --- | --- |
| `answer` | A chunk of the final answer. |
| `thinking` | Provider reasoning content, when available and enabled. |
| `tool` | A tool or Skill result. |

Do not assume every provider emits every event type. Always branch on the event type and handle unknown events safely.
