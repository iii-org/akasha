# Build an agent with Tools and Skills

This tutorial starts with a model-only agent, then adds one application Tool and one Skill. Follow the sections in order so that each new capability is easy to test.

## Before you start

Create and activate a virtual environment, then install Akasha:

```bash
uv venv --python 3.11

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

uv pip install "akasha-terminal[light]"
```

Set the key for your selected chat model. This tutorial uses Gemini:

```powershell
$env:GEMINI_API_KEY = "your_key"
```

Never put a real key in a committed Python file.

## Step 1: Create a model-only agent

Create `agent_step1.py`:

```python
import akasha

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[],
    stream=False,
)

answer = agent("Explain what a tool-calling agent is in two sentences.")
print(answer)
```

Run it:

```bash
python agent_step1.py
```

At this point the agent can answer questions, but it cannot perform an application action. The empty `tools` list is intentional.

## Step 2: Add a Tool

A Tool is a callable operation that the model may choose to invoke. Give it a clear name, a precise description, typed arguments, and a safe boundary.

Create `agent_step2.py`:

```python
import akasha


def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


add_tool = akasha.create_tool(
    "Add two integers. Use this when the user asks for an addition.",
    add_numbers,
    tool_name="add_numbers",
)

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    tools=[add_tool],
    stream=False,
    max_round=4,
)

answer = agent("Use the add_numbers tool to calculate 20 + 22.")
print(answer)
```

Run it:

```bash
python agent_step2.py
```

The model decides whether to call `add_numbers`; the Python function performs the actual operation. The tool description is part of the model's instructions, so make it specific and truthful.

!!! warning
    Do not expose unrestricted shell, filesystem, database, or network functions as Tools. Validate arguments and allow only the operations your application needs.

## Step 3: Add a Skill

A Skill is a directory containing a `SKILL.md` instruction file and optional resources or scripts. It describes when and how an agent should use a capability.

The repository includes a working Skill at:

```text
examples/examples_skills/hello-skill/
├─ SKILL.md
└─ scripts/greet.py
```

Create `agent_step3.py` in the repository root:

```python
from pathlib import Path

import akasha


skill_path = Path("examples/examples_skills/hello-skill").resolve()

agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    skills=[str(skill_path)],
    stream=False,
)

answer = agent(
    "Use the hello-skill to greet Alice. Follow the Skill instructions "
    "and return the script output."
)
print(answer)
```

Run it from the repository root:

```bash
python agent_step3.py
```

The Skill is not a replacement for a Tool. A Skill provides instructions, resources, and controlled tool capabilities; the agent still needs to follow the Skill's declared workflow.

## Step 4: Observe what happened

Use `stream=True` when your application needs thinking, tool, and answer events while the agent runs:

```python
agent = akasha.agents(
    model="gemini:gemini-2.5-flash",
    skills=[str(skill_path)],
    stream=True,
    thinking=True,
    verbose=True,
)

for event in agent("Use the hello-skill to greet Alice."):
    if event["type"] == "tool":
        print("[tool]", event["data"])
    elif event["type"] == "thinking":
        print("[thinking]", event["data"])
    elif event["type"] == "answer":
        print(event["data"], end="", flush=True)
```

For the event meanings, see [Streaming events](streaming.md).

## Troubleshooting

- `GEMINI_API_KEY` missing: set the environment variable in the same terminal that runs Python.
- Skill not found: run the command from the repository root or use an absolute Skill path.
- Tool is ignored: check that `create_tool()` returned a Tool and that it is passed in the `tools` list.
- Unexpected tool behavior: improve the function type annotations and description, then make the prompt explicit.

The repository contains related runnable examples in `examples/examples_skills/`.
