# Agent progress contract

Akasha automatically appends user-visible progress instructions to the agent's
system prompt, retaining caller instructions and dynamic skill instructions.
Progress describes observable findings and the next operation, independently of
the provider's optional thinking output.

Public seams: `agents(...)`, `agent(question)`, `await agent.acall(question)`,
stream events, console output and `agent.logs`.

- Assistant text accompanying tool calls is progress, never the final answer.
- Missing narration gets a factual tool-name fallback, without invented findings.
- Verbose displays `[progress]`, `[tool]`, and `[answer]`; quiet mode remains quiet.
- Streams add `{type: "progress", data: str}` to the existing event contract.
- Final response and logged response exclude progress. Logs retain progress.
- Tool progress is visible before execution, including non-streaming calls.
- Existing caller prompts, skills, thinking controls and async tools remain usable.

Text must be classified at the end of a model turn because a tool call can arrive
after text chunks. Buffer ambiguous text until that point rather than leak
progress into answer events. Thinking can still stream independently.

Validation uses a scripted model at the external model boundary and real
LangChain orchestration/tools; no provider credentials are required. Live model
compliance is separate from deterministic event-routing verification.

## Validation (2026-09-18)

- TDD red/green confirmed automatic prompt composition, stream separation,
  pre-execution async-tool progress, missing-final-answer handling, history
  isolation, and filtering model output produced inside tools.
- New public-facade integration tests: **16 passed**, including fragmented tool
  arguments and independent thinking events.
- Repository local suite (`-m 'not live and not full_only'`): **198 passed,
  1 skipped, 62 deselected**. This run collected the first 15 progress cases;
  the additional fragmented-arguments case passed in the subsequent 16-case run.
- The skip is the existing Windows symlink-permission case. Live provider tests
  and full dependency-profile tests were not run.
- Compile checks and whitespace checks passed for the changed code/docs.
- Used `..\.venv\Scripts\python.exe`; a unique external `--basetemp` avoids
  pre-existing Windows access restrictions on the repository `.pytest-tmp`.
