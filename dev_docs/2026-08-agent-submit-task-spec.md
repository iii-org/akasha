# Agent Submit Task Specification

## Problem Statement

Akasha currently exposes two ways to invoke an Agent:

- the synchronous callable interface, which returns the final answer after the
  Agent finishes;
- the asynchronous `acall` interface, which must be awaited by async callers.

Both interfaces ultimately produce the same final answer, but users who want
to start work and continue with another task must understand coroutines,
`await`, event loops, and `asyncio.create_task`. This makes the intended
"start now, collect the result later" workflow harder to discover than it
needs to be.

The requested improvement is a task-oriented public interface. A synchronous
method should make the caller wait for the answer. A submission method should
return a task object immediately, expose observable lifecycle state, and allow
the caller to retrieve the final Agent answer after the task completes.

ProcPulse provides useful prior art for lifecycle state, waiting, cancellation,
timeouts, and result objects. However, ProcPulse manages external OS
processes, while an Akasha Agent is an in-process LLM/tool workflow. The
feature must therefore borrow the task contract without making ProcPulse the
execution engine for Agent calls.

## Solution

Add an Agent task abstraction with two clear user-facing operations:

- a synchronous operation that waits and returns the final answer;
- a submission operation that starts an Agent call and returns an `AgentTask`.

The `AgentTask` represents one independent Agent invocation. It owns that
invocation's state, final response, error, timing, logs, messages, tool calls,
and cancellation outcome. It must be safe to inspect while the task is
running and must provide a clear way to wait for completion.

The existing callable and `acall` interfaces remain compatible during the MVP.
The beginner documentation may present the synchronous operation and task
submission as the primary concepts, while `acall` remains an advanced
integration interface for applications that already use an async event loop.

## User Stories

1. As an Akasha user, I want one synchronous Agent operation, so that I can
   receive the final answer without learning coroutine terminology.
2. As an Akasha user, I want to submit an Agent task and immediately receive a
   task object, so that I can continue doing other work.
3. As an Akasha user, I want to inspect whether a submitted task is pending,
   running, completed, failed, or cancelled, so that I can present accurate
   progress to my application.
4. As an Akasha user, I want to retrieve the final answer from a completed
   task, so that submitted work has the same useful result as a synchronous
   Agent call.
5. As an Akasha user, I want waiting for a task to be explicit, so that I can
   choose whether to continue other work or block until the answer is ready.
6. As an Akasha user, I want task failures to be observable as task errors,
   so that I can handle provider, tool, and runtime failures without losing
   the task identity.
7. As an Akasha user, I want to request cancellation, so that a long-running
   Agent task does not continue indefinitely when its result is no longer
   needed.
8. As an Akasha user, I want optional task timeouts, so that provider or tool
   calls cannot exceed the time budget of the surrounding workflow.
9. As an Akasha user, I want the submitted task to preserve its own messages,
   tool calls, thinking data, logs, and response, so that concurrent tasks do
   not overwrite one another's observable results.
10. As an Akasha user, I want multiple submitted tasks to run concurrently,
    so that independent Agent requests do not need to wait in sequence.
11. As an Akasha user, I want to use the task abstraction from an async
    application, so that I do not need to wrap every call manually with
    `asyncio.create_task`.
12. As an Akasha user, I want to wait for a task from synchronous code, so that
    task submission is useful outside an async application as well.
13. As an Akasha maintainer, I want the task result to have the same final
    answer contract as the existing Agent call, so that the feature does not
    introduce a second answer format.
14. As an Akasha maintainer, I want existing callable and `acall` behavior to
    remain compatible, so that current applications do not need an immediate
    migration.
15. As an Akasha maintainer, I want task state transitions to be deterministic,
    so that status displays and tests do not depend on timing races.
16. As an Akasha maintainer, I want task cleanup to be explicit, so that
    background work does not leak after the owning application shuts down.
17. As an Akasha maintainer, I want Agent tools and Skills to work through a
    submitted task exactly as they work through a normal Agent call, so that
    submission changes scheduling rather than tool semantics.
18. As an Akasha maintainer, I want ProcPulse to remain optional, so that
    in-process Agent execution does not become dependent on an external
    process manager.
19. As an Akasha documentation maintainer, I want the basic guide to explain
    waiting versus background submission without requiring users to learn
    event-loop internals first.
20. As an Akasha documentation maintainer, I want the advanced async API to
    remain documented for FastAPI, MCP, and other native async integrations.

## Implementation Decisions

- Introduce a dedicated `AgentTask` abstraction for one Agent invocation.
  Task state and result data must be per invocation, not stored only on the
  shared Agent facade.
- Use an explicit submission name such as `submit` or `start`; do not expose a
  method named `async`, because `async` is a Python language keyword and does
  not form a suitable public method name.
- Keep the synchronous operation semantically simple: it waits for completion
  and returns the final answer string or raises the underlying failure.
- Make submission return a task handle immediately. The handle must expose
  lifecycle status, response, error, start/end timing, and a wait operation.
- Define the initial lifecycle states as `pending`, `running`, `completed`,
  `failed`, and `cancelled`. State transitions must be monotonic and terminal
  states must not be overwritten by later polling.
- A task response is unavailable until successful completion. Access before
  completion must have a documented behavior, preferably `None` or a clear
  task-state error rather than a misleading empty answer.
- Provide both a synchronous wait boundary and an async wait boundary only if
  they can share one task state/result implementation. The public task object
  should not require users to understand the internal coroutine used to run the
  Agent.
- Preserve the existing Agent tool-calling loop. Submission must still invoke
  the configured model, tools, Skills, recursion limit, logging, and final
  answer extraction through the same high-level execution seam.
- Isolate mutable per-call fields. Concurrent submissions must not race over
  the Agent facade's question, response, messages, thoughts, tool calls,
  tokens, timestamps, or logs.
- Define cancellation semantics explicitly. Cancellation should request
  cancellation of the underlying async work where supported; it must report
  whether the task was cancelled before completion and must not falsely claim
  cancellation after a completed response exists.
- Define timeout semantics explicitly. A timeout should produce a distinct
  failed or timed-out outcome and should preserve the underlying diagnostic
  information without silently converting it into a successful empty answer.
- Keep `acall` as the native async escape hatch for applications that need to
  compose Agent calls directly with other coroutines. It is not removed from
  the implementation or advanced documentation in the MVP.
- Do not use ProcPulse to launch the Agent in a separate OS process. ProcPulse's
  process lifecycle model is useful design prior art, but its subprocess
  boundary would break in-memory tools, Skills, provider objects, and Agent
  logs.
- ProcPulse may remain useful for a separate integration boundary: an Akasha
  tool that intentionally runs an external command can use ProcPulse for
  process status, output, timeout, and termination. That is not part of this
  AgentTask MVP.
- Preserve backward compatibility for existing `agent(...)` and
  `await agent.acall(...)` callers. Any new method should be additive first.
- Update English and Traditional Chinese user-guide pages together. Basic
  examples should focus on synchronous use and task submission; native
  `acall` usage should be described as an advanced async integration pattern.
- Do not promise that submission means a separate OS process. The documented
  guarantee is that the caller receives a task handle before the final answer
  is available and may continue compatible work while the task runs.

## Testing Decisions

- Tests must assert public task behavior rather than private scheduling
  implementation details. The key evidence is returned task state, response,
  error, waiting behavior, cancellation result, and isolation between tasks.
- Add deterministic tests using a fake Agent runner or fake model/tool seam so
  tests do not require provider credentials or network access.
- Test synchronous execution returns the same final answer contract as the
  current callable interface.
- Test submission returns before a deliberately delayed fake task completes,
  then test that waiting exposes the final answer.
- Test every lifecycle transition, including successful completion, provider or
  tool failure, cancellation before completion, cancellation after completion,
  and timeout if timeout is included in the first implementation.
- Test response and error access before and after terminal states according to
  the documented contract.
- Test two concurrent submissions on one Agent facade and assert that each
  task retains its own question, response, messages, tool calls, logs, and
  timing. This test is required because the current facade stores mutable
  call data on the Agent instance.
- Test a submitted Agent with a tool and a Skill at the public Agent seam.
  Assert observable tool invocation and final answer behavior, not the exact
  internal LangGraph node sequence.
- Test synchronous waiting from a normal Python context and async waiting from
  an existing event loop. A test must also verify that the documented API does
  not call `asyncio.run` from an already-running loop.
- Test cleanup and manager/application shutdown so submitted work does not
  leave unobserved background exceptions or daemon tasks.
- Keep existing callable, `acall`, stream, MCP, Skill, and logging regression
  tests. New task tests supplement rather than replace those contracts.
- Use focused tests first, followed by syntax checks and the relevant Agent,
  Tool, Skill, MCP, and Observability test groups. Live provider validation
  remains separately identified from deterministic task-contract validation.

## Out of Scope

- Replacing the existing Agent implementation or LangChain `create_agent`
  execution loop.
- Removing or immediately deprecating the existing callable or `acall` APIs.
- Making `agent(...)` automatically detect every sync and async environment.
- Turning every Agent call into an OS subprocess.
- Making ProcPulse a required Akasha dependency.
- Building a distributed job queue, persistent remote worker, web dashboard,
  or cross-process task registry.
- Guaranteeing hard cancellation of provider-side work that has already been
  sent to an external service.
- Defining a new streaming protocol for submitted tasks. Existing stream
  behavior should remain a separate, explicitly scoped concern.
- Adding fire-and-forget behavior with no task handle, error collection, or
  cleanup mechanism.
- Changing tool security, Skill resource limits, provider configuration, or
  MCP transport behavior as part of this feature.

## Further Notes

The central product decision is to distinguish two user goals:

- "Give me the answer before I continue" — synchronous operation;
- "Start the work and give me a handle so I can continue" — submitted task.

`await agent.acall(...)` and a task's eventual response can produce the same
answer, but they serve different scheduling interfaces. The task abstraction
exists to make delayed result collection visible and understandable; it is not
needed merely to obtain the same final answer asynchronously.

The first implementation should settle the task contract and per-call state
ownership before adding convenience properties or persistence. A small task
object with reliable state, result, error, wait, and cleanup behavior is more
valuable than a broad API whose status fields are race-prone.

The local `to-spec` workflow normally publishes a spec to an issue tracker,
but this request explicitly asks for a project-local discussion document. No
issue-tracker publication is performed here.
