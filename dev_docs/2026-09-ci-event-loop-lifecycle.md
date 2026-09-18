# CI event-loop lifecycle fix (2026-09-19)

## Failure and reproduction

The OpenAI MCP live test failed in HTTP connection cleanup with
`RuntimeError: Event loop is closed`. The synchronous Agent facade uses
`asyncio.run()` per call, while pooled async HTTP connections can survive that
loop. LangChain also caches its default async HTTP client across model instances.

A local HTTP/1.1 server with persistent connections reproduces the failure without
MCP or credentials. Windows tests select the Selector event-loop policy to match
Linux transport behavior. SDK retries are disabled in the regression tests:
locally the installed SDK otherwise masked the first failed request by retrying.
TLS was initially checked, then removed from the minimal reproduction because
ordinary HTTP reproduces the same closed-loop exception with retries disabled.

## Change

OpenAI and Azure model construction now supplies a dedicated
`DefaultAsyncHttpxClient` with `Connection: close`. Async HTTP/1.1 responses release
their sockets while their owning loop is still running, and the factory bypasses
LangChain's globally cached async client. The existing synchronous HTTP pool is
unchanged. This trades async connection reuse for correct operation across the
short-lived loops used by the public synchronous Agent API. Concurrent use of a
single client across multiple running loops is not promised by this change.

The change applies to factory-created OpenAI/Azure models. Applications supplying
their own model/client remain responsible for its lifecycle. No CI tests were
skipped, no provider versions were pinned, and the existing user edit in
`pyproject.toml` was preserved.

## Verification

- Before the fix: the local real-HTTP regression fails with the original
  `RuntimeError: Event loop is closed` (wrapped by local SDK APIConnectionError).
- After the fix: `tests/provider/chat`, `tests/agent`, `tests/mcp` with
  `-m "not live"`: **44 passed, 43 deselected**.
- Eight regression cases cover OpenAI/Azure, same/new model, and direct async
  model calls/public synchronous Agent calls; each runs three consecutive calls.
- Original live case
  `tests/mcp/stdio/test_mcp_pipeline.py::test_mcp_tools_are_executed_by_real_agent_non_stream[openai:gpt-5.4]`:
  **1 passed**, using existing test credentials and the real stdio MCP server.
- Validation ran locally on Windows/Python 3.11. GitHub Linux/Python 3.12 and its
  exact installed SDK versions have not been rerun here. Deprecation warnings
  remain in existing LangChain/MCP dependencies.
