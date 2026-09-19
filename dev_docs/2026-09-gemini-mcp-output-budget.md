# Gemini MCP smoke-test output budget (2026-09-19)

The original Gemini 3.5 Flash live MCP case was reproduced unchanged: with
max_output_tokens=128, the visible answer was `I will now` and no tool was called.
A callback inspecting only completion metadata reported `MAX_TOKENS`, 124 output
tokens and 121 reasoning tokens. No private reasoning text was recorded.

Changing only the cap to 1024 produced a completed mcp_add call and final answer
containing 42. The model turns reported STOP, with 182 and 592 output tokens
respectively. Thus the failed assertion was caused by a truncated model turn,
not MCP discovery failure. The SDK warning about lifespan appeared on successful
runs too and was not the cause.

The final test reads `capabilities.output_token_limit` from the model manifest.
For `gemini:gemini-3.5-flash`, it uses the officially documented 65,536 token
ceiling (verified 2026-09-19). Models without a documented manifest limit use a
2,048-token test budget. This is a cap, not a request to generate that many tokens;
it allows more potential usage than the earlier empirically sufficient budget.
Assertions still require the answer 42, the mcp_add call and serializable logs.
Failure diagnostics include finish reasons, token usage and tool names.

Official sources:

- [Gemini 3.5 Flash model specification](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash):
  output token limit 65,536.
- [Gemini thinking token limits](https://ai.google.dev/gemini-api/docs/generate-content/thinking):
  max_output_tokens includes thought tokens and can truncate generation with
  MAX_TOKENS; lowering this cap does not lower the model's thinking level.

No public Agent/provider behavior was changed. In the current Gemini adapter,
thinking=False does not explicitly disable provider-side reasoning; it omits the
native thinking options, allowing provider defaults. This test must therefore
budget for those tokens even though it does not request thinking output.

Verification:

- Original Gemini case: reproduced failure.
- Corrected OpenAI and Gemini MCP cases: 2 passed, 5 deselected.
- Related Gemini provider and Agent live cases: 11 passed, 24 deselected without
  changing those tests, so the budget change remains scoped to MCP tool execution.
- Hosted GitHub CI has not been rerun here.
