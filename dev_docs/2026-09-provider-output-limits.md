# Provider output limits and live-test budgets (2026-09-19)

## Scope

Audited all five chat models in tests/config/model_manifest.yaml. Model limits
are capability metadata; test_max_output_tokens is a separate execution budget.
The shared tests/support/model_limits.py reads the budget and rejects values that
exceed a documented ceiling. Production Akasha defaults are unchanged.

| Configured model | Documented output ceiling | Test max_output_tokens |
| --- | ---: | ---: |
| openai:gpt-5.4 | 128,000 | 8,192 |
| azure:DeepSeek-V4-Flash | 384,000 | 8,192 |
| gemini:gemini-3.5-flash | 65,536 | 65,536 (retained) |
| anthropic:claude-sonnet-4-6 | 128,000 | 8,192 |
| ollama:gemma4:26b | Deployment-dependent, no separate fixed ceiling confirmed | 8,192 |

The 8,192 figure is an engineering smoke-test budget, not an official maximum or
guarantee against truncation for arbitrary prompts. These tests request short
fixed answers; reasoning and tool execution still require headroom beyond the old
64/128/256-token caps. Gemini's previously selected 65,536 budget is retained.
Azure deployment names may be aliases: recheck metadata if the backing model changes.

Akasha's current Anthropic adapter reserves answer space when thinking is enabled.
With max_output_tokens=8192 and thinking_budget=medium it sends budget_tokens=8192
and max_tokens=9216. Factory-boundary tests verify both ordinary and thinking
requests remain within the official ceiling and the thinking budget is smaller
than the total cap. We deliberately avoid passing an official ceiling together
with a proportional thinking budget that could make the adapter exceed it.

Ollama documents num_predict as the generation cap. Gemma 4 26B's 256K context
length is not an independent output allowance; input, local num_ctx and deployment
settings matter. No hosted-model output ceiling is invented for Ollama.

## Official sources

Verified 2026-09-19; links and verification date are also stored in the manifest.

- [OpenAI GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4):
  128,000 maximum output tokens.
- [Azure DeepSeek model catalog](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure#deepseek-models-sold-by-azure):
  DeepSeek-V4-Flash output 384,000 tokens. This is the Azure-hosted specification.
- [Azure reasoning parameters](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-chat-reasoning#choose-parameters-for-reasoning-models):
  reasoning and final-answer tokens share max_completion_tokens.
- [Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash):
  65,536 output tokens.
- [Claude Sonnet 4.6](https://platform.claude.com/docs/en/models/sonnet-4-6/overview):
  standard max output 128K. Batch beta limits are not used.
- [Ollama Modelfile](https://docs.ollama.com/modelfile): num_predict versus num_ctx.
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4):
  26B A4B context length 256K.

## Affected tests

Provider Ask/Agent, ordinary/streaming/thinking contracts and MCP now share model
budgets. Other live Gemini Ask/Agent/RAG examples and the RAG provider matrix also
use the helper (8,192 fallback for models outside the manifest, such as the
existing Gemini 2.5 Flash cases). Offline parameter tests keep their deliberate
small values. The MCP streaming-rejection test does not call a model and retains
its old cap. Embedding dimensions are unrelated to generation output limits.

## Verification and limits

The user's Azure thinking stream case passed locally even with its original
256-token cap. Therefore this audit does not claim that the CI failure's exact
finish reason was confirmed. Insufficient budget is consistent with the reported
symptom and official accounting rules; a future failure still requires the actual
completion metadata to distinguish truncation from a stream parsing issue.

31 local model-limit/thinking tests passed, including actual adapter construction
for all four hosted providers in ordinary and thinking modes. Live tests completed with **38 passed, 7 skipped, 2 deselected** in 142.65 seconds.
The skipped cases require an explicitly configured Ollama endpoint. Azure's exact
thinking-stream case and all four configured cloud providers passed; coverage
includes provider contracts, MCP, Gemini Agent/Ask, and RAG pipelines. Existing
SDK deprecation/tokenizer warnings remain. GitHub Actions has not been rerun here.

Validation commands:

```powershell
.venv/Scripts/python.exe -m pytest tests/provider/chat/test_model_limits.py tests/provider/thinking -m "not live" -q -o addopts=''
$env:RUN_LIVE_TESTS = "1"
.venv/Scripts/python.exe -m pytest tests/provider/chat/test_provider_contract.py tests/mcp/stdio/test_mcp_pipeline.py tests/agent/stream/test_live_gemini.py tests/ask/basic/test_live_ask.py tests/rag/provider/test_provider_matrix.py tests/rag/provider/test_gemini_pipeline.py -m live -q -o addopts=''
```

Compilation and Git whitespace checks passed. No commit was created.
