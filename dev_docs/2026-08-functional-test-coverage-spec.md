# Functional Test Coverage and Classification Spec

## Problem Statement

The test suite is currently organized primarily by execution characteristics such as unit, contract, smoke, integration, and upgrade tests. That distinction is useful for selecting test cost, but it does not answer the product question quickly enough: which Akasha capabilities are covered, and which capabilities are currently green?

Several important capabilities are also hidden inside broad regression files. Vision and Memory are examples: both have live coverage, but neither has an obvious feature-level test entry point or a complete coverage view. Skills, tools, and MCP are similarly distributed across agent tests, contract tests, fixtures, and provider smoke tests.

## Solution

Organize the test suite by functional capability while retaining test-level markers as a second, orthogonal dimension. Add a maintained functional coverage matrix that records the expected behavior, test level, dependency requirements, and current verification status for every supported capability.

The physical test tree is feature-first. Examples include `tests/ask/stream/`, `tests/ask/info/url/`, `tests/agent/stream/`, `tests/skill/loading/`, `tests/tool/schema/`, `tests/rag/parameters/`, `tests/rag/retrieval/`, and `tests/mcp/streamable_http/`. `unit`, `contract`, `integration`, `smoke`, and `upgrade` are pytest markers only, not top-level test directories. The cross-feature API stability file is the explicit compatibility exception.

The feature taxonomy is:

- Ask: basic ask, non-stream, stream, thinking, file/directory/URL `info` input, and response contract.
- Agents: basic agent execution, tool calling, stream events, final answer handling, skills, and MCP tools.
- Memory: add, search, show, persistence/reload, path handling, and memory-assisted ask/agent behavior.
- Vision: image input, prompt formatting, provider integration, multiple images, invalid paths, and response contract.
- RAG: file/directory/list/database input, chunking, retrieval strategy, Chroma persistence, grounded answer, prompt and generation controls, streaming, logging, and path safety.
- MCP: stdio, Streamable HTTP, discovery, schemas, invocation, result normalization, and async-only behavior.
- Skills: metadata loading, deferred loading, resources, scripts, REPL, path safety, and tool exposure.
- Tools: public tool creation, schema validation, invocation, error propagation, and agent integration.
- Observability: verbose output, keep_logs, log serialization, stream events, and console/file logging.
- Providers: provider factory boundaries, aliases, chat contracts, embedding contracts, and thinking configuration.

## User Stories

1. As an Akasha maintainer, I want tests grouped by capability, so that I can find the tests for a feature without searching every test layer.
2. As an Akasha maintainer, I want to run all Ask tests with one command, so that I can validate the Ask feature as a whole.
3. As an Akasha maintainer, I want to run all Agent tests with one command, so that tool calling, skills, and MCP behavior can be reviewed together.
4. As an Akasha maintainer, I want Memory tests to be a first-class feature group, so that persistence and retrieval coverage is visible.
5. As an Akasha maintainer, I want Vision tests to be a first-class feature group, so that image-input coverage is not hidden inside API stability tests.
6. As an Akasha maintainer, I want MCP tests to be visible as a feature group, so that transport and tool-result compatibility can be verified independently.
7. As an Akasha maintainer, I want Skills and Tools to have explicit coverage, so that agent extensibility regressions are detected before release.
8. As an Akasha maintainer, I want stream, verbose, and keep_logs behavior to be traceable to the feature that owns it, so that cross-cutting behavior is not mistaken for complete feature coverage.
9. As a release maintainer, I want the matrix to distinguish unit, contract, integration, smoke, and upgrade coverage, so that a green unit test does not imply a green provider integration.
10. As a contributor, I want the matrix to show required credentials and external services, so that I know why a test is skipped or how to run it.
11. As a contributor, I want deterministic tests for pure formatting and normalization boundaries, so that most regressions can be checked without provider API calls.
12. As a maintainer, I want live tests to use fixed fixtures and explicit assertions, so that a non-empty provider response is not treated as proof that the feature worked.
13. As a maintainer, I want test output and generated artifacts isolated from versioned feature data, so that running tests does not pollute the repository's test fixtures.
14. As a maintainer, I want every supported feature to have an owner row in the matrix, so that missing coverage is visible rather than inferred from filenames.
15. As a release reviewer, I want the matrix to record the last verified command and result, so that feature readiness is evidence-based.
16. As an Akasha user, I want `ask(info=...)` to accept a file, directory, URL, or a mixture of these sources, so that I can ask questions over the information I already have.
17. As an Akasha maintainer, I want separate tests for each `ask(info=...)` source type, so that a passing URL test does not falsely imply that local files and directories work.
18. As an Akasha maintainer, I want invalid `ask(info=...)` inputs to have explicit behavior, so that missing paths and empty sources do not silently produce misleading answers.
19. As an Akasha maintainer, I want every public RAG parameter represented in the coverage matrix, so that an initialization assertion is not mistaken for proof that the parameter affects runtime behavior.
20. As an Akasha maintainer, I want RAG tests to reuse the existing deterministic documents and directory fixtures, so that retrieval assertions remain stable and the repository does not accumulate duplicate test data.
21. As an Akasha maintainer, I want RAG tests to distinguish source loading, retrieval, prompt construction, model generation, persistence, and observability, so that a failure identifies the affected stage.
22. As an Akasha maintainer, I want RAG provider tests to use a small representative matrix rather than every parameter combination, so that provider validation remains useful without becoming unmaintainable.

## Implementation Decisions

- Use feature-first test directories as the primary navigation model. Keep `unit`, `contract`, `integration`, `smoke`, and `upgrade` as pytest markers and, where useful, as secondary grouping rather than the only taxonomy.
- Store each test under its public feature first and its behavior second. Ask tests use paths such as `ask/stream`, `ask/info/file`, and `ask/info/url`; RAG tests use `rag/input`, `rag/parameters`, `rag/retrieval`, and `rag/chroma`; Agent, MCP, Skill, Tool, Provider, and Observability tests follow the same rule.
- Permit only explicit cross-feature compatibility tests to remain outside a feature directory. The current API stability regression is such an exception because one parametrized module covers Ask, Agent, RAG, Vision, and Memory together.
- Use one functional feature directory per capability. Do not create a top-level directory for every parameter. Stream tests belong under Ask, Agents, or MCP according to the public API they exercise; verbose and keep_logs belong under Observability with feature-specific assertions where needed.
- Treat the public behavior seam as the preferred integration seam: `akasha.ask`, `akasha.agents`, `MemoryManager`, the public Vision callable, public tool creation/invocation, skill loading/execution, and normalized MCP tools. Test internal helpers directly only when they represent deterministic format, validation, or safety boundaries that cannot be observed economically at a higher seam.
- Add a functional coverage matrix in the test documentation. Each row must include feature, behavior, test location, test level, external dependency, command, and status. Status values are `covered`, `partial`, `missing`, `skipped-by-environment`, or `blocked-by-provider`.
- The matrix must explicitly include Ask, Agents, Memory, Vision, RAG, MCP, Skills, Tools, Observability, and Providers. A feature is not considered complete merely because one live regression test exists.
- Split broad API-stability cases into feature-owned tests without changing their public behavior or assertions except where the existing assertion is too weak to establish the stated behavior.
- Treat `ask(info=...)` as a first-class Ask capability. Its public input matrix must include a single local file, a local directory, a URL, multiple sources, mixed file/directory/URL sources, `Path` values, and already-created `Document` values where the public contract supports them.
- Separate deterministic source-loading contract tests from live answer tests. The deterministic seam verifies the loaded `Document` content and source count before model invocation; the live seam verifies that a configured model can answer from the loaded source.
- Use repository fixtures for text, structured data, and directory inputs. Use an opt-in network test for URLs and keep URL assertions limited to stable source-specific signals.
- Define explicit negative cases for an empty `info`, a missing local path, an unsupported source type, and a directory containing unreadable or empty files. The expected behavior must be documented before asserting it.
- Treat the public RAG constructor and call boundary as the primary parameter seam. Do not create a separate test directory for each parameter; represent parameters as rows in the functional matrix and group tests by observable RAG stage.
- Reuse the existing RAG fixtures: the single-fact text fixture for grounded retrieval and answer assertions, the directory fixture for multi-document selection, the empty and Unicode text fixtures for edge cases, and the existing document corpus for format/provider compatibility. New fixtures are only justified when an existing fixture cannot express a stable behavior.
- Cover RAG parameters by behavior rather than by object state alone. A test that checks `rag.temperature == value` is initialization coverage only; contract coverage must also verify forwarding to the model, prompt construction, retrieval selection, output mode, persistence, or log result as applicable.
- The RAG parameter matrix must include `model`, `embeddings`, `chunk_size`, `search_type`, `max_input_tokens`, `max_output_tokens`, `temperature`, `threshold`, `language`, `record_exp`, `system_prompt`, `prompt_format_type`, `keep_logs`, `use_chroma`, `stream`, `verbose`, and `env_file`, plus the call inputs `data_source`, `prompt`, and `history_messages`.
- Test RAG source forms independently: one file, one directory, a list of sources, an existing database object where supported, and a Windows absolute path. Do not infer list or database-object support from the file and directory tests.
- Test retrieval controls with deterministic local seams: `chunk_size` must affect document segmentation, `search_type` must select the documented strategy or callable, and `threshold` must have its documented current or deprecated behavior. Unknown values and boundary values must have an explicit expected error, warning, or fallback.
- Test generation and prompt controls through a fake or captured model boundary before enabling live providers: input/output token limits, temperature, language, system prompt, prompt format type, and history messages must be observed in the request sent to the model or in the resulting contract.
- Test `use_chroma` and reload behavior using temporary or dedicated test storage. The existing staged Chroma test remains the persistence prior art; it must not write production-like paths or rely on an untracked developer database.
- Test `stream`, `verbose`, `keep_logs`, and `record_exp` as observable output/side-effect contracts. Streaming must preserve the documented return/event shape; verbose output must use the documented channel; logs must be serializable and contain the required RAG context; experiment recording must have an explicit enabled/disabled expectation.
- Use pairwise representative combinations for RAG provider smoke tests. At minimum retain the existing OpenAI and Gemini chat-plus-embedding paths and the existing Anthropic upgrade path; add other supported providers only when their RAG support is explicitly in scope. Do not multiply every provider by every parameter in live tests.
- Memory coverage must include deterministic path/format behavior, add/search/show behavior, persistence after re-instantiation, and a provider-backed memory-assisted response. Memory tests must use temporary storage or a dedicated test data root and must not write to production-like `docs/` paths.
- Vision coverage must include deterministic image input and invalid-input contract cases, plus opt-in provider smoke tests. Provider smoke assertions must use a fixture-specific signal when claiming that the model understood the image; a merely non-empty response is insufficient.
- Skills coverage must retain the existing fixture-based tests and expose their capabilities in the matrix: metadata-only loading, deferred instructions, resource access, script execution, persistent REPL, path traversal rejection, and collision handling.
- Tools coverage must include schema creation, sync/async invocation where supported, validation/error propagation, and execution through an Agent. MCP-discovered tools count as a separate MCP capability and must also be represented under Agents when testing Agent integration.
- MCP coverage must retain stdio and Streamable HTTP tests, and must record discovery, input schema, direct invocation, structured result preservation, text fallback, and the async-only stream guard.
- Keep provider and environment markers authoritative. Feature directories must not cause live provider tests to run implicitly; provider tests remain opt-in or skip according to existing environment rules.
- Keep fixtures, reusable test data, path helpers, and coverage utilities outside feature test directories. Feature tests may reference them through the shared path helper.
- Keep Skill tests under `tests/skill/` and Tool tests under `tests/tool/`; Agent tests may additionally cover their integration with an Agent, but that does not relocate the owning Skill or Tool contract.
- Add a test-map document section with commands such as `pytest tests/ask -q`, `pytest tests/memory -q`, `pytest tests/mcp -q`, and marker-based commands such as `pytest -m unit`.

## Testing Decisions

- Tests must assert externally observable behavior: return values, event shapes, persisted records, tool calls, errors, files created in temporary roots, and JSON-serializable logs. Avoid asserting private implementation details or provider-specific wording unless the fixture requires a stable signal.
- Unit tests cover deterministic helpers and safety boundaries without network or provider credentials.
- Contract tests cover public API schemas, event contracts, tool schemas, skill exposure, MCP result normalization, and serialization boundaries.
- Integration tests cover complete feature flows with real local runtime components and opt-in provider calls where necessary.
- Smoke tests cover a minimal provider-backed path for each capability that depends on a provider or service.
- Upgrade tests cover dependency compatibility and regressions, but they are not the sole home of a feature's tests.
- RAG tests should be staged at the highest useful seam: reuse the existing input-contract tests for path and database-object handling, the staged pipeline tests for Chroma/retrieval/answer stages, and the public RAG smoke tests for provider-backed file, directory, Windows-path, logs, and grounded-answer contracts.
- Existing RAG test data is the default source of truth for new cases: `tests/data/rag/single_fact.txt` for a stable fact, `tests/data/rag/directory/alpha.txt` and `beta.txt` for directory selection, `tests/data/rag/empty.txt` for empty-input behavior, and `tests/data/rag/unicode_繁中.txt` for encoding/language behavior. The larger files under `tests/data/documents/` remain available for format and ingestion compatibility cases.
- The RAG matrix must distinguish `covered` from `partial`: an existing test that only constructs RAG or asserts stored attributes covers initialization, not end-to-end parameter behavior.
- The first acceptance pass must include a collected-test audit and a matrix audit: every feature row must point to an existing test and every feature marked `covered` must have a passing command or an explicitly recorded environment skip.
- Required focused scenarios include:
  - Ask basic, stream, thinking, file/directory/URL info, mixed info sources, verbose, and keep_logs.
  - Agents basic execution, tool calling, stream events, skills, MCP tools, and final answer handling.
  - Memory add/search/show, persistence reload, path safety, and memory-assisted response.
  - Vision valid image, multiple images, invalid path/format, and provider smoke.
  - MCP stdio, Streamable HTTP, schema/discovery, structured result normalization, and async-only behavior.
  - Skills metadata/deferred loading, resource/script/REPL execution, and path safety.
  - Tools schema, invocation, error handling, and Agent execution.
  - RAG source forms (file, directory, list, database object, Windows absolute path), chunking, retrieval strategies, Chroma reload, grounded answer, token/generation controls, prompt formatting, stream, verbose, keep_logs, record_exp, and environment-file handling.
  - Observability verbose, keep_logs, stream event normalization, file output, and JSON serialization.
- Regression commands must include feature-level tests, marker-level tests, `pytest --collect-only -q`, syntax compilation, and the complete suite. Provider-dependent failures must remain distinguishable from skipped unavailable environments.

### Initial coverage audit

This audit records the current state before the feature-oriented reorganization is implemented:

| Capability | Current evidence | Status |
|---|---|---|
| Ask `info` with URL(s) | Opt-in Gemini smoke test with two URLs | partial |
| Ask `info` with a local file | Loader supports it; no direct public Ask test | missing |
| Ask `info` with a directory | RAG directory tests exist, but no direct public Ask test | missing |
| Ask `info` with mixed sources | No direct test | missing |
| Ask `info` with `Path`/`Document` values | Loader contract exists in implementation, but no Ask-level test | missing |
| Ask `info` invalid/empty inputs | No complete public contract | missing |
| Provider Ask matrix | Opt-in provider smoke covers OpenAI, Azure, Gemini, Anthropic, and Ollama | covered-by-environment |
| Provider RAG/Memory/Vision matrix | Uneven; feature-specific gaps remain | partial |

### RAG parameter and fixture audit

The following audit records reusable prior art and the intended next coverage. It is deliberately separate from the provider matrix because most RAG parameter behavior can be tested without a live model.

| RAG area | Existing reusable test/data | Current status | Required contract |
|---|---|---|---|
| Single-file source and grounded answer | `tests/rag/parameters/test_parameter_contract.py`, `tests/rag/provider/test_rag_contract.py` + `tests/data/rag/single_fact.txt` | covered at deterministic seam; provider smoke opt-in | Load the fact, retrieve the supporting document, and return a grounded response |
| Directory source and document selection | `tests/rag/provider/test_rag_contract.py` + `tests/data/rag/directory/` | partial; opt-in provider smoke | Load the directory and select the relevant document rather than merely returning non-empty text |
| Windows absolute path | `tests/rag/input/test_input_contract.py`, `tests/rag/provider/test_rag_contract.py` | partial | Preserve and accept an absolute path through the public RAG boundary |
| Path and database-object handling | `tests/rag/input/test_input_contract.py`, `tests/rag/parameters/test_parameter_contract.py` | covered at deterministic seam | Add public-call coverage where the supported contract requires it |
| Chroma build and reload | `tests/rag/parameters/test_parameter_contract.py`, `tests/rag/chroma/test_pipeline_stages.py` | deterministic mode covered; live reload opt-in | Reopen the generated store and preserve retrievable documents without production-path writes |
| `chunk_size` | `tests/rag/parameters/test_parameter_contract.py`, `tests/rag/chroma/test_pipeline_stages.py` | forwarding covered; segmentation behavior partial | Prove segmentation changes at the document-processing seam |
| `search_type` and `threshold` | `tests/rag/retrieval/test_retrievers.py`, staged pipeline smoke | partial; helper coverage exists | Verify public RAG forwarding, strategy selection, threshold behavior, and documented deprecated behavior |
| Input/output token limits and temperature | `tests/rag/parameters/test_parameter_contract.py`; RAG provider tests; provider helper tests | deterministic forwarding covered; live provider behavior opt-in | Capture the model request and verify the configured values |
| Language, system prompt, prompt format, history | `tests/rag/parameters/test_parameter_contract.py`, prompt helper tests and staged pipeline setup | RAG call forwarding covered | Verify final prompt structure through the RAG call seam |
| `stream` | `tests/rag/parameters/test_parameter_contract.py`; `tests/agent/stream/` | deterministic RAG chunk contract covered; live provider behavior opt-in | Define and verify RAG stream return/event contract |
| `verbose`, `keep_logs`, `record_exp` | RAG smoke log assertions and logging tests | partial | Verify output channel, log schema/serialization, and experiment-recording behavior |
| `use_chroma` and `env_file` | `tests/rag/parameters/test_parameter_contract.py`, staged Chroma test and provider setup | deterministic forwarding/loader mode covered; live environment opt-in | Verify explicit Chroma mode and environment source without hidden machine state |
| Empty and Unicode inputs | `tests/data/rag/empty.txt`, `tests/data/rag/unicode_繁中.txt` | missing as public RAG cases | Define and test empty-source and encoding/language behavior |
| Multiple source list | existing path loader seam, no public RAG case | missing | Verify deterministic ordering, source count, and retrieval across mixed documents |

## Out of Scope

- Changing the public Ask, Agent, Memory, Vision, RAG, MCP, Skill, or Tool APIs solely for test organization.
- Adding a new provider matrix beyond the minimum existing supported providers.
- Treating model answer quality or semantic judgment as a deterministic unit-test concern.
- Replacing existing MCP protocol implementation work or expanding the MVP to MCP resources/prompts unless separately specified.
- Removing provider smoke tests, changing their opt-in behavior, or hiding external-service failures as passes.
- Renaming fixture contents without a functional reason.

## Further Notes

- The current suite already has useful Skill and MCP seams; the classification work should index and reorganize them rather than duplicate them.
- Current Memory and Vision live tests should be considered partial coverage until they have dedicated feature entries and deterministic contract cases.
- The matrix should be updated whenever a new public capability or test layer is added. A feature directory without a matrix row is considered incomplete documentation.
- The issue-tracker publishing step from the local `$to-spec` skill could not be performed because no issue-tracker connector or configured tracker vocabulary is available in this environment; this file is the project-local spec artifact.
