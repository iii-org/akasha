# Gemini Developer API 與 Vertex AI API Key 共存規格

## Problem Statement

Akasha 的 Gemini ChatModel 目前只支援以 `GEMINI_API_KEY` 呼叫 Gemini
Developer API。使用者若持有可用於 Vertex AI Express Mode 的 API key，無法在
保留既有 Gemini Developer API 設定的同時，明確選擇 API-key Vertex 後端。

使用者需要以同一個 Gemini provider alias 與同一個 `GEMINI_API_KEY` 設定名稱
支援兩種情境：預設的 Gemini Developer API，以及明確啟用的 Vertex AI API-key
模式。本次不以 Application Default Credentials (ADC) 或 service account 為目標。

## Solution

Gemini ChatModel factory 保持為唯一的設定 seam。呼叫端繼續使用
`gemini:<model>`，並使用同一個 `GEMINI_API_KEY`；factory 根據
`GOOGLE_GENAI_USE_VERTEXAI` 決定後端：

- 未設定或為 false 時，維持既有 Gemini Developer API 行為。
- 設為 true 時，使用 Vertex AI Express Mode，只以 API key 驗證。

Express Mode 不使用 `GOOGLE_CLOUD_PROJECT` 或 `GOOGLE_CLOUD_LOCATION`；兩者
屬於完整 Vertex AI 的 ADC 認證模式，與 API-key Express Mode 互斥。factory 在
建構 Google adapter 的短暫區段隔離這兩個環境變數，確保呼叫端即使已載入完整
Google Cloud 設定，也不會意外回退至 ADC。

## User Stories

1. As an Akasha user with an existing Gemini Developer API key, I want my existing `GEMINI_API_KEY` configuration to continue working without new variables, so that upgrades do not break my current workloads.
2. As an Akasha user with a Vertex AI-compatible API key, I want to enable Vertex AI with an explicit environment variable, so that I can select the intended Google backend without changing my model alias.
3. As an Akasha user, I want to use `gemini:<model>` for both Google backends, so that provider selection remains stable across deployment environments.
4. As an Akasha user, I want to keep the key in `GEMINI_API_KEY` in both modes, so that I do not need duplicate key names or provider-specific application code.
5. As an Akasha user enabling Vertex AI Express Mode, I want no project, location, or ADC configuration to be required, so that I can use my API key alone.
6. As an Akasha user, I want ambient Google Cloud project and location variables to be ignored for Express Mode, so that a previously loaded `.env` cannot silently switch my request to ADC.
7. As an Akasha user, I want false or absent Vertex selection to remain on Gemini Developer API even if unrelated Google Cloud variables exist, so that backend choice is never inferred accidentally.
9. As an Akasha user, I want configuration errors to name missing variable names but never print secret values, so that diagnostics are safe in terminals and logs.
10. As an Akasha library caller passing an environment mapping, I want the factory to make the same backend decision as the CLI and environment-file paths, so that programmatic and command-line use are consistent.
11. As an Akasha maintainer, I want all Google chat backend selection contained in one factory seam, so that future credential modes do not require duplicated branching across callers.
12. As an Akasha maintainer, I want focused tests for both modes and their validation failures, so that a future adapter upgrade cannot silently route a request to the wrong Google backend.
13. As an Akasha user with an invalid or unauthorized Vertex API key, I want the provider's authentication response to remain visible as a runtime failure, so that Akasha does not falsely claim it can validate cloud authorization locally.

## Implementation Decisions

- The public model-selection interface remains unchanged. `gemini`, `google`, and `gemi` continue to identify the same Gemini provider family; no separate `vertex` model alias is introduced.
- `GEMINI_API_KEY` remains required in both supported modes. The implementation must use the adapter's current `api_key` interface rather than relying on a legacy argument spelling.
- `GOOGLE_GENAI_USE_VERTEXAI` is the sole backend selector. Its accepted truthy values are `true`, `1`, and `yes`, evaluated case-insensitively. Any other value, including absence, selects Gemini Developer API.
- Vertex API-key mode is Vertex AI Express Mode. It requires no project or location, and must not forward either setting to the Google chat adapter.
- The factory explicitly provides the resolved backend choice to the Google chat adapter and uses an empty location to suppress the adapter's Vertex location default.
- During Express Mode adapter construction, the factory serializes and temporarily removes ambient `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`, then restores their exact prior values. This prevents the SDK from preferring ADC configuration over the API key.
- The existing temperature, output-token, and thinking configuration must be preserved unchanged in both modes.
- The environment-loading interface retains the Vertex selector but does not load project or location for this API-key-only mode.
- Availability checks that currently treat a non-empty `GEMINI_API_KEY` as Gemini availability remain valid for this API-key-only scope.
- The factory is the high-level seam for backend-resolution behavior. Callers should not construct provider-specific Google adapter arguments themselves.

## Testing Decisions

- Tests target externally observable factory behavior: the adapter receives Developer API arguments by default; it receives Express Mode selection without project or location when Vertex mode is explicitly enabled.
- A Developer API regression test passes only `GEMINI_API_KEY` and verifies that construction succeeds without additional Google Cloud configuration.
- A Vertex Express Mode test provides the key and selector while project/location are present both in the input mapping and process environment. It verifies that the adapter is constructed without those settings and that the process environment is restored afterward.
- A Vertex Express Mode test omits project and verifies model construction succeeds.
- Tests cover accepted case-insensitive truthy selector values and at least one false value, ensuring the setting cannot accidentally enable Vertex AI.
- Environment-loading tests verify that the selector is retained and project/location stay outside this mode's configuration.
- Tests use the existing chat-model factory and environment-loading seams, with the Google adapter replaced by a test double. They must not require live keys, a GCP project, ADC, or network access.
- Existing provider factory tests are the prior art; this change adds focused Google-mode coverage rather than broad end-to-end agent tests.

## Out of Scope

- Full Vertex AI project/location operation with ADC, service-account credentials, workload identity, and `GOOGLE_APPLICATION_CREDENTIALS` support.
- Validating whether an API key is provisioned, authorized, restricted correctly, or billed for Vertex AI; Google performs that validation at request time.
- Adding a second key name, automatic migration of secrets, or exposing key values in diagnostics.
- Changing Gemini embeddings, token counting, image generation, or other Google client paths outside the LangChain chat-model factory.
- Changing model names, thinking semantics, streaming behavior, API response handling, or non-Google providers.
- Live integration tests against Gemini Developer API or Vertex AI as part of this implementation.

## Further Notes

- Example Developer API configuration:

  ```env
  GEMINI_API_KEY=...
  ```

- Example Vertex AI API-key configuration:

  ```env
  GEMINI_API_KEY=...
  GOOGLE_GENAI_USE_VERTEXAI=true
  ```

- The two examples are alternative deployment configurations. A deployment that
  sets the Vertex selector uses Vertex AI Express Mode; one that omits it
  preserves the current Gemini Developer API path. Do not add
  `GOOGLE_CLOUD_PROJECT` or `GOOGLE_CLOUD_LOCATION` to the Express Mode
  configuration.
- Vertex API-key support depends on the installed `langchain-google-genai`
  version and Google Cloud key permissions. This specification assumes the
  project's declared 4.x integration line, but implementation should verify
  the exact resolved package behavior before release.
