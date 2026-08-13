# MCP Integration Specification

## Problem Statement

Akasha 目前可以作為 MCP client，透過 `langchain-mcp-adapters` discovery MCP tools，並把 tools 傳給 `akasha.agents()`。本地 stdio MCP server 已有 deterministic smoke test，且 MCP tools 的 async-only 限制已有明確的 stream guard。

但目前的 MCP 整合仍停留在「取得 tools 後交給 agent」的最小路徑：

- 遠端 MCP 範例仍以舊的 HTTP+SSE transport 為主，沒有以 Streamable HTTP 為標準路徑。
- 沒有驗證 protocol version、capability negotiation、HTTP session header、重連與 request timeout。
- tool metadata、pagination、list-changed、structured result、resource link 與 input-required result 尚未形成 Akasha 的明確 contract。
- 多個 MCP server 的工具名稱可能碰撞。
- 遠端 authorization、Origin 驗證、token audience 與使用者確認尚未形成安全邊界。
- 文件與範例未完整反映目前公開 API；範例中的舊式 MCP 呼叫流程不可作為新的整合契約。

這會使 Akasha 在接上較新的 MCP server 時，可能 discovery 成功但在 transport、結果正規化、session、授權或 agent streaming 階段失敗，而且錯誤訊息不足以判斷責任邊界。

## Solution

建立一個明確的 MCP client integration contract，讓 Akasha：

1. 保持現有 stdio MCP tools 的相容性。
2. 將 Streamable HTTP 定為新的遠端 MCP transport；舊 HTTP+SSE 僅保留為 adapter 相容性問題，不在新文件與新範例中主推。
3. 在 MCP server discovery、tool normalization、agent invocation、events/logs/errors 之間建立可測試的單一整合 seam。
4. 將 MCP tool metadata、結果與錯誤轉換成穩定的 Akasha agent contract，而不把 adapter 的內部物件形狀直接暴露給使用者。
5. 對 async-only MCP tools 維持明確行為：non-stream agent 使用 async invocation；不支援 async-only tools 的同步 streaming 必須在執行前拒絕並給出可行修正。
6. 對遠端 server 加入 transport-level security policy 與使用者可見的 tool invocation boundary。

## User Stories

1. As an Akasha application developer, I want to connect to a local stdio MCP server, so that I can use local tools without an external network dependency.
2. As an Akasha application developer, I want to connect to a Streamable HTTP MCP server through one MCP endpoint, so that remote MCP integrations use the current standard transport.
3. As an Akasha application developer, I want transport configuration to be explicit, so that stdio and Streamable HTTP do not depend on ambiguous legacy `sse` settings.
4. As an Akasha application developer, I want MCP protocol and adapter compatibility to be checked independently, so that a package upgrade does not silently claim protocol compatibility.
5. As an Akasha application developer, I want server capability negotiation to be respected, so that Akasha does not call features the server did not advertise.
6. As an Akasha application developer, I want tool discovery to handle pagination, so that servers with many tools are not truncated.
7. As an Akasha application developer, I want tool-list changes to be observable, so that newly available or removed tools do not remain stale in the agent context.
8. As an Akasha application developer, I want tool names from multiple servers to be namespaced or deterministically disambiguated, so that one server cannot overwrite another server's tool.
9. As an Akasha application developer, I want tool titles, descriptions, input schemas and output schemas to be preserved, so that the model receives enough information to call tools safely and correctly.
10. As an Akasha application developer, I want malformed tool schemas to produce an isolated diagnostic, so that one invalid tool does not make every valid MCP tool unavailable.
11. As an Akasha application developer, I want text, image, audio, embedded-resource and resource-link results to be normalized, so that MCP results can be recorded and consumed without adapter-specific assumptions.
12. As an Akasha application developer, I want structured tool results and output schemas to be preserved, so that downstream code does not have to parse structured data from text.
13. As an Akasha application developer, I want tool errors and protocol errors to be distinguishable, so that I can decide whether to retry, fix arguments, or repair the server connection.
14. As an Akasha application developer, I want a server that requires additional user input to be represented explicitly, so that the agent does not treat an incomplete tool result as a final answer.
15. As an Akasha application developer, I want request timeouts and cancellation to be configured, so that a broken MCP server cannot hang an agent indefinitely.
16. As an Akasha application developer, I want Streamable HTTP sessions to handle session identifiers and protocol-version headers, so that stateful servers remain interoperable.
17. As an Akasha application developer, I want disconnected HTTP streams to be handled separately from explicit cancellation, so that transient network failures are not misreported as user cancellation.
18. As an Akasha application developer, I want remote MCP authorization to use the server's declared authorization flow, so that tokens are not sent to the wrong server or exposed in URLs.
19. As an Akasha application developer, I want local HTTP MCP servers to be protected against Origin and DNS-rebinding risks, so that a browser-originated request cannot invoke local tools unexpectedly.
20. As an Akasha application user, I want to see which MCP server and tool are being invoked, so that I understand where data is going and what action is being taken.
21. As an Akasha application user, I want destructive or externally visible tool calls to be confirmable, so that the model cannot perform high-impact actions without my consent.
22. As an Akasha application developer, I want sensitive tool parameters excluded from logs and transport-derived headers, so that credentials, tokens and private data are not leaked.
23. As an Akasha application developer, I want MCP tools to work with non-stream agents through the existing async path, so that current async-only server tools remain usable.
24. As an Akasha application developer, I want synchronous streaming with async-only tools to fail before model execution, so that the failure is deterministic and actionable.
25. As an Akasha application developer, I want MCP tool calls and results to remain JSON-serializable in logs, so that traces can be persisted and inspected.
26. As an Akasha maintainer, I want deterministic local MCP fixtures for protocol contracts, so that CI does not depend on external MCP providers or private credentials.
27. As an Akasha maintainer, I want the documentation to show the supported public API, so that users do not copy obsolete MCP helper methods or deprecated transport examples.
28. As an Akasha maintainer, I want live remote MCP tests to be opt-in, so that ordinary CI remains stable while remote transport coverage can run in a dedicated integration job.

## Implementation Decisions

- Akasha remains an MCP host-side client integration, not a general-purpose MCP server framework. Implementing server-side resources, prompts or tool hosting is not part of this change.
- The public integration boundary is the existing `agents(tools=...)` contract. MCP connection setup and discovery remain outside the model provider adapters.
- The primary supported transports are stdio and Streamable HTTP. New documentation and examples must not introduce HTTP+SSE as the preferred transport.
- The adapter layer must expose a transport-neutral internal representation for discovered tools. At minimum it preserves server identity, original tool name, display metadata, input schema, optional output schema, annotations, and the callable async operation.
- Aggregated tools must have deterministic names. The original server and tool names must remain available for diagnostics and logs even when the model-facing name is prefixed or otherwise disambiguated.
- Discovery must respect pagination and must not assume that one `tools/list` response contains the complete set.
- Tool-list cache invalidation must be tied to the server's list-change capability/notification when available. If notifications are unavailable, the implementation may use an explicit refresh boundary rather than polling implicitly.
- Tool results must retain structured content and content item types. Text extraction is allowed for model compatibility, but it must not destroy structured content, resource links, embedded resources, image data or audio data in the normalized result/log representation.
- MCP protocol errors, transport errors, tool execution errors and model/tool-selection errors must remain distinguishable in diagnostics and persisted logs.
- `agents(stream=False)` remains the supported path for async-only MCP tools. Synchronous streaming must continue to reject an agent containing async-only tools before invoking the model, with an error that names the required async/non-stream usage.
- Request timeout, cancellation and connection shutdown behavior must be configurable at the MCP client boundary. A progress notification may extend an operation's working timeout, but a maximum timeout must always apply.
- Streamable HTTP clients must carry the negotiated protocol version and session identifier according to the transport contract, handle session expiration by reinitializing, and distinguish a dropped response stream from explicit cancellation.
- HTTP authorization must use the MCP server as the resource audience. Access tokens must be sent in authorization headers, never query parameters, and must not be forwarded blindly to downstream services.
- Local HTTP MCP fixtures must bind to localhost and validate Origin. Remote deployment guidance must require authentication and TLS where applicable.
- Tool annotations are advisory metadata and must not be treated as a security guarantee. User confirmation and policy decisions must be based on trusted application configuration and the actual operation.
- The agent event/log contract must include the MCP server identity, model-facing tool name, original tool name, call arguments subject to redaction, outcome, elapsed time and an error classification.
- The dependency policy must document the supported `mcp` and `langchain-mcp-adapters` ranges separately from the supported MCP protocol revisions. Upgrading either dependency requires the focused MCP contract tests.
- Documentation and examples must use the currently supported public API and must remove obsolete helper-method assumptions. The example must show direct construction of an MCP client, discovery, passing tools to `agents`, and the correct async/non-stream behavior.
- The implementation should be delivered in vertical slices: first stabilize stdio and result/error contracts, then add Streamable HTTP, then add authorization and user-confirmation policy. Each slice must keep the deterministic stdio path green.

## Testing Decisions

- Tests assert externally observable behavior: discovered tool names and metadata, callable behavior, agent response, tool-call trace, event shape, logs, errors, timeout/cancellation behavior and transport headers. Tests should not assert adapter-private classes or internal method call order.
- The highest test seam is one deterministic MCP pipeline: server discovery → tool contract normalization → agent tool binding → tool invocation → response/event/log/error normalization.
- Existing deterministic stdio coverage remains the baseline and must continue to verify discovery, input schema, direct async invocation, real agent non-stream invocation, JSON-safe logs and the async-only stream guard.
- Add a local Streamable HTTP fixture that supports one MCP endpoint and exercises initialization, negotiated protocol version, session identifier, JSON response, SSE response, session expiration and explicit cancellation.
- Add contract cases for paginated tool discovery, stable ordering, tool-list change notification, duplicate names across servers, invalid schemas and server-unavailable diagnostics.
- Add result fixtures for text, image, audio, resource link, embedded resource, structured content, output schema and input-required results. Verify that logs remain JSON-serializable without collapsing structured data into an unbounded text string.
- Add security-focused tests for Origin rejection, localhost binding guidance, authorization header usage, query-string token rejection, resource audience handling and sensitive-argument redaction.
- Add timeout and cancellation tests using a deterministic delayed server. Verify both protocol cancellation and transport shutdown, including that a disconnected stream is not mislabeled as explicit cancellation.
- Keep live remote MCP tests opt-in and separate from basic CI. They may validate interoperability with a real Streamable HTTP server, but they must not be the only proof of the local contract.
- Existing provider smoke tests may be reused for model tool selection, but MCP protocol tests must remain independently runnable without provider credentials.
- A passing discovery-only test is insufficient for completion. Acceptance requires at least one end-to-end agent invocation and the complete deterministic contract suite.

## Out of Scope

- Building Akasha as an MCP server.
- Adding first-class resources, prompts, sampling, roots, elicitation, Tasks, Skills over MCP or MCP Apps to the public Akasha API.
- Supporting deprecated HTTP+SSE as a new feature or documenting it as the preferred remote transport.
- Implementing an MCP registry, marketplace, server installation manager or automatic trust system.
- Automatically approving destructive tools or treating MCP annotations as proof that a tool is safe.
- Making external MCP services, private databases, provider accounts or live network dependencies mandatory for ordinary CI.
- Reworking all LangChain tool abstractions unrelated to MCP.
- Changing the existing non-MCP agent event contract except where an MCP-specific field is required for provenance or error classification.

## Further Notes

- Current implementation evidence: local stdio discovery/direct invocation passes; the repository already has an explicit async-only MCP stream guard.
- The installed dependency versions are implementation evidence only, not a protocol compatibility guarantee. The compatibility matrix should be recorded when the first Streamable HTTP slice is implemented.
- The latest MCP specification marks HTTP+SSE deprecated and defines Streamable HTTP as the current remote transport. It also introduces stricter expectations around metadata, structured tool results, session handling and security.
- The spec intentionally keeps the first delivery centered on tools because that is the existing Akasha integration surface. Resources/prompts and server-initiated client features can be evaluated as separate specs after the tool contract is stable.
- Proposed acceptance gate: deterministic stdio suite green, deterministic Streamable HTTP suite green, focused unit tests green, documentation/example validation green, and no new MCP live test required for ordinary CI.
