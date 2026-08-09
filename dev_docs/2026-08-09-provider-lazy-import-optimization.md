# Provider Lazy Import 優化紀錄

Date: 2026-08-09

## 目的

降低 `ask()` 與 `agents()` 初始化時不必要的 provider SDK 載入，讓使用者選擇哪一種模型時，才載入對應的 LangChain provider adapter。

## 問題

`akasha.utils.models.chat.build_chat_model()` 原本已經在各 provider 分支內使用 lazy import，但下列 helper 在模組頂層直接載入 OpenAI embedding adapter：

- `akasha/helper/handle_objects.py`
- `akasha/helper/base.py`

因此即使使用 Gemini、Anthropic 或 Ollama，也可能先載入 `langchain_openai` 與 OpenAI SDK。

## 變更

- 將 `OpenAIEmbeddings` 從 `handle_objects.py` 的頂層 import 移至實際建立 OpenAI／Azure embedding 的分支。
- 將 `OpenAIEmbeddings` 與 `AzureOpenAIEmbeddings` 從 `base.py` 的頂層 import 移至 `decide_embedding_type()`。
- 保留 `build_chat_model()` 現有的 provider-specific lazy import：
  - OpenAI／Azure：`langchain_openai`
  - Gemini：`langchain_google_genai`
  - Anthropic：`langchain_anthropic`
  - Ollama：`langchain_ollama`
- 新增 provider import boundary regression tests，確認未選用的 provider 不會被載入。

## 驗證結果

在 Python 3.11 的 light 隔離環境中驗證：

- `import akasha.agent.agents` 後，`langchain_openai` 與 `langchain_google_genai` 都未載入。
- 建立 Gemini ChatModel 後，只載入 `langchain_google_genai`，不載入 `langchain_openai`。
- Helper import 約由 2.7 秒降至約 1.2–1.6 秒。
- 真實 Gemini chat 呼叫成功，回傳非空 `str`。
- 真實 Gemini agent 呼叫成功，回傳非空 `str`。

## 測試

Focused tests:

```text
21 passed
3 passed, 2 skipped
```

既有真實 Gemini agent smoke tests:

```text
3 passed
```

測試使用 `tests/.env` 中既有的 provider 設定；沒有輸出或保存 API key，也沒有保存模型回應內容。

## 效能限制

選定 provider 的首次初始化仍需載入該 provider SDK。例如 OpenAI SDK 本身的 import 成本仍存在；lazy import 的主要收益是避免載入未使用的 provider，而不是消除選定 provider 的首次初始化成本。

同一程序中應重用已建立的 `ask()`／`agents()` instance，避免每個問題重新建立 model client。
