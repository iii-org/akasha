# 設定模型 Provider

akasha 使用模型別名選擇 Provider。請將憑證放在 shell 環境變數或本機 `.env` 檔案中，絕對不要提交 API key。

```env
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

## Gemini 後端模式

使用 Gemini Developer API 時，只需設定 API key：

```env
GEMINI_API_KEY=your_key
```

若要以 API key 使用 Vertex AI **Express Mode**，使用同一把 key 並開啟 Vertex selector：

```env
GEMINI_API_KEY=your_key
GOOGLE_GENAI_USE_VERTEXAI=true
```

Express Mode 不要設定 `GOOGLE_CLOUD_PROJECT` 或 `GOOGLE_CLOUD_LOCATION`。
這兩個變數設定的是完整 Vertex AI 的 project/location 認證路徑；該路徑需要
Application Default Credentials (ADC)，且會優先於 API key。Akasha 的 Vertex
API-key 支援是 Express Mode，不會設定完整的 ADC 路徑。

模型別名範例：

```text
gemini:gemini-2.5-flash
openai:gpt-4o
anthropic:claude-3-5-sonnet-latest
ollama:qwen3:8b
```

`:` 前面的 Provider 名稱決定整合方式，後面的模型名稱決定要使用的模型。

## 安全地檢查設定

不要把 key 印出來，只檢查環境變數是否存在：

```python
import os

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("請先設定 GEMINI_API_KEY")

print("Provider 設定存在")
```

!!! warning
    有 API key 不代表模型一定可用，也可能產生費用。執行即時範例前，請先確認 Provider 的模型權限與計費設定。

下一步：[執行第一個對話](first-chat.md)。
