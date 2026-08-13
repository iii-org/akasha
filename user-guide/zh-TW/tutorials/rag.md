# 建立 RAG 流程

RAG 會先擷取相關文件內容，再請模型回答。akasha 分別使用聊天模型與 Embedding 模型處理這兩個步驟。

## 完整範例

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
)

answer = rag(
    "./docs",
    "這些文件的主要內容是什麼？",
)
print(answer)
```

在這段程式中：

- `model` 負責產生回答。
- `embeddings` 將文件文字與問題轉成向量。
- `./docs` 是文件來源。

## 常見問題

- 文件路徑必須存在，且包含支援的檔案。
- Embedding Provider 需要獨立設定，不一定與聊天模型相同。
- 第一次執行可能較久，因為需要準備文件與 Embedding。

如果模型需要呼叫工具，而不只是搜尋文件，請接著閱讀 [Agent](agents.md)。
