# 選擇 akasha API

請先使用符合需求的最小 API。

| 你的需求 | 建議入口 |
| --- | --- |
| 詢問圖片內容 | `asker.vision()` |
| 生成新圖片 | `akasha.gen_image()` |
| 編輯既有圖片 | `akasha.edit_image()` |
| 直接向模型提問 | `akasha.ask()` |
| 詢問本機文件內容 | `akasha.RAG()` |
| 讓模型呼叫工具 | `akasha.agents()` |
| 摘要檔案或 URL | `akasha.summary()` |
| 在不同對話之間保留資訊 | `MemoryManager` |

## 同一個問題，在有無檢索時的差異

```python
import akasha

# 使用模型的一般知識
qa = akasha.ask(model="gemini:gemini-2.5-flash")
print(qa("什麼是向量資料庫？"))

# 使用自己的文件作為背景資料
rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
)
print(rag("./docs", "我們的文件如何描述向量資料庫？"))
```

如果答案依賴你的檔案，建議使用 RAG，而不是自行把整份檔案塞進 Prompt。
