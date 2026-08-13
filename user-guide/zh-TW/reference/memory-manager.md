# `MemoryManager`

`MemoryManager` 會儲存對話中的重要資訊，並透過語意檢索在之後搜尋。

## 建立記憶管理器

```python
import akasha

memory = akasha.MemoryManager(
    memory_name="assistant",
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    memory_dirname="docs",
)
```

記憶檔案會儲存在 `memory_dirname / memory_name` 底下。

## 新增與搜尋記憶

```python
memory.add_memory(
    user_prompt="我偏好簡短的技術說明。",
    ai_response="之後我會保持說明簡潔。",
    language="ch",
)

matches = memory.search_memory("我的說明偏好是什麼？", top_k=3)
for item in matches:
    print(item)
```

常用方法：

| 方法 | 用途 |
| --- | --- |
| `add_memory(user_prompt, ai_response, language="ch")` | 從一輪對話擷取並儲存重要資訊。 |
| `search_memory(query, top_k=3)` | 回傳與問題相關的記憶。 |
| `show_memory(num=100)` | 回傳記憶內容供檢查。 |

記憶功能會建立或更新本機檔案與向量資料庫。請謹慎選擇記憶目錄，並避免把私人記憶資料提交到公開 repository。
