# API 參考

本區說明 akasha 的公開 Python 入口。請先看 API 總覽，再進入需要的功能頁面。

## 公開入口

| API | 用途 | 詳細參考 |
| --- | --- | --- |
| `asker.vision()` | 詢問圖片內容並取得文字回答。 | [`圖片理解與圖片編輯`](vision.md) |
| `akasha.gen_image()` | 根據文字描述生成新圖片。 | [`圖片理解與圖片編輯`](vision.md) |
| `akasha.edit_image()` | 移除、增加或修改既有圖片內容。 | [`圖片理解與圖片編輯`](vision.md) |
| `akasha.ask()` | 聊天與問答，可選擇加入背景資料。 | [`ask`](ask.md) |
| `akasha.RAG()` | 載入、檢索文件並回答問題。 | [`RAG`](rag.md) |
| `akasha.agents()` | 執行可使用 Tools、Skills 或 MCP 工具的 Agent。 | [`agents`](agents.md) |
| `akasha.summary()` | 摘要文字、檔案或 URL。 | [`summary`](summary.md) |
| `akasha.MemoryManager` | 儲存與搜尋長期語意記憶。 | [`MemoryManager`](memory-manager.md) |

## 如何閱讀簽名

範例會展示最常使用的選項，不會列出所有進階參數。每個 API 還支援 Token 限制、日誌、Prompt 與 Provider 相關設定；需要特殊選項時，再對照實際原始碼簽名。

!!! warning
    API key、模型權限與回答內容會依 Provider 而不同。文件範例不包含真實憑證，也不保證每次產生完全相同的文字。
