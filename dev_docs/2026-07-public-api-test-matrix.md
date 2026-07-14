# akasha-repo Public API 測試矩陣表

Date: 2026-07-14

## 目的

這份矩陣表用來定義 public API 應該被哪些測試覆蓋，重點不是全排列，而是：

- 先保住主要使用路徑
- 先保住高風險參數
- 先保住 provider / platform 差異
- 讓 mock 測試與真實 smoke 測試分工明確

## 測試層級

| 層級 | 目的 | 是否碰真模型 |
|---|---|---|
| Unit | 驗證純邏輯、參數轉換、錯誤處理 | 否 |
| Contract | 驗證 public API shape、logs、stream/event 格式 | 視需要 |
| Smoke | 驗證真實 provider / 真實模型路徑 | 是 |

## 1. `akasha.ask()`

| 測項 | 建議組合 | 目的 | 類型 |
|---|---|---|---|
| 最小問答 | `stream=False`, `thinking=False` | 確認回傳 `str` 與基本可用性 | Smoke |
| 串流輸出 | `stream=True`, `thinking=False` | 確認 chunk event / 字串串流行為 | Smoke / Contract |
| thinking 啟用 | `stream=True`, `thinking=True` | 確認 `thinking` 與 `answer` 分離 | Smoke |
| thinking 停用 | `thinking=False`, `thinking_budget` 有值 | 確認 budget 不會誤傷非 thinking 模式 | Unit / Contract |
| logs | `keep_logs=True` | 確認 logs 有寫入且可序列化 | Contract |
| token 上限 | `max_output_tokens=small` | 確認上限真的生效 | Smoke |
| env 指定 | `env_file=<path>` | 確認讀 env 行為一致 | Smoke / Contract |
| provider alias | `openai:` / `gemini:` / `anthropic:` / `ollama:` | 確認 alias 與 factory 相容 | Smoke |

### 建議最小覆蓋

- 1 條 `gemini` smoke
- 1 條 `openai` smoke
- 1 條 `thinking=True` smoke
- 1 條 `keep_logs=True` contract

## 2. `akasha.agents()`

| 測項 | 建議組合 | 目的 | 類型 |
|---|---|---|---|
| 非串流 final answer | `stream=False` | 確認回傳仍是 `str` | Smoke |
| 串流 events | `stream=True` | 確認事件格式穩定 | Smoke |
| tool calling | 有 tool | 確認原生 tool calling 正常 | Smoke |
| thinking | `thinking=True` | 確認 thinking event 不混入 answer | Smoke |
| logs | `keep_logs=True` | 確認 messages / tool calls 可保存 | Contract |
| multi-tool | 兩個以上 tools | 確認多工具呼叫不壞 | Smoke / Contract |
| provider compatibility | `openai:` / `gemini:` / `anthropic:` | 確認不同 provider 可進入同一路徑 | Smoke |

### 建議最小覆蓋

- 1 條 `gemini` tool-calling smoke
- 1 條 `thinking=True` streaming smoke
- 1 條 `keep_logs=True` contract

## 3. `akasha.RAG()`

| 測項 | 建議組合 | 目的 | 類型 |
|---|---|---|---|
| file input | 單一文件 | 確認最基本檢索可用 | Smoke |
| directory input | 資料夾 | 確認批次文件載入正常 | Smoke |
| Windows absolute path | `C:\...` | 確認路徑 sanitize 與 Chroma path 正常 | Regression |
| provider alias | `openai:` / `gemini:` | 確認不同 provider embedding / model 路徑 | Smoke |
| embeddings alias | `openai:text-embedding-3-small` / `gemini:gemini-embedding-001` | 確認 embeddings factory 正常 | Smoke |
| search_type | `auto` / `merge` / `rerank` | 確認檢索策略切換正常 | Contract / Smoke |
| env 指定 | `env_file=<path>` | 確認 provider key 來源一致 | Contract |

### 建議最小覆蓋

- 1 條 file smoke
- 1 條 directory smoke
- 1 條 Windows absolute path regression

## 4. `akasha.summary()`

| 測項 | 建議組合 | 目的 | 類型 |
|---|---|---|---|
| text input | 純文字 | 確認 summary 基本流程 | Smoke |
| file input | `.txt` / `.md` / `.pdf` | 確認檔案讀取與摘要 | Smoke |
| URL input | 網址 | 確認 web summary 路徑 | Smoke / Contract |
| `map_reduce` | 預設路徑之一 | 確認多 chunk 合併 | Smoke |
| `refine` | 另一條路徑 | 確認逐步 refine | Smoke |
| `keep_logs=True` | 如支援 | 確認日誌行為 | Contract |

### 建議最小覆蓋

- 1 條 text smoke
- 1 條 file smoke
- 1 條 URL smoke

## 5. `MemoryManager`

| 測項 | 建議組合 | 目的 | 類型 |
|---|---|---|---|
| add_memory | 連續寫入 2 筆以上 | 確認儲存正常 | Smoke |
| search_memory | 查詢先前寫入內容 | 確認檢索正常 | Smoke |
| persistence | 重啟後仍可取回 | 確認磁碟持久化 | Smoke |
| Windows absolute path | `memory_dirname=C:\...` | 確認目錄建立安全 | Regression |
| provider alias | `openai:` / `gemini:` | 確認模型與 embedding 可共存 | Smoke |

### 建議最小覆蓋

- 1 條 persistence smoke
- 1 條 Windows regression

## 6. `akasha.agent.*` / 內部但對外可感知的行為

| 測項 | 建議組合 | 目的 | 類型 |
|---|---|---|---|
| message normalization | legacy / native message shape | 確認格式轉換沒有壞 | Unit |
| reasoning separation | reasoning 與 answer 分離 | 確認 answer 不被污染 | Unit / Contract |
| tool schema | BaseTool schema | 確認 tool calling 能取到正確參數 | Unit / Contract |
| stream events | answer / tool / thinking | 確認事件鍵穩定 | Contract |

## 7. 共通高風險參數

這些參數在各 API 都應該至少出現在一個測試裡：

| 參數 | 風險 | 建議覆蓋 |
|---|---|---|
| `stream` | 會改回傳型態與事件格式 | ask / agents |
| `thinking` | 會改 event 與 logs | ask / agents |
| `keep_logs` | 會改 side effect | ask / agents / summary |
| `max_output_tokens` | 會影響 provider 行為 | ask / agents / summary |
| `env_file` | 會改 key 來源 | ask / RAG / agents / MemoryManager |
| provider alias | 換模型即可能炸 | 所有 public API |
| Windows absolute path | OS 差異 | RAG / MemoryManager |

## 8. 測試優先順序

### P0

1. `ask()` 最小 smoke + thinking smoke
2. `agents()` streaming / tool calling smoke
3. `RAG()` file / directory / Windows regression
4. `MemoryManager` persistence smoke

### P1

1. `summary()` smoke
2. provider alias matrix
3. logs / event schema contract tests

### P2

1. 細部 helper unit tests
2. 低風險 parameter edge cases
3. 歷史 bug regression 擴充

## 9. 驗收條件

當以下條件達成時，代表 public API 矩陣已具備基本防護：

- public API 至少每個都有一條真實 smoke
- stream / thinking / logs / env / Windows 路徑都被覆蓋
- provider 換版本時能在 CI 早期暴露 breakage
- mock 測試只負責純邏輯，不再假扮整體可用性驗證

