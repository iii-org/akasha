# `myTokenizer`

`myTokenizer` 是用來依照模型 tokenizer 計算 Token 數量的 Helper API。

## `compute_tokens()`

```python
import akasha.helper as ah

count = ah.myTokenizer.compute_tokens(
    "請計算這段文字的 Token。",
    "openai:gpt-4o",
)
print(count)
```

簽名：

```python
myTokenizer.compute_tokens(
    text,
    model_id,
    model_path="./tokenizers",
    save_tokenizer=True,
    *,
    exact=False,
)
```

| 參數 | 意義 |
| --- | --- |
| `text` | 要計算的文字。 |
| `model_id` | 模型別名，例如 `openai:gpt-4o` 或 Hugging Face 模型別名。 |
| `model_path` | Hugging Face tokenizer 檔案使用的本機目錄。 |
| `save_tokenizer` | 是否將載入的 Hugging Face tokenizer 儲存到本機。 |
| `exact` | 僅 Gemini：明確要求呼叫 Provider 遠端計數。預設為 `False`。 |

## 支援的模型家族

- OpenAI 模型使用 `tiktoken`。
- Gemini 別名（`gemini:`、`google:`、`gemi:`）預設使用本機 `cl100k_base` 估算，不另外發送 Gemini token 計數請求。
- Hugging Face 模型會載入對應 tokenizer，第一次可能需要下載。
- 其他模型別名會使用 fallback tokenizer 計算。

請使用與實際請求相同的模型別名。這個數量適合用來規劃輸入長度，但不保證等於 Provider 最終回報或計費的使用量。

需要 Gemini 遠端文字計數時，可以明確指定：

```python
count = ah.myTokenizer.compute_tokens(
    "請計算這段文字。", "gemini:gemini-2.5-flash", exact=True,
)
```

此模式需要 `GEMINI_API_KEY`，並遵循 `GOOGLE_GENAI_USE_VERTEXAI` 設定。
遠端計數的請求 timeout 為 10 秒、SDK 嘗試次數為一次，結束後會關閉 client。
失敗時會回報例外，不會偷偷改回本機估算。計數對象是傳入的文字，
並非完整格式化後的生成請求。其他模型家族不支援 `exact=True`。

上方範例已經完整，可以直接複製到自己的專案使用。
