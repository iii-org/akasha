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
)
```

| 參數 | 意義 |
| --- | --- |
| `text` | 要計算的文字。 |
| `model_id` | 模型別名，例如 `openai:gpt-4o` 或 Hugging Face 模型別名。 |
| `model_path` | Hugging Face tokenizer 檔案使用的本機目錄。 |
| `save_tokenizer` | 是否將載入的 Hugging Face tokenizer 儲存到本機。 |

## 支援的模型家族

- OpenAI 模型使用 `tiktoken`。
- Gemini 模型在可用時使用 Gemini Token 計算，失敗時使用近似估算。
- Hugging Face 模型會載入對應 tokenizer，第一次可能需要下載。
- 其他模型別名會使用 fallback tokenizer 計算。

請使用與實際請求相同的模型別名。這個數量適合用來規劃輸入長度，但不保證等於 Provider 最終回報或計費的使用量。

上方範例已經完整，可以直接複製到自己的專案使用。
