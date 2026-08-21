# Token 計算與輸入長度控制

Token 是模型處理文字時使用的單位。Token 數量不等於字元數或單字數，並且會依照選用的模型與 tokenizer 而不同。

## 直接計算 Token

```python
import akasha.helper as ah

text = "這是一段用來計算 Token 的短文字。"
tokens = ah.myTokenizer.compute_tokens(
    text,
    "openai:gpt-4o",
)

print(f"Token 數量：{tokens}")
```

請使用與實際請求相同的模型別名。同一段文字在不同模型家族中可能得到不同的 Token 數量。

## 控制 `ask` 請求

`ask`、`RAG`、`summary` 與相關 API 會在內部自動計算輸入 Token。可以使用 `max_input_tokens` 限制請求接受的輸入量：

```python
import akasha

qa = akasha.ask(
    model="gemini:gemini-2.5-flash",
    max_input_tokens=3000,
)

answer = qa(
    prompt="請整理重要內容。",
    info="./docs",
)
print(answer)
```

當 Prompt 或背景資料太大時，akasha 可能會分割或截斷文件內容；不同流程也可能直接回報輸入大小錯誤。

## 在 logs 中查看 Token 資訊

除錯輸入大小時，可以啟用 logs 或 verbose 輸出：

```python
qa = akasha.ask(
    model="openai:gpt-4o",
    max_input_tokens=3000,
    keep_logs=True,
    verbose=True,
)
```

請求的 logs 可能包含 Prompt 與文件的 Token 資訊。

!!! warning
    本機計算的 Token 數量是 tokenizer 的估算結果，可能與 Provider 最終回報或計費的使用量不同，尤其是系統訊息、請求 metadata 或 Provider 特有格式也可能被計入時。
