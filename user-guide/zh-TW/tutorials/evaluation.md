# 使用 `eval` 評估模型

`eval` 可以從文件建立問題集，再將模型回答與產生的參考答案比較，用來觀察模型表現。

## 支援的設定

| 設定 | 可用值 |
| --- | --- |
| `question_type` | `fact`、`summary`、`irrelevant` 或 `compared` |
| `question_style` | `essay` 或 `single_choice` |
| Essay 評估 | BERTScore 與 ROUGE-L 類型分數 |
| 選擇題評估 | 正確答案數量 |

## 完整範例

```python
import akasha

evaluator = akasha.eval(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    question_type="fact",
    question_style="essay",
    keep_logs=True,
    verbose=True,
)

questions, answers = evaluator.create_questionset(
    data_source=["./docs"],
    question_num=3,
    output_file_path="questions.json",
)

result = evaluator.evaluation(
    questionset_file="questions.json",
    data_source=["./docs"],
)
print(result)
```

產生的問題集會包含問題與參考答案。`evaluation()` 會使用設定好的模型回答問題，再依照題目形式回傳評估分數。

## 建立特定主題的問題集

如果文件目錄包含多個主題，可以使用 `create_topic_questionset()`：

```python
questions, answers = evaluator.create_topic_questionset(
    data_source=["./docs"],
    topic="檢索增強生成",
    question_num=3,
    output_file_path="rag-questions.json",
)
```

## 重要注意事項

- 建立問題集與評估時，`question_type` 和 `question_style` 必須保持一致。
- 建立與評估問題集時，請使用相同或等效的文件來源。
- 建立問題集與評估都會呼叫模型，可能產生 Provider 費用。
- 不要提交 API key、含私人資料的問題集，或包含敏感內容的 logs。
- 評估分數只是觀察模型的其中一種訊號，不代表完整的回答品質。

上方已提供完整流程，包含資料集建立、評估執行與結果檢查。
