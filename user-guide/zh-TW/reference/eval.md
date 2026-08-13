# `eval`

`akasha.eval` 可以根據文件建立問題集，並將模型回答與參考答案比較。

## 建立物件

```python
evaluator = akasha.eval(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    question_type="fact",
    question_style="essay",
)
```

常用建立參數：

| 參數 | 意義 |
| --- | --- |
| `model` | 建立問題與答案時使用的模型。 |
| `embeddings` | 搜尋文件時使用的 Embedding 模型。 |
| `question_type` | `fact`、`summary`、`irrelevant` 或 `compared`。 |
| `question_style` | `essay` 或 `single_choice`。 |
| `chunk_size` | 文件分段的大約大小。 |
| `search_type` | 文件檢索策略。 |
| `keep_logs` | 是否保留請求紀錄供檢查。 |
| `verbose` | 是否顯示進度與診斷資訊。 |

## `create_questionset()`

```python
questions, answers = evaluator.create_questionset(
    data_source=["./docs"],
    question_num=10,
    choice_num=4,
    output_file_path="questions.json",
)
```

此方法會根據文件建立問題集，並回傳兩個清單：產生的問題與參考答案。`choice_num` 只適用於 `single_choice` 題型。

## `create_topic_questionset()`

```python
questions, answers = evaluator.create_topic_questionset(
    data_source=["./docs"],
    topic="檢索增強生成",
    question_num=10,
    output_file_path="topic-questions.json",
)
```

此方法會先縮小到指定主題，再建立問題集。

## `evaluation()`

```python
result = evaluator.evaluation(
    questionset_file="questions.json",
    data_source=["./docs"],
    eval_model="gemini:gemini-2.5-flash",
)
```

Essay 題型會回傳分數與各題結果；`single_choice` 會回傳正確答案相關結果與各題結果。實際 tuple 結構會依 `question_style` 不同而變化，請依實際結果處理，不要假設兩種題型完全相同。

!!! warning
    `eval` 會在建立問題與評估時呼叫即時模型。建議先使用小型文件集測試，留意 Provider 費用，並將私人資料從產生的檔案中移除。
