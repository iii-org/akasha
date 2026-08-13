# Evaluate a model with `eval`

The `eval` API helps you create a question set from documents and measure a model's answers against the generated reference answers.

## What it supports

| Setting | Values |
| --- | --- |
| `question_type` | `fact`, `summary`, `irrelevant`, or `compared` |
| `question_style` | `essay` or `single_choice` |
| Essay evaluation | BERTScore and ROUGE-L style scores |
| Single-choice evaluation | Correct-answer count |

## Complete example

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

The generated question set contains questions and reference answers. The `evaluation()` call asks the configured model the questions and returns scores based on the selected question style.

## Create questions about a topic

Use `create_topic_questionset()` when the source directory contains several subjects:

```python
questions, answers = evaluator.create_topic_questionset(
    data_source=["./docs"],
    topic="retrieval-augmented generation",
    question_num=3,
    output_file_path="rag-questions.json",
)
```

## Important notes

- Keep `question_type` and `question_style` consistent between question generation and evaluation.
- Use the same or equivalent document source when generating and evaluating the question set.
- Question generation and evaluation both call a model and may incur provider charges.
- Do not commit API keys, generated question sets containing private data, or logs containing sensitive content.
- Treat generated scores as an evaluation signal, not as a complete measure of answer quality.

The repository also contains a longer example in [`examples/ex_eval.py`](https://github.com/iii-org/akasha/blob/master/examples/ex_eval.py).
