# `eval`

`akasha.eval` creates document-based question sets and evaluates model answers against reference answers.

## Constructor

```python
evaluator = akasha.eval(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
    question_type="fact",
    question_style="essay",
)
```

Common constructor options:

| Option | Meaning |
| --- | --- |
| `model` | Model used to generate questions and answers. |
| `embeddings` | Embedding model used to search the source documents. |
| `question_type` | `fact`, `summary`, `irrelevant`, or `compared`. |
| `question_style` | `essay` or `single_choice`. |
| `chunk_size` | Approximate source-document chunk size. |
| `search_type` | Document retrieval strategy. |
| `keep_logs` | Keep request logs for inspection. |
| `verbose` | Show progress and diagnostic information. |

## `create_questionset()`

```python
questions, answers = evaluator.create_questionset(
    data_source=["./docs"],
    question_num=10,
    choice_num=4,
    output_file_path="questions.json",
)
```

Creates a question set from the supplied documents and returns two lists: generated questions and reference answers. `choice_num` applies to `single_choice` questions.

## `create_topic_questionset()`

```python
questions, answers = evaluator.create_topic_questionset(
    data_source=["./docs"],
    topic="retrieval-augmented generation",
    question_num=10,
    output_file_path="topic-questions.json",
)
```

Creates questions after narrowing the source documents to a topic.

## `evaluation()`

```python
result = evaluator.evaluation(
    questionset_file="questions.json",
    data_source=["./docs"],
    eval_model="gemini:gemini-2.5-flash",
)
```

For essay questions, the result contains score values and per-question results. For single-choice questions, it contains the correct-answer result and per-question results. The exact tuple shape depends on `question_style`; inspect the result rather than assuming one shape for both styles.

!!! warning
    `eval` performs live model calls for question generation and evaluation. Use a test subset first, monitor provider costs, and redact private data from generated artifacts.
