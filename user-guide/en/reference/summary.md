# `summary`

`summary` summarizes text, files, URLs, or supported document objects.

## Create a summarizer

```python
summarizer = akasha.summary(
    model="gemini:gemini-2.5-flash",
    sum_type="map_reduce",
    sum_len=500,
    chunk_size=500,
    chunk_overlap=50,
)
```

Common constructor options:

| Option | Meaning |
| --- | --- |
| `model` | Model used to generate the summary. |
| `sum_type` | Summary strategy: `map_reduce` or `refine`. |
| `sum_len` | Target summary length. |
| `chunk_size` | Size of input chunks. |
| `chunk_overlap` | Overlap between adjacent chunks. |
| `temperature` | Sampling temperature. |

## Summarize content

```python
import akasha

summarizer = akasha.summary(model="gemini:gemini-2.5-flash")
summary = summarizer("akasha is a Python toolkit for document-aware AI applications.")
print(summary)
```

The `content` argument can also be a file path, URL, list of sources, or supported document object. The normal return value is a final `str`.
