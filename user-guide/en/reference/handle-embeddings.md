# `handle_embeddings`

`akasha.helper.handle_embeddings()` creates an embedding object from a provider alias, an existing embedding object, or a custom embedding function.

## Create an embedding object

```python
import akasha.helper as ah

embedding_model = ah.handle_embeddings(
    "openai:text-embedding-3-small"
)
```

The returned object follows the LangChain `Embeddings` interface.

## Available methods

```python
query_vector = embedding_model.embed_query("What is retrieval?")
document_vectors = embedding_model.embed_documents(
    ["Document one", "Document two"]
)
```

| Method | Return value |
| --- | --- |
| `embed_query(text)` | One vector as a list of floating-point values. |
| `embed_documents(texts)` | A list of vectors, one for each input text. |

## Provider aliases

Common aliases include:

```text
openai:text-embedding-3-small
gemini:gemini-embedding-001
ollama:nomic-embed-text
hf:BAAI/bge-base-en-v1.5
```

Ollama embeddings use the Ollama server and do not require a cloud API key. Pull the embedding model before using it:

```bash
ollama pull nomic-embed-text
```

Then create the embedding object with the `ollama:` alias:

```python
import akasha.helper as ah

embedding_model = ah.handle_embeddings("ollama:nomic-embed-text")
query_vector = embedding_model.embed_query("What is retrieval?")
```

By default, Akasha connects to `http://localhost:11434`. Set `OLLAMA_API_BASE` to use another Ollama server:

```text
OLLAMA_API_BASE=http://192.168.1.10:11434
```

You can also put the endpoint directly in the alias:

```python
embedding_model = ah.handle_embeddings(
    "ollama:http://192.168.1.10:11434@nomic-embed-text"
)
```

Local Hugging Face embeddings may require the full installation. Other remote providers require their corresponding environment variables.

!!! note
    The vector dimension is determined by the selected embedding model. Store and compare vectors produced by compatible models; vectors from different embedding spaces should not be mixed directly.
