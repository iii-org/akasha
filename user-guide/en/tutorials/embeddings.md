# Convert text into embeddings

An embedding model converts text into a numeric vector. Texts with related meaning are usually closer together in the embedding space, which makes embeddings useful for semantic search, RAG, classification, and recommendations.

## Create an embedding model

```python
import akasha.helper as ah

embedding_model = ah.handle_embeddings(
    "openai:text-embedding-3-small"
)
```

The provider alias selects the embedding integration. You must configure the corresponding provider credentials before making a remote request.

## Embed one query

```python
vector = embedding_model.embed_query(
    "This is text that will be converted into an embedding."
)

print(type(vector))
print("Dimensions:", len(vector))
print("First values:", vector[:5])
```

`embed_query()` returns a list of floating-point values. The vector dimension depends on the selected embedding model.

## Embed multiple documents

```python
texts = [
    "The first document discusses retrieval.",
    "The second document discusses agents.",
    "The third document discusses memory.",
]

vectors = embedding_model.embed_documents(texts)

print("Documents:", len(vectors))
print("Dimensions:", len(vectors[0]))
```

`embed_documents()` returns one vector for each input text, in the same order as the input list.

## Embeddings in RAG

You usually do not need to call the embedding methods directly for RAG. Pass an embedding alias to `akasha.RAG()` and Akasha handles document and query embeddings:

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="gemini:gemini-embedding-001",
)

answer = rag("./docs", "What is the main topic?")
print(answer)
```

!!! warning
    Remote embedding calls may incur provider charges. Do not commit private text, API keys, or generated vectors to a public repository.
