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

## Use an Ollama embedding model

Akasha can use an embedding model served by Ollama. First make sure Ollama is running and pull the model:

```bash
ollama pull nomic-embed-text
```

Create the embedding object with an `ollama:` alias, then use the same `embed_query()` and `embed_documents()` methods:

```python
import akasha.helper as ah

embedding_model = ah.handle_embeddings("ollama:nomic-embed-text")

query_vector = embedding_model.embed_query("What is retrieval?")
document_vectors = embedding_model.embed_documents(
    ["The first document discusses retrieval."]
)

print("Query dimensions:", len(query_vector))
print("Document dimensions:", len(document_vectors[0]))
```

Akasha connects to `http://localhost:11434` by default. For another Ollama server, set `OLLAMA_API_BASE` or include the endpoint in the alias:

```text
OLLAMA_API_BASE=http://192.168.1.10:11434
```

```python
embedding_model = ah.handle_embeddings(
    "ollama:http://192.168.1.10:11434@nomic-embed-text"
)
```

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
    embeddings="ollama:nomic-embed-text",
)

answer = rag("./docs", "What is the main topic?")
print(answer)
```

!!! warning
    Remote embedding calls may incur provider charges. Do not commit private text, API keys, or generated vectors to a public repository.
