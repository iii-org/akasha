# 將文字轉成 Embedding

## 使用 Ollama embedding model

Akasha 可以使用 Ollama 提供的 embedding model。請先啟動 Ollama，並下載模型：

```bash
ollama pull nomic-embed-text
```

使用 `ollama:` alias 建立 embedding object，後續可以使用相同的 `embed_query()` 與 `embed_documents()` 方法：

```python
import akasha.helper as ah

embedding_model = ah.handle_embeddings("ollama:nomic-embed-text")

query_vector = embedding_model.embed_query("什麼是檢索？")
document_vectors = embedding_model.embed_documents(
    ["第一份文件介紹檢索。"]
)

print("Query 維度：", len(query_vector))
print("Document 維度：", len(document_vectors[0]))
```

預設會連線到 `http://localhost:11434`。如果 Ollama 執行在其他主機，可以設定 `OLLAMA_API_BASE`，或直接在 alias 中指定 endpoint：

```text
OLLAMA_API_BASE=http://192.168.1.10:11434
```

```python
embedding_model = ah.handle_embeddings(
    "ollama:http://192.168.1.10:11434@nomic-embed-text"
)
```

Embedding 模型會將文字轉成數值向量。語意相關的文字通常會在向量空間中比較接近，因此 Embedding 適合用於語意搜尋、RAG、分類與推薦。

## 建立 Embedding 模型

```python
import akasha.helper as ah

embedding_model = ah.handle_embeddings(
    "openai:text-embedding-3-small"
)
```

Provider 別名會決定使用哪一種 Embedding 整合。執行遠端請求前，必須先設定對應 Provider 的憑證。

## 將單一查詢轉成向量

```python
vector = embedding_model.embed_query(
    "這是一段要轉成 Embedding 的文字。"
)

print(type(vector))
print("向量維度：", len(vector))
print("前五個數值：", vector[:5])
```

`embed_query()` 會回傳一個浮點數清單。向量維度會依選用的 Embedding 模型而不同。

## 將多段文件轉成向量

```python
texts = [
    "第一份文件討論檢索。",
    "第二份文件討論 Agent。",
    "第三份文件討論記憶功能。",
]

vectors = embedding_model.embed_documents(texts)

print("文件數量：", len(vectors))
print("向量維度：", len(vectors[0]))
```

`embed_documents()` 會為每段輸入文字回傳一個向量，順序與輸入清單相同。

## RAG 中的 Embedding

通常不需要在 RAG 中直接呼叫 Embedding 方法，只要將 Embedding 別名傳給 `akasha.RAG()`，Akasha 就會自動處理文件與查詢的向量化：

```python
import akasha

rag = akasha.RAG(
    model="gemini:gemini-2.5-flash",
    embeddings="ollama:nomic-embed-text",
)

answer = rag("./docs", "主要主題是什麼？")
print(answer)
```

!!! warning
    遠端 Embedding 請求可能產生 Provider 費用。不要將私人文字、API key 或產生的向量提交到公開 repository。
