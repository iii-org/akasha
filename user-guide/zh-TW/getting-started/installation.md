# 安裝 akasha

## 系統需求

- Python 3.11 或 3.12
- 一個虛擬環境
- 使用遠端模型時，需要對應的模型 Provider 帳號

## 輕量安裝

如果使用服務型聊天模型、遠端 Embedding、Chroma RAG、LLM reranker 與記憶功能，可以使用 lightweight extra：

```bash
uv venv --python 3.11

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

uv pip install "akasha-terminal[light]"
```

`light` 的 `reranker="llm"` 會透過聊天模型 API 排列候選文件，不會在
Akasha process 中載入 Torch 或 rerank 模型權重。它保留 PDF、DOCX、CSV、
Markdown 與純文字載入，但不會安裝 Streamlit、MLflow、Unstructured、
PPTX parser 或 FAISS。

需要額外功能時，可以只加上對應的 extra：

```bash
uv pip install "akasha-terminal[light,ui]"         # Streamlit toy UI
uv pip install "akasha-terminal[light,tracking]"   # MLflow 實驗追蹤
uv pip install "akasha-terminal[light,documents]"  # PPTX 與 Unstructured 解析
uv pip install "akasha-terminal[light,faiss]"      # search_type="faiss"
```

Akasha 只會在功能被選用時檢查對應套件，因此 light 環境仍可正常 import。
若缺少 extra，會在模型推論或檢索開始前拋出 `OptionalDependencyError`，並列出
`uv add` 與 `pip install` 的完整指令。批次載入目錄是唯一例外：只要仍有其他
文件成功載入，就會警告並略過不支援的文件；若全部文件都被略過，則會明確報錯。

## 完整安裝

如果需要本機 Hugging Face 模型、本機 Embedding、BGE reranker、BERTScore、
PEFT、GPTQ 或 llama.cpp：

```bash
uv pip install "akasha-terminal[full]"
```

!!! note
    除非你確定需要本機模型功能，否則建議先使用 `light`，安裝較簡單。

| 功能 | `light` | `full` |
| --- | --- | --- |
| Cloud、Ollama、vLLM、OpenAI-compatible 聊天模型 | 支援 | 支援 |
| 遠端 Embedding 與本機 Chroma | 支援 | 支援 |
| `reranker="llm"` | 支援 | 支援 |
| 本機 BGE reranker | 不支援 | 支援 |
| 本機 Hugging Face、BERTScore、PEFT、GPTQ、llama.cpp | 不支援 | 支援 |
| Streamlit UI、MLflow、PPTX／Unstructured、FAISS | 依需求加裝 extra | 全部包含 |

`full` 仍是一次安裝所有功能的相容性 profile；除了本機模型 backend，也包含
`ui`、`tracking`、`documents` 與 `faiss` 的功能。

所有安裝方式都要求 Python `>=3.11,<3.13` 與 NumPy `>=2,<3`。
`full` 中部分套件含有 native extension；Windows 使用者可能仍需 Visual
Studio C++ Build Tools 或專案指定的預編譯 wheel。

## 確認安裝成功

```bash
python -c "import akasha; print('akasha imported successfully')"
```

下一步：[設定模型 Provider](providers.md)。
