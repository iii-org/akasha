# 安裝

## 系統需求

- Python 3.11 或 3.12
- 一個虛擬環境
- 使用遠端模型時，需要對應的模型 Provider 帳號

## 輕量安裝

如果使用遠端聊天模型、遠端 Embedding、Chroma RAG 與記憶功能，可以使用 lightweight extra：

```bash
uv venv --python 3.11

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

uv pip install "akasha-terminal[light]"
```

## 完整安裝

如果需要本機 Hugging Face 模型、本機 Embedding、Reranking 或其他本機 ML 功能：

```bash
uv pip install "akasha-terminal[full]"
```

!!! note
    除非你確定需要本機模型功能，否則建議先使用 `light`，安裝較簡單。

## 確認安裝成功

```bash
python -c "import akasha; print('Akasha imported successfully')"
```

下一步：[設定模型 Provider](providers.md)。
