# 常見問題與疑難排解

## `ModuleNotFoundError: No module named akasha`

請啟用安裝 akasha 的虛擬環境，再確認：

```bash
python -c "import akasha; print(akasha.__file__)"
```

## Provider 驗證失敗

確認你選用的 Provider 對應正確的環境變數名稱。除錯時不要印出 key：

```python
import os
print(bool(os.getenv("GEMINI_API_KEY")))
```

## RAG 找不到文件

請從程式實際執行的工作目錄確認路徑：

```python
from pathlib import Path

path = Path("./docs")
print(path.resolve(), path.exists())
```

## 每次回答不完全一樣

模型輸出可能變動。請測試你真正需要的行為，例如必要事實或事件格式，不要直接比對整段生成文字。

如果問題仍存在，請記錄 Python 版本、akasha 版本、Provider，以及去除敏感資訊後的錯誤訊息。
