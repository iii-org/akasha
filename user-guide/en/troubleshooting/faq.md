# FAQ and troubleshooting

## `ModuleNotFoundError: No module named akasha`

Activate the virtual environment where Akasha was installed, then verify:

```bash
python -c "import akasha; print(akasha.__file__)"
```

## Provider authentication fails

Check the environment variable name for the selected provider. Do not print the key while debugging:

```python
import os
print(bool(os.getenv("GEMINI_API_KEY")))
```

## RAG cannot find documents

Check the path from the same working directory where the program runs:

```python
from pathlib import Path

path = Path("./docs")
print(path.resolve(), path.exists())
```

## The answer is different each time

Model output can vary. Test the behavior you need—such as required facts or a valid event shape—instead of comparing an entire generated answer.

If an issue remains, record the Python version, Akasha version, selected provider, and a redacted error message.
