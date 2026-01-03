# Akasha Upgrade Plan: Light Version Support

## 1. Objective
The goal of this upgrade is to enable a "light" installation of the `akasha` package. This version will only support API-based models (Cloud/Remote) and will avoid installing heavy dependencies like `torch`, `transformers`, and `onnxruntime`, which are significantly large and often unnecessary for users who only rely on OpenAI, Gemini, or Anthropic.

## 2. Proposed Syntax
- **Light Installation (API / Remote Only):**
  ```bash
  pip install akasha-terminal[light]
  ```
  *Ideal for serverless environments, lightweight containers, or users only using OpenAI/Gemini/Anthropic.*

- **Full Installation (Default - Local Models & Heavy Analytics):**
  ```bash
  pip install akasha-terminal[full]
  ```
  **OR**
  ```bash
  pip install akasha-terminal
  ```
  *Includes all local inference support and heavy evaluation metrics. This remains the default behavior to ensure backward compatibility.*

## 3. Light vs. Full Comparison

| Feature | Light (`akasha-terminal[light]`) | Full (`akasha-terminal[full]`) |
| :--- | :---: | :---: |
| **OpenAI / Gemini / Anthropic** | ✅ | ✅ |
| **Remote TGI / vLLM API** | ✅ | ✅ |
| **Local HuggingFace Models** | ❌ | ✅ |
| **Local GGUF / Llama-cpp** | ❌ | ✅ |
| **Local GPTQ / AWQ Models** | ❌ | ✅ |
| **Local Embeddings (Sentence-Transformers)** | ❌ | ✅ |
| **Heavy Metrics (BERTScore)** | ❌ | ✅ |
| **Rerank (Local Models)** | ❌ | ✅ |
| **Memory Requirement** | ⬇️ Low (~500MB) | ⬆️ High (4GB+) |
| **Disk Space** | ~200MB | 5GB+ |

---

## 4. Why Rerank & BERTScore are moved to `[full]`?

The removal of **Rerank** and **BERTScore** from the light version is driven by their heavy reliance on local machine learning frameworks:

1.  **Heavy Dependencies**: Both features require `torch` and `transformers`. `torch` alone consumes several gigabytes of disk space and complex C++ extensions (CUDA, etc.), which contradicts the "lightweight" goal.
2.  **Local Model Inference**: 
    - **Rerank** uses Cross-Encoder models (e.g., `BAAI/bge-reranker-base`) to score document relevance locally.
    - **BERTScore** loads a BERT/RoBERTa model into memory to compute semantic similarity scores.
3.  **Future Opportunity**: Features like "Remote Rerank" (via Cohere API or custom TGI endpoints) can be added to the `[light]` version in the future as they only require standard HTTP calls.

## 5. Is RAG still functional in `[light]`?

**YES.** Standard RAG remains fully functional because its core steps can be entirely API-based:

1.  **Embeddings**: You can use OpenAI/Gemini/Anthropic APIs to generate vectors.
2.  **Vector Search**: The similarity calculation (KNN) and keyword search (BM25) use standard Python libraries (scikit-learn, numpy, or pure math) and do not strictly require `torch`.
3.  **Generation**: The retrieval context is passed to the LLM API as usual.

**Standard Usage in `[light]`:**
```python
ak = akasha.RAG(model="openai:gpt-4o", search_type="auto")
ak(data_source="./data", prompt="Your question")
```
*(Note: `search_type="auto"` in light mode will skip the rerank step if torch is missing.)*

---

## 6. Dependency Reorganization
The following dependencies will be moved from the core `dependencies` to the `[full]` optional extra in `pyproject.toml`:

| Package | Category | Reason |
|---------|----------|--------|
| `torch` | Heavy | Core for local ML |
| `torchvision` | Heavy | Vision support for local models |
| `transformers` | Heavy | HuggingFace model support |
| `accelerate` | Heavy | Training/Inference optimization |
| `sentence-transformers` | Heavy | Local embeddings |
| `langchain-huggingface` | Heavy | Integration with local models |
| `onnxruntime` | Medium | Inference optimization |
| `tokenizers` | Medium | HuggingFace tokenization |
| `sentencepiece` | Medium | HuggingFace tokenization |
| `bert-score` | Evaluation | Uses transformers/torch |

---

## 7. User Experience: Error & Warning Handling

In the `light` version, we must ensure users are properly informed when they hit a boundary:

- **Rerank Attempts**: If `search_type="auto_rerank"` or `"rerank:..."` is used, the system should check for `torch`. If missing, it should display:
  > **Warning**: Rerank requires local model support. This is only available in the `full` version. Switching to standard RAG...
- **BERTScore Attempts**: If an evaluation is triggered using BERTScore, it should display:
  > **Error**: BERTScore requires the `full` version (`pip install akasha-terminal[full]`) to run local scoring models.
- **Local Model Usage**: Initializing any `hf:`, `llama:`, or `gptq:` model should immediately prompt the user to install the `full` version.

## 8. Implementation Steps

### Phase 1: Refactor Code for Optional Dependencies
1.  **Wrap Imports**: Identify all modules that import the heavy dependencies and wrap them in `try...except ImportError` blocks.
    -   `akasha/utils/models/hf.py`
    -   `akasha/utils/models/gtq.py`
    -   `akasha/utils/models/llamacpp2.py`
    -   `akasha/utils/search/retrievers/retri_rerank.py` (if it uses cross-encoders)
2.  **Graceful Errors & Warnings**: 
    -   Update `akasha/helper/handle_objects.py`: If a user specifies a model type that requires an uninstalled package (e.g., `model="hf:..."`), raise an `ImportError` with a message like:
        > "Feature requiring 'torch/transformers' is not installed. Please install with: pip install akasha-terminal[full]"
    -   Update `akasha/utils/search/retrievers/base.py`: If `search_type` includes `rerank` and `torch` is missing, print a warning and fallback to standard retrieval, or raise an error depending on the strictly asked mode.
    -   Update `akasha/helper/scores.py`: If `get_bert_score` is called and `bert_score` package is missing, raise an `ImportError` or `RuntimeError` stating the feature is only in the `full` version.

### Phase 2: Update Configuration Files
1.  **Modify `pyproject.toml`**:
    -   Remove heavy packages from `dependencies`.
    -   Add `full` extra containing the heavy packages.
    -   Add `light` extra (which can be empty or contain a subset if preferred).
2.  **Update `requirements.txt`**: Create a `requirements-light.txt` and a `requirements.txt` (full).

### Phase 3: Verification
1.  Create a separate testing directory `upgrade_tests/`.
2.  Perform tests in a clean environment without `torch`.

---

## 9. Testing Plan (`upgrade_tests`)

We will use `pytest` for verification. These tests will use real LLM APIs to ensure the package remains functional in a light environment.

### Test Items:
1.  **Base Functionality (No Torch)**:
    -   Initialize `RAG` with OpenAI/Gemini.
    -   Verify that `torch`, `transformers`, etc., are NOT loaded into `sys.modules`.
    -   Run a standard RAG query.
2.  **Graceful Failure**:
    -   Try to initialize an `hf_model`.
    -   Assert that a clear `ImportError` is raised.
3.  **API Compatibility**:
    -   RAG with `openai:gpt-4o`.
    -   RAG with `gemini:gemini-2.5-flash`.
    -   RAG with `anthropic:claude-3-5-sonnet`.
4.  **Embedding Compatibility**:
    -   RAG with `openai` embeddings.
    -   RAG with `gemini` embeddings.
5.  **CLI/UI Check**:
    -   Ensure CLI starts without crashing due to missing imports.

### Execution with `uv` (Recommended)

Using **`uv`** is the fastest way to set up a clean testing environment for both `light` and `full` versions.

1.  **Prepare Environment Variables**:
    Create a `.env` file inside the `upgrade_tests/` directory with your API keys:
    ```env
    OPENAI_API_KEY=sk-...
    GEMINI_API_KEY=AIza...
    ```
    *Note: The included `conftest.py` will automatically load these variables when running pytest.*

2.  **Run Light Version Tests**:
    This simulates an environment without heavy dependencies:
    ```bash
    uv run --extra light --extra dev pytest upgrade_tests/
    ```

3.  **Run Full Version Tests**:
    This ensures all local model features work as expected:
    ```bash
    uv run --extra full --extra dev pytest upgrade_tests/
    ```

4.  **Standard Test Command**:
    If you just want to run with default dependencies:
    ```bash
    uv run --extra dev pytest upgrade_tests/
    ```
