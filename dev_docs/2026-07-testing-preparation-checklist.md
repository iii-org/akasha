# 2026-07 測試實際準備清單

Date: 2026-07-14

這份文件只列「開始建置測試前，需要實際準備好的資源」。測試邏輯、capability 判斷、降級規則與驗收標準，請看 2026-07-testing-strengthening-plan.md。

不需要先把所有測試程式寫好；先準備下列 key、模型存取權、資料檔案與執行環境。每項完成後可回報 A-1 done。

## A. API keys 與 provider 帳號

### 本機 .env

請在 akasha-repo/tests/.env 建立本機測試環境檔。這個檔案不可 commit：

    OPENAI_API_KEY=
    OPENAI_BASE_URL=                 # 一般 OpenAI-compatible endpoint 時才需要
    GEMINI_API_KEY=
    ANTHROPIC_API_KEY=

    # 要測 Azure OpenAI 才需要
    # Azure OpenAI-compatible endpoint, kept separate from regular OpenAI
    AZURE_OPENAI_API_KEY=
    AZURE_OPENAI_BASE_URL=

    # 要測 Ollama 才需要
    OLLAMA_API_BASE=http://localhost:11434

- [v] 已準備 OpenAI API key。
- [v] 已準備 Gemini API key。
- [v] 已準備 Anthropic API key；若第一階段不測，可先標記暫緩。
- [v] 已決定是否測 Azure OpenAI，若是，已準備 Azure 設定。
- [v] 已決定是否測 Ollama，若是，已準備 server 與 endpoint。
- [v] tests/.env 已加入 .gitignore 或確認不會被提交。
- [v] API key 有足夠權限呼叫 chat、embedding、vision 所需 API。
- [v] 已確認測試帳號的 rate limit 足以支撐 PR 測試。

不需要準備 BRAVE_API_KEY。tool-calling 測試會使用本地固定工具，不依賴搜尋服務。

Chat API 與 embedding API 可以共用同一個 provider key，但兩者都必須確認帳號權限與模型可用。

### CI secrets

- [ ] CI 已建立 OPENAI_API_KEY secret。
- [ ] 若一般 OpenAI 使用自訂 endpoint，CI 已建立 OPENAI_BASE_URL。
- [ ] CI 已建立 GEMINI_API_KEY secret。
- [ ] CI 已建立 ANTHROPIC_API_KEY secret；若第一階段不測，可先暫緩。
- [ ] Azure/Ollama 若要進 CI，已準備對應 secret 或 runner service。
- [ ] secrets 設為 masked/protected。
- [ ] 已確認 fork PR 不會取得這些 secrets。

## B. 第一批要測的模型清單

請把實際要測的 chat model ID 寫在 tests/config/model_manifest.yaml；不要只提供 provider 名稱。這個檔案可以提交，不得放 API key。

建議先準備每個 provider 一個 PR 模型：

    openai:<model-id>
    azure:<deployment-name>
    gemini:<model-id>
    anthropic:<model-id>
    ollama:<model-id>       # 有固定 Ollama runner 才加入

- [v] 已列出第一批 PR 必測模型的完整 alias。
- [v] Azure deployment 已使用 azure:<deployment-name> 單獨列出。
- [v] 已確認每個 model ID 在帳號中可用。
- [v] 已確認模型能使用的 API endpoint。
- [v] 已確認模型是否需要額外的 deployment name 或 region。
- [v] 已決定哪些模型放 PR、哪些模型放 nightly。
- [v] 已填寫 tests/config/model_manifest.yaml 中的 chat model ID。

能力「支援或不支援」的判斷不需要你先人工填完；之後會由 manifest、probe 與測試共同確認。但 model ID 和帳號存取權必須先有。

## C. Embedding model 清單

請把實際要測的 embedding model ID 寫在 tests/config/model_manifest.yaml 的 embeddings 區段。Embedding model 不是 chat model 的附屬設定，RAG 與 MemoryManager 都會直接依賴它。

建議第一批先準備：

    openai:text-embedding-3-small
    gemini:gemini-embedding-001

- [ ] 已列出第一批 PR 必測 embedding model 的完整 alias。
- [ ] 已確認每個 embedding model 在 provider 帳號中可用。
- [ ] 已確認每個 embedding model 使用哪個 API key。
- [ ] 已確認是否有維度限制、region 或 deployment 設定。
- [ ] 已決定 chat model × embedding model 的最小測試組合。
- [ ] 已確認至少一個同 provider 組合，例如 OpenAI chat + OpenAI embedding。
- [ ] 已確認至少一個跨 provider 組合，例如 Anthropic chat + OpenAI embedding。
- [ ] 已填寫 tests/config/model_manifest.yaml 中的 embedding model ID。

之後的 embedding manifest 會記錄 model ID、provider、維度驗證方式與 PR/nightly 分組；你現在只需要準備可呼叫的 model ID 與帳號權限。

## D. Python 與套件環境

- [ ] 已準備 Python 3.11 或 3.12。
- [ ] 已在 akasha-repo 建立測試 virtual environment。
- [ ] 已安裝專案 dependencies。
- [ ] 已安裝 dev dependencies，至少包含 pytest、pytest-asyncio、pytest-cov。
- [ ] 已安裝要測 provider 的 adapter：langchain-openai、langchain-google-genai、langchain-anthropic、langchain-ollama。
- [ ] 已確認 Chroma 與 embedding 相關套件可 import。
- [ ] 已能在沒有 API key 時執行 unit tests。
- [ ] 已能讀取 tests/.env 執行現有 live test。

## E. 測試資料夾與檔案

請先建立以下資料夾與小型固定檔案；內容可先照範例建立，之後我會依測試需要補強。

    akasha-repo/
    └── tests/
        ├── .env                         # 本機 secrets，不 commit
        ├── data/
        │   ├── rag_smoke/
        │   │   ├── single_fact.txt
        │   │   ├── unicode_繁中.txt
        │   │   ├── empty.txt
        │   │   └── directory/
        │   │       ├── alpha.txt
        │   │       └── beta.txt
        │   ├── vision/
        │   │   ├── text_card.png
        │   │   ├── text_card.jpg
        │   │   └── invalid_image.txt
        │   ├── messages/
        │   │   └── history.json
        │   └── summary/
        │       ├── short.txt
        │       └── short.md
        ├── fixtures/
        │   └── mcp/
        │       └── echo_server.py
        └── artifacts/                   # 測試輸出，可 gitignore

### RAG 文件實際內容

single_fact.txt：

    Akasha RAG smoke verification code: RAG-7319-TAIPEI.

directory/alpha.txt：

    Device Alpha uses protocol NEBULA-441.

directory/beta.txt：

    Device Beta uses protocol ORBIT-882.

unicode_繁中.txt：

    Akasha 中文測試識別碼：繁中-RAG-2026。

empty.txt 保持空白，用來測試空文件處理。

- [ ] 已建立 tests/data/rag/。
- [ ] 已建立上述單檔與多檔文件。
- [ ] 已確認文件是 UTF-8。
- [ ] 已確認文件內容不含秘密或私人資料。
- [ ] 已準備一個小型 PDF 或 DOCX；若第一階段只測 txt，可暫緩。

## F. Vision 實際需要的東西

Vision 測試不是要準備模型輸出答案，而是準備模型可以讀取的圖片檔和固定圖片路徑。

至少準備兩張相同內容、不同格式的圖片：

1. text_card.png
2. text_card.jpg

圖片內容請放大、清楚、不要有複雜背景，至少包含：

    AKASHA VISION TEST
    Code: VISION-4826
    Name: Test User
    Company: Akasha Lab

圖片要求：

- [ ] PNG 一張。
- [ ] JPG/JPEG 一張。
- [ ] 圖片長寬至少約 400×250 px。
- [ ] 文字清楚可讀，避免手寫或藝術字體。
- [ ] 圖片檔案不超過約 1 MB。
- [ ] 圖片不含真實個資。
- [ ] 已準備一個不存在的圖片路徑，用於錯誤處理。
- [ ] 已準備 invalid_image.txt 或損壞圖片，用於格式錯誤處理。

Vision smoke 只需要確認圖片 request 能正確送出、回應能正常接回、response contract 正確；不會用模型回答內容做品質評分。但固定代碼可用來確認模型確實看到了圖片，而不是只回一般文字。

## G. Tool-calling 實際需要的東西

不需要 API key 以外的外部服務，也不需要 Brave key。需要的是三個會由測試程式建立的本地工具：

    add(a, b)
    get_weather(city)
    lookup_version(package)

- [ ] 確認接受使用本地 deterministic tools。
- [ ] 確認不需要外部搜尋、資料庫或網路服務。
- [ ] 若工具有業務專用名稱或輸入格式，已提供給我。

工具函數、invocation record 和預期參數會由我在測試 fixture 中建立，不需要你另外準備檔案。

## H. MCP 實際需要的東西

目前 repo 的 MCP 用法是：先用 langchain-mcp-adapters 連到 MCP server、取得 MCP tools，再把這些 tools 傳給 akasha.agents()。因此 MCP 主要要測的是「MCP tool discovery → agent tool binding → tool invocation → result/event/log normalization」。

第一階段建議使用本地 deterministic MCP server，不需要外部 MCP 服務、資料庫或 API key。測試 fixture 會放在：

    tests/fixtures/mcp/
    └── echo_server.py

MCP server 至少提供三個工具：

    mcp_add(a, b)
    mcp_get_weather(city)
    mcp_lookup_version(package)

- [ ] 確認可以使用本地 MCP server 作為測試服務。
- [ ] 確認已安裝 langchain-mcp-adapters。
- [ ] 確認第一階段測 stdio transport。
- [ ] 若要測 Streamable HTTP，準備可在測試期間啟動的本地 server 與 `/mcp` endpoint。
- [ ] 舊 HTTP+SSE 僅作相容性觀察，不作為新的測試或文件主路徑。
- [ ] MCP server 不依賴外部網路、資料庫或私人資料。
- [ ] MCP 工具名稱、description、input schema 固定且可重現。

測試程式會建立 MCP client、取得 tools，並驗證 agent 確實使用 MCP tool；你不需要另外申請 MCP provider 帳號。若你有指定的實際 MCP server，也可以之後再加入 nightly integration。

## I. RAG vector store 與路徑

- [ ] 確認測試可以在暫存目錄建立 Chroma/vector store。
- [ ] 確認測試帳號可呼叫至少一個 embedding provider。
- [ ] 確認不使用正式資料庫或正式 memory 目錄。
- [ ] 確認測試可以使用 pytest tmp_path。
- [ ] 若要測 Windows path，準備 Windows runner 或可執行 Windows 測試的環境。
- [ ] 若要測 Ollama embedding，確認 Ollama 已安裝對應 embedding model。

不需要手動建立 Chroma 資料庫；測試會每次在 temporary directory 建立並清理。

## J. CI、平台與網路

- [ ] 確認 PR CI runner 可以連線到 OpenAI。
- [ ] 確認 PR CI runner 可以連線到 Gemini。
- [ ] 確認 PR CI runner 可以連線到 Anthropic；若要測。
- [ ] 確認 provider API 沒有被公司 proxy 或 firewall 阻擋。
- [ ] 確認有一個 Windows runner，若要執行 Windows absolute path smoke。
- [ ] 確認有一個 Ollama runner，若要把 Ollama 放入 PR。
- [ ] 確認 CI 能保存 tests/artifacts/ 或等價測試診斷檔。
- [ ] 確認 CI logs 不會印出 secrets。

## K. 先不用準備的東西

以下項目不需要你現在先準備：

- [ ] 不需要先手寫完整 model capability 判斷。
- [ ] 不需要先決定每個 unsupported capability 的程式碼實作。
- [ ] 不需要手動建立 Chroma collection。
- [ ] 不需要準備 Brave API key。
- [ ] 不需要準備大型 PDF 文件。
- [ ] 不需要準備真實業務文件或私人圖片。
- [ ] 不需要準備模型品質評分資料集。

這些會在實作測試時，先寫測試、確認目前失敗行為，再修改 production code。

## 完成回報格式

請依區段逐項回報，例如：

    A-1 done
    A-2 done
    B-1 done：openai:gpt-4o-mini、gemini:gemini-2.5-flash
    D-1 done，已建立資料夾與檔案
    E-1 blocked，尚未準備圖片
    H-4 blocked，目前沒有 Windows runner

所有必要項目完成後，再開始第一批測試案例建置。
