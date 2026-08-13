# Akasha 雙語使用說明文件規格

## Problem Statement

Akasha 目前已有 README、開發文件與範例，但使用者需要自己從不同位置拼湊安裝方式、Provider 設定、聊天、RAG、Agent、Streaming、Skills 與 MCP 的使用方法。

現有內容比較適合熟悉專案的開發者快速查閱，還不完全適合第一次接觸 Akasha 的使用者。文件也需要同時提供繁體中文與英文版本，並且發布到 GitHub Pages，讓讀者可以從瀏覽器依照學習順序閱讀。

本規格要建立一份簡單易懂、範例導向、可逐步擴充的 Akasha User Guide。

## Solution

建立一套以任務為中心的雙語使用說明：

- 繁體中文與英文各有完整的導覽與內容。
- 文件由簡單的使用情境開始，再逐步進入核心概念、進階功能與 API 參考。
- 每個主要章節都包含可複製執行的最小範例。
- 每個範例都說明前置條件、設定方式、預期輸出與常見錯誤。
- 文件網站提供明顯的語言切換、章節導覽、搜尋與回到首頁的路徑。
- 原本 `docs/` 中的測試資料與範例資料保持不變；使用說明使用獨立的網站內容來源。

文件首頁應先回答三個問題：

1. Akasha 是什麼？
2. 我應該先看哪一篇？
3. 我可以用哪一個範例開始？

## User Stories

### 初次使用者

1. As a new Python developer, I want to understand what Akasha does, so that I can decide whether it fits my project.
2. As a new user, I want to follow a short installation guide, so that I can prepare a working environment without reading the source code.
3. As a Windows user, I want to see PowerShell commands, so that I can follow the guide in my normal development environment.
4. As a Chinese-speaking user, I want to read the guide in Traditional Chinese, so that I can understand the concepts quickly.
5. As an English-speaking user, I want an equivalent English guide, so that I can share it with international collaborators.
6. As a reader, I want to switch between Chinese and English on every page, so that I do not need to return to the home page.

### 模型與 Provider

7. As a user, I want to know which environment variables are required, so that I can configure a model provider successfully.
8. As a user, I want to see provider-specific examples, so that I can distinguish common settings from provider-specific settings.
9. As a user, I want to know which features require extra packages or services, so that I can avoid unnecessary installation failures.
10. As a user, I want troubleshooting guidance for missing keys, invalid model names, and connection errors, so that I can diagnose setup problems.

### 核心功能

11. As a user, I want a minimal `ask` example, so that I can send my first question and receive an answer.
12. As a user, I want to understand the difference between `ask`, `RAG`, `agents`, `summary`, and `MemoryManager`, so that I can choose the right API.
13. As a user, I want to provide local documents to Akasha, so that I can ask questions about my own content.
14. As a user, I want to provide URLs or other supported information sources, so that I can use external content in a question.
15. As a user, I want to see a complete RAG example, so that I understand ingestion, retrieval, and answering as one workflow.
16. As a user, I want to create an Agent with tools, so that I can let Akasha perform a multi-step task.
17. As a user, I want to understand synchronous and streaming responses, so that I can choose the correct interface for my application.
18. As a user, I want to see the structure of streaming events, so that I can render thinking, tool, and answer events correctly.
19. As a user, I want to load Skills, so that I can extend the agent without changing the core application.
20. As a user, I want to understand MCP integration, so that I can connect external tools or servers safely.

### 應用整合與維運

21. As an application developer, I want a FastAPI integration example, so that I can expose Akasha through an HTTP service.
22. As an application developer, I want guidance for concurrent requests, so that I do not accidentally share unsafe mutable agent state.
23. As an application developer, I want logging and troubleshooting guidance, so that I can investigate failed requests.
24. As a maintainer, I want examples to declare their required provider and environment variables, so that CI can validate them consistently.
25. As a maintainer, I want the Chinese and English pages to have matching names and sections, so that translations remain synchronized.
26. As a maintainer, I want broken links and invalid examples to be detected before publication, so that the public guide remains trustworthy.

## Implementation Decisions

- 文件採用「開始使用、核心概念、操作教學、設定參考、API Reference、疑難排解」的資訊架構。
- 內容採用 Markdown 作為主要格式，網站建置工具負責導覽、搜尋、程式碼高亮與雙語切換。
- 中文和英文頁面使用對應的章節結構；同一主題在兩種語言中使用穩定且可辨識的對應名稱。
- 第一版以繁體中文為主要編寫語言，再產生英文對應內容；英文版不得改變 API 名稱、參數名稱或程式碼行為。
- 每一篇教學遵循固定順序：學習目標、前置條件、最小可執行範例、逐段說明、預期結果、常見錯誤、下一步。
- 程式範例優先使用公開 API：`ask`、`RAG`、`agents`、`summary` 與 `MemoryManager`。
- 範例應區分「完整可執行程式」與「片段說明」，避免讀者誤以為片段可以單獨執行。
- Provider、API key、外部服務與可能產生費用的操作必須清楚標示，且不得把憑證寫入文件或範例。
- 既有測試資料文件不納入公開使用說明的主導覽；如果需要介紹，使用說明只描述用途與使用方式。
- 公開網站使用專案型 GitHub Pages 網址，語言入口以清楚的 `/zh-tw/` 與 `/en/` 區分。
- MVP 先以八個核心主題建立雙語骨架，再逐篇補充進階內容；不在第一版承諾完整覆蓋所有內部 API。

### MVP 文件目錄

每個主題都應有中文與英文版本：

1. Akasha overview
2. Installation
3. Configure a model provider
4. First chat with `ask`
5. Build a RAG workflow
6. Build an Agent and use tools
7. Handle streaming events
8. Troubleshooting and FAQ

### 每篇文件的範例要求

每個主要功能至少提供一個最小範例。範例應包含：

- 完整 import。
- 必要的環境變數名稱，但不包含真實值。
- 可複製的安裝或執行指令。
- 預期輸出或輸出形狀。
- 不確定或依模型而變動的結果說明。
- 一個常見失敗情境與修正方式。

例如，`ask` 教學應先展示最小問題與答案流程，再說明如何加入文件、URL、模型選項與 streaming；RAG 教學則應展示從文件輸入到問題回答的完整流程，而不是只介紹某一個內部類別。

## Testing Decisions

- 文件測試以讀者可觀察的行為為主，不測試 Markdown 排版的內部實作細節。
- 每次建置都應檢查網站是否能成功產生、首頁是否存在、語言入口是否存在，以及頁面間連結是否有效。
- 程式碼區塊至少通過語法檢查；不需要 API key 的範例應在 CI 中執行。
- 需要真實 Provider、憑證或外部服務的範例，標示為 live example，不把未執行過的結果宣稱為已驗證。
- 需要 provider 的範例應提供 mock 或離線驗證方式，讓文件建置不依賴個人憑證。
- 中文與英文版本應檢查標題、章節順序、連結與範例是否一一對應。
- 發布前應人工開啟網站檢查首頁、語言切換、程式碼顯示、手機版排版與錯誤頁面。
- 文件變更應與功能變更一起更新，避免 README、API 行為與 User Guide 出現互相矛盾的說明。

## Out of Scope

- 第一版不建立完整自動產生的所有 API reference。
- 第一版不翻譯既有 `docs/` 測試資料、PDF 或內部測試文件。
- 第一版不提供所有模型 Provider 的完整等量教學；先選擇穩定且可驗證的代表性 Provider。
- 第一版不承諾所有範例都能在沒有任何外部服務的環境中執行。
- 第一版不改變 Akasha 的 Python API、參數名稱、預設行為或依賴設計。
- 第一版不把文件網站改造成產品操作介面，也不加入登入、使用者帳號或線上 Playground。
- 第一版不處理完整版本化文件平台；先保留目前版本的清楚標示，日後再評估多版本文件。

## Further Notes

建議寫作順序如下：

1. 先完成文件首頁與導覽。
2. 寫安裝、Provider 設定與第一個 `ask` 範例。
3. 寫 RAG 與 Agent 兩個核心使用情境。
4. 補上 Streaming、Skills、MCP 與疑難排解。
5. 逐篇建立英文版本。
6. 將所有不需要憑證的範例加入自動檢查。
7. 最後再補 API Reference、效能、部署與進階架構。

文件的語氣應該像一位耐心的工程師帶著讀者操作：先說明「這篇要完成什麼」，接著給完整範例，再解釋原理。避免一開始使用過多內部類別、抽象架構或沒有上下文的 API 清單。

這份規格是 User Guide 的 MVP 基礎；後續若新增功能，應先判斷它屬於新的使用情境、設定參考、API reference 或 troubleshooting，再放入對應區域。
