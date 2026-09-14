# Graph Report - akasha-repo  (2026-09-14)

## Corpus Check
- 3 files · ~310 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 14 nodes · 11 edges · 3 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `09aa1aa1`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Q: 那沒有local rerank model的話，那light版是如何用llm api運作的呢
- Q: 我執行 repl_app.py 然後再看輸出時，看不出是模型自己想的，還是有根據 skill 做的? 能在加上一些過程的訊息嗎?
- Q: verbose=True 應由 akasha 內部印出載入 skill 與使用工具的過程，而不是要求範例使用者解析 stream event。

## God Nodes (most connected - your core abstractions)
1. `Q: 我執行 repl_app.py 然後再看輸出時，看不出是模型自己想的，還是有根據 skill 做的? 能在加上一些過程的訊息嗎?` - 4 edges
2. `Q: verbose=True 應由 akasha 內部印出載入 skill 與使用工具的過程，而不是要求範例使用者解析 stream event。` - 4 edges
3. `Q: 那沒有local rerank model的話，那light版是如何用llm api運作的呢` - 3 edges
4. `Answer` - 1 edges
5. `Outcome` - 1 edges
6. `Source Nodes` - 1 edges
7. `Answer` - 1 edges
8. `Outcome` - 1 edges
9. `Source Nodes` - 1 edges
10. `Answer` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities (3 total, 0 thin omitted)

### Community 0 - "Q: 那沒有local rerank model的話，那light版是如何用llm api運作的呢"
Cohesion: 0.50
Nodes (3): Answer, Outcome, Q: 那沒有local rerank model的話，那light版是如何用llm api運作的呢

### Community 124 - "Q: 我執行 repl_app.py 然後再看輸出時，看不出是模型自己想的，還是有根據 skill 做的? 能在加上一些過程的訊息嗎?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: 我執行 repl_app.py 然後再看輸出時，看不出是模型自己想的，還是有根據 skill 做的? 能在加上一些過程的訊息嗎?, Source Nodes

### Community 125 - "Q: verbose=True 應由 akasha 內部印出載入 skill 與使用工具的過程，而不是要求範例使用者解析 stream event。"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: verbose=True 應由 akasha 內部印出載入 skill 與使用工具的過程，而不是要求範例使用者解析 stream event。, Source Nodes

## Knowledge Gaps
- **8 isolated node(s):** `Answer`, `Outcome`, `Source Nodes`, `Answer`, `Outcome` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Work-memory lessons

**Known dead ends** — questions that led nowhere; don't re-derive.
- "那沒有local rerank model的話，那light版是如何用llm api運作的呢"

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `Answer`, `Outcome`, `Source Nodes` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._