# Graph Report - akasha-repo  (2026-08-13)

## Corpus Check
- 2 files · ~234 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 10 nodes · 8 edges · 2 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4fb684f9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Q: 我執行 repl_app.py 然後再看輸出時，看不出是模型自己想的，還是有根據 skill 做的? 能在加上一些過程的訊息嗎?
- Q: verbose=True 應由 akasha 內部印出載入 skill 與使用工具的過程，而不是要求範例使用者解析 stream event。

## God Nodes (most connected - your core abstractions)
1. `Q: 我執行 repl_app.py 然後再看輸出時，看不出是模型自己想的，還是有根據 skill 做的? 能在加上一些過程的訊息嗎?` - 4 edges
2. `Q: verbose=True 應由 akasha 內部印出載入 skill 與使用工具的過程，而不是要求範例使用者解析 stream event。` - 4 edges
3. `Answer` - 1 edges
4. `Outcome` - 1 edges
5. `Source Nodes` - 1 edges
6. `Answer` - 1 edges
7. `Outcome` - 1 edges
8. `Source Nodes` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities (2 total, 0 thin omitted)

### Community 124 - "Q: 我執行 repl_app.py 然後再看輸出時，看不出是模型自己想的，還是有根據 skill 做的? 能在加上一些過程的訊息嗎?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: 我執行 repl_app.py 然後再看輸出時，看不出是模型自己想的，還是有根據 skill 做的? 能在加上一些過程的訊息嗎?, Source Nodes

### Community 125 - "Q: verbose=True 應由 akasha 內部印出載入 skill 與使用工具的過程，而不是要求範例使用者解析 stream event。"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: verbose=True 應由 akasha 內部印出載入 skill 與使用工具的過程，而不是要求範例使用者解析 stream event。, Source Nodes

## Knowledge Gaps
- **6 isolated node(s):** `Answer`, `Outcome`, `Source Nodes`, `Answer`, `Outcome` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `Answer`, `Outcome`, `Source Nodes` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._