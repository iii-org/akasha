# API Reference

This section describes akasha's public Python entry points. Start with the API overview, then open the page for the feature you need.

## Public entry points

| API | Purpose | Detailed reference |
| --- | --- | --- |
| `akasha.ask()` | Chat and question answering, optionally with context. | [`ask`](ask.md) |
| `asker.vision()` | Ask questions about images and receive text answers. | [`Vision and image editing`](vision.md) |
| `akasha.gen_image()` | Generate a new image from a text prompt. | [`Vision and image editing`](vision.md) |
| `akasha.edit_image()` | Remove, add, or change content in an existing image. | [`Vision and image editing`](vision.md) |
| `akasha.RAG()` | Load, retrieve, and answer questions about documents. | [`RAG`](rag.md) |
| `akasha.agents()` | Run an agent with tools, Skills, or MCP tools. | [`agents`](agents.md) |
| `akasha.summary()` | Summarize text, files, or URLs. | [`summary`](summary.md) |
| `akasha.MemoryManager` | Store and search long-term semantic memory. | [`MemoryManager`](memory-manager.md) |

## How to read the signatures

The examples show the most common options, not every advanced parameter. Every API accepts more configuration for token limits, logging, prompts, and provider-specific behavior. Use the examples as a starting point and consult the source signature when you need an uncommon option.

!!! warning
    API keys, model access, and generated answers depend on the selected Provider. Documentation examples do not include real credentials or promise identical text output.
