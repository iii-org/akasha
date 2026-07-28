---
name: python-repl-skill
description: Demonstrate a persistent Python workspace across multiple calculation steps.
---

# Persistent Python Workspace Demo

Use this skill when the user asks for a multi-step Python calculation that reuses intermediate values.

1. In the first Python execution, define values = [2, 4, 6, 8] and total = sum(values).
2. In a separate Python execution, reuse those existing variables to calculate average = total / len(values).
3. Return the resulting average.
4. Do not create or execute a bundled script for this workflow.
