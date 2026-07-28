---
name: hello-skill
description: Generate a deterministic greeting with a bundled script.
---

# Hello Skill

Use this skill when the user asks you to greet someone.

1. Execute the bundled Python script at scripts/greet.py.
2. Pass a list containing exactly one name as its arguments.
3. Return the script stdout as the final answer.

The script needs no third-party packages. If it fails, report its exit code and stderr.
