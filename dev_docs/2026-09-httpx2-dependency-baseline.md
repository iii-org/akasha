# Shared HTTPX2 dependency baseline (2026-09-19)

## Decision and scope

Use OpenAI 3.16.2, langchain-openai 1.6.2, HTTPX2 2.13.0 and HTTPCore2
2.13.0 as the current development/CI baseline. Public metadata now requires
`openai>=3.16.2,<4.0` and `langchain_openai>=1.6.2,<2.0`; the user-selected
package version 1.8.1 is preserved. Generated public requirements were updated.

The committed `uv.lock` covers Python 3.11/3.12 and all declared extras. Both
`constraints/light-dev.txt` and `constraints/full-dev.txt` are exports of this
same lock. uv 0.12.13 is the baseline resolver. CI checks that exports match,
installs with the relevant constraints and checks installed dependencies.
Git and system CA certificates are installed before checkout in slim containers.

The existing ignored lock was backed up locally before updating. Its previously
compatible dependencies were retained where possible. Independently resolving
latest light packages first overconstrained full: tokenizers/Hugging Face/fsspec
conflicted with GPTQ's datasets requirement. Resolving all extras together with
the project's Python upper bound preserved a compatible combination, including
huggingface-hub 1.27.0, fsspec 2026.6.0 and tokenizers 0.22.2.

Legacy HTTPX remains for other providers. Akasha's OpenAI async client was checked
at runtime and is an HTTPX2 AsyncClient. The earlier event-loop lifecycle fix is
retained, including its async connection-reuse tradeoff.

The new repo-local `.venv` is the baseline environment. The pre-existing parent
`..\.venv` was not upgraded. Installation and testing documentation now points
to the new environment and version constraints.

## Validation

- Fresh Windows/Python 3.11 light/dev install: 166 packages; `uv pip check` passed.
- Runtime client check: HTTPX2 async client and all four exact SDK versions above.
- Original GPT-5.4 MCP live test: 1 passed with the new SDK/HTTPX2 environment.
- OpenAI provider and embedding selection: 13 passed, 21 deselected; includes
  chat, streaming and Agent contracts as selected by the existing suite.
- Lock check and byte-identical regeneration of both exports passed.
- Full dependency resolution with constraints passed for Linux x86_64 on both
  Python 3.11 and Python 3.12. This checks resolution, not native builds/runtime.
- Packaging/CI contracts: 11 passed after replacing a brittle single-command
  assertion with verification that the full job installs both Git and build tools.
  Removed the obsolete exact 1.8.0 assertion so the user's patch bump is allowed.
- Final complete local suite: **265 passed, 62 deselected, 4 warnings** in
  53.91 seconds; selection `not live and not full_only`.
- CI YAML parsing, changed Python compilation and Git whitespace checks passed.

GitHub Actions itself and the full native-model installation/runtime have not
been executed locally. The full CI job retains those checks. No commit was made.

## Follow-up: checkout-independent export validation

CI later reported `Could not access constraints/uv.lock` after both exports
succeeded. The exact message was reproduced by running the old two-path
`git diff` command outside a Git worktree. This establishes a failure mode,
not the precise reason the CI checkout was not recognized as a worktree.
The provided log does not distinguish missing metadata, repository trust or
working-directory issues.

Both CI jobs now call `scripts/check_dependency_baseline.py`. It runs the same
locked exports, compares file contents and restores originals in a finally
block, without requiring Git metadata. Only LF/CRLF differences are normalized;
stale exports still fail. Missing files and export failures also return nonzero.

Validation: the real baseline check passed; 17 relevant tests passed, including
six cases for valid/stale light/stale full exports with LF and CRLF in a copied
checkout without `.git`. CI YAML and whitespace checks passed. The hosted
GitHub Actions run has not been rerun here.
