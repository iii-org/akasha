"""Shared pytest policy for the repository."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.live import live_tests_enabled


EXECUTION_TIERS = ("unit", "integration", "live")
FEATURE_MARKERS = {
    "agent",
    "ask",
    "cli",
    "compatibility",
    "db",
    "eval",
    "mcp",
    "memory",
    "observability",
    "provider",
    "rag",
    "skill",
    "summary",
    "tool",
    "vision",
}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Attach feature markers and enforce one execution tier per test."""

    errors: list[str] = []
    live_enabled = live_tests_enabled()
    live_skip = pytest.mark.skip(
        reason="set RUN_LIVE_TESTS=1 to enable external-service tests"
    )

    for item in items:
        path = Path(str(item.path))
        try:
            parts = path.resolve().relative_to(Path(__file__).parent.resolve()).parts
        except ValueError:
            parts = ()

        if parts and parts[0] in FEATURE_MARKERS:
            item.add_marker(getattr(pytest.mark, parts[0]))

        tiers = [name for name in EXECUTION_TIERS if item.get_closest_marker(name)]
        if len(tiers) != 1:
            errors.append(
                f"{item.nodeid}: expected exactly one execution tier, found {tiers}"
            )
            continue

        if tiers[0] == "live" and not live_enabled:
            item.add_marker(live_skip)

    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise pytest.UsageError(f"Invalid test marker policy:\n{joined}")
