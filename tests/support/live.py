"""Shared configuration helpers for external-service tests."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
from dotenv import dotenv_values, load_dotenv

from tests.support.paths import TEST_ENV_FILE


TRUTHY = {"1", "true", "yes", "on"}


def live_tests_enabled() -> bool:
    """Return whether tests may contact external networks or real services."""

    return os.getenv("RUN_LIVE_TESTS", "").strip().lower() in TRUTHY


def configured_env_file() -> Path | None:
    """Return the explicitly configured or repository-local test env file."""

    configured_path = os.getenv("ENV_FILE")
    if configured_path:
        path = Path(configured_path)
        return path if path.is_file() else None
    return TEST_ENV_FILE if TEST_ENV_FILE.is_file() else None


def load_test_env() -> str:
    """Load test credentials without replacing process-level CI values."""

    path = configured_env_file()
    if path is None:
        return ""
    load_dotenv(path, override=False)
    return str(path)


def configured(key: str) -> bool:
    """Check process and test-file configuration without exposing the value."""

    if os.getenv(key):
        return True
    path = configured_env_file()
    values = dotenv_values(path) if path is not None else {}
    return bool(values.get(key))


def require_keys(*keys: str) -> None:
    """Skip one provider case when any required credential is absent."""

    missing = [key for key in keys if not configured(key)]
    if missing:
        pytest.skip(f"missing required configuration: {', '.join(missing)}")
    load_test_env()


def require_ollama() -> str:
    """Require an explicitly configured and reachable Ollama service."""

    base_url = os.getenv("OLLAMA_API_BASE", "").strip()
    if not base_url:
        pytest.skip("OLLAMA_API_BASE is not configured")

    probe_url = base_url.rstrip("/") + "/api/version"
    try:
        with urlopen(Request(probe_url, method="GET"), timeout=5) as response:
            if 200 <= response.status < 300:
                return base_url
    except (OSError, URLError):
        pass
    pytest.skip("configured OLLAMA_API_BASE is unreachable")
