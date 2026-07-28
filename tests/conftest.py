"""Shared pytest configuration for the repository."""

from pathlib import Path
import shutil


def pytest_sessionfinish(session, exitstatus):
    """Remove the project-local pytest temporary directory after a run."""

    basetemp = getattr(session.config.option, "basetemp", None)
    if not basetemp:
        return
    shutil.rmtree(Path(basetemp), ignore_errors=True)
