"""Verify baseline validation against a real export in a checkout without Git."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def baseline_checkout(tmp_path):
    if shutil.which("uv") is None:
        pytest.skip("uv is required for the real dependency export check")
    for name in ("pyproject.toml", "uv.lock", "constraints/light-dev.txt",
                 "constraints/full-dev.txt", "scripts/check_dependency_baseline.py"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    assert not (tmp_path / ".git").exists()
    return tmp_path


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("stale", [None, "light", "full"])
def test_baseline_check_without_git_detects_drift_and_preserves_files(baseline_checkout, stale, newline):
    root = baseline_checkout
    for path in (root / "constraints").glob("*.txt"):
        path.write_bytes(path.read_text(encoding="utf-8").replace("\n", newline).encode("utf-8"))
    if stale:
        with (root / f"constraints/{stale}-dev.txt").open("a", encoding="utf-8") as stream:
            stream.write("\n# stale baseline\n")
    paths = [root / "uv.lock", *sorted((root / "constraints").glob("*.txt"))]
    before = {path: path.read_bytes() for path in paths}
    result = subprocess.run(
        [sys.executable, str(root / "scripts/check_dependency_baseline.py")],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == (1 if stale else 0), result.stdout + result.stderr
    assert ("baseline is stale" in result.stderr) == bool(stale)
    assert all(path.read_bytes() == content for path, content in before.items())
