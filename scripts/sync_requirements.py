#!/usr/bin/env python3
"""Sync requirements files from pyproject.toml."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
FULL_REQUIREMENTS = ROOT / "requirements.txt"
LIGHT_REQUIREMENTS = ROOT / "requirements-light.txt"
HEADER = [
    "# Generated from pyproject.toml by scripts/sync_requirements.py",
    "# Do not edit manually.",
    "",
]


def _parse_array(lines: list[str], start: int) -> tuple[list[str], int]:
    values: list[str] = []
    index = start
    pattern = re.compile(r'"([^"]+)"')

    while index < len(lines):
        line = lines[index]
        values.extend(pattern.findall(line))
        if "]" in line:
            return values, index
        index += 1

    raise ValueError("Unterminated array in pyproject.toml")


def _split_dependencies() -> tuple[list[str], list[str]]:
    lines = PYPROJECT.read_text(encoding="utf-8").splitlines()
    in_project_optional = False
    dependencies: list[str] = []
    full_extra: list[str] = []
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()

        if stripped == "[project.optional-dependencies]":
            in_project_optional = True
        elif stripped.startswith("[") and stripped != "[project.optional-dependencies]":
            in_project_optional = False

        if stripped.startswith("dependencies = ["):
            dependencies, index = _parse_array(lines, index)
        elif in_project_optional and stripped.startswith("full = ["):
            full_extra, index = _parse_array(lines, index)

        index += 1

    base = [dep for dep in dependencies if "extra" not in dep]
    return base, full_extra


def _write_requirements(path: Path, requirements: list[str]) -> None:
    path.write_text("\n".join([*HEADER, *requirements, ""]), encoding="utf-8")


def main() -> None:
    base, full_extra = _split_dependencies()
    _write_requirements(LIGHT_REQUIREMENTS, base)
    _write_requirements(FULL_REQUIREMENTS, [*base, *full_extra])


if __name__ == "__main__":
    main()
