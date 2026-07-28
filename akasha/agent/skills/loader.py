"""Load Agent Skills from SKILL.md files with YAML frontmatter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import Skill


_FRONTMATTER_DELIMITER = "---"
_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")


def _read_skill_document(
    path: Path, include_instructions: bool
) -> tuple[dict[str, Any], str]:
    metadata_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
        if first_line.lstrip(chr(0xFEFF)).strip() != _FRONTMATTER_DELIMITER:
            raise ValueError(
                f"skill must start with YAML frontmatter delimited by '---': {path}"
            )

        for line in handle:
            if line.strip() == _FRONTMATTER_DELIMITER:
                break
            metadata_lines.append(line)
        else:
            raise ValueError(f"skill frontmatter is not closed: {path}")

        instructions = handle.read() if include_instructions else ""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "SKILL.md frontmatter requires PyYAML; install the package dependency first"
        ) from exc

    value = yaml.safe_load("".join(metadata_lines))
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"SKILL.md frontmatter must contain a mapping: {path}")

    return value, instructions.lstrip()


def _validate_metadata(metadata: dict[str, Any], root: Path) -> None:
    name = metadata.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("SKILL.md frontmatter name must be a non-empty string")
    valid_characters = all(
        character == "-"
        or (character.isalnum() and (not character.isalpha() or character.islower()))
        for character in name
    )
    if len(name) > 64 or not valid_characters:
        raise ValueError(
            "SKILL.md frontmatter name must use lowercase letters, numbers, and hyphens"
        )
    if name.startswith("-") or name.endswith("-") or "--" in name:
        raise ValueError("SKILL.md frontmatter name contains invalid hyphen placement")
    if name != root.name:
        raise ValueError(
            f"skill name {name!r} must match its directory name {root.name!r}"
        )

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            "SKILL.md frontmatter description must be a non-empty string"
        )
    if len(description) > 1024:
        raise ValueError(
            "SKILL.md frontmatter description must be at most 1024 characters"
        )

    for field in ("license", "compatibility", "allowed-tools"):
        value = metadata.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"SKILL.md frontmatter {field} must be a string")

    additional = metadata.get("metadata", {})
    if not isinstance(additional, dict):
        raise ValueError("SKILL.md frontmatter metadata must be a mapping")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in additional.items()
    ):
        raise ValueError(
            "SKILL.md frontmatter metadata keys and values must be strings"
        )


def _build_skill(root: Path, include_instructions: bool) -> Skill:
    instructions_path = root / "SKILL.md"
    if not instructions_path.is_file():
        raise FileNotFoundError(f"skill is missing SKILL.md: {root}")

    metadata, instructions = _read_skill_document(
        instructions_path,
        include_instructions=include_instructions,
    )
    _validate_metadata(metadata, root)

    return Skill(
        name=metadata["name"],
        description=metadata["description"],
        instructions=instructions,
        metadata=metadata,
    )


def load_skill_metadata(path: str | Path) -> Skill:
    """Load only standard SKILL.md metadata; defer instructions."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"skill directory does not exist: {root}")
    return _build_skill(root, include_instructions=False)


def load_skill_directory(path: str | Path) -> Skill:
    """Load a skill directory containing standard SKILL.md frontmatter."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"skill directory does not exist: {root}")
    return _build_skill(root, include_instructions=True)
