from pathlib import Path

import pytest

from akasha.agent.skills import (
    Skill,
    SkillRegistry,
    load_skill_directory,
    load_skill_metadata,
    resolve_skills,
)
from tests.support.paths import FIXTURES_ROOT


pytestmark = pytest.mark.unit


def test_resolve_skills_accepts_names_and_deduplicates():
    research = Skill(
        name="research",
        description="Research workflow",
        instructions="Use reliable sources.",
        version="1.0.0",
    )
    registry = SkillRegistry([research])

    context = resolve_skills(["research", research], registry)

    assert context.names == ["research"]
    assert context.versions == {"research": "1.0.0"}
    assert "Use reliable sources." in context.instructions


def test_resolve_skills_rejects_unknown_names():
    with pytest.raises(LookupError, match="unknown skill"):
        resolve_skills("missing", SkillRegistry())


def test_load_skill_directory_reads_only_skill_markdown():
    skill_dir = FIXTURES_ROOT / "skills" / "research"

    skill = load_skill_directory(skill_dir)

    assert skill.name == "research"
    assert skill.instructions == "Use reliable sources.\n"
    assert skill.tools == ()

def test_skill_context_formats_multiple_instructions():
    context = resolve_skills(
        [
            Skill(name="one", instructions="First"),
            Skill(name="two", instructions="Second"),
        ]
    )

    assert context.instructions == "## Skill: one\nFirst\n\n## Skill: two\nSecond"
def test_load_skill_metadata_defers_instructions():
    skill_dir = FIXTURES_ROOT / "skills" / "research"

    skill = load_skill_metadata(skill_dir)

    assert skill.name == "research"
    assert skill.description == "Research workflow"
    assert skill.instructions == ""
    assert skill.metadata["license"] == "MIT"
    assert skill.metadata["compatibility"] == "Requires Python 3.11"
    assert skill.metadata["metadata"]["author"] == "akasha"
    assert skill.metadata["allowed-tools"] == "Read"
