from pathlib import Path

import pytest

from akasha.agent.skills import resolve_skills


pytestmark = pytest.mark.unit


def test_resolve_skills_loads_directory_without_global_registration():
    skill_path = Path(__file__).parents[1] / "fixtures" / "skills" / "research"

    context = resolve_skills([str(skill_path)])

    assert context.names == ["research"]


def test_resolve_skills_rejects_missing_path_like_value():
    with pytest.raises(FileNotFoundError, match="skill directory"):
        resolve_skills(["skills/missing"])
def test_resolve_skills_validates_standard_skill_metadata():
    from akasha.agent.skills import load_skill_directory

    invalid = Path(__file__).parents[1] / "fixtures" / "skills" / "invalid-name"
    mismatched = Path(__file__).parents[1] / "fixtures" / "skills" / "mismatched"

    with pytest.raises(ValueError, match="lowercase"):
        load_skill_directory(invalid)
    with pytest.raises(ValueError, match="match its directory"):
        load_skill_directory(mismatched)
