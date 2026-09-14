from pathlib import Path
import tomllib

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def _project_metadata():
    pyproject = Path(__file__).parents[3] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]


def _requirement_name(requirement: str) -> str:
    return requirement.split(";", maxsplit=1)[0].split("[", maxsplit=1)[0].split("<", maxsplit=1)[0].split(">", maxsplit=1)[0].split("=", maxsplit=1)[0].strip().lower()


def test_release_targets_python_311_312_and_numpy_2():
    project = _project_metadata()

    assert project["version"] == "1.8.0"
    assert project["requires-python"] == ">=3.11,<3.13"
    assert "numpy>=2,<3" in project["dependencies"]


def test_runtime_uses_numpy_2():
    assert int(np.__version__.split(".", maxsplit=1)[0]) == 2


def test_core_dependencies_include_pillow_for_public_image_apis():
    project = _project_metadata()
    base = {_requirement_name(dep) for dep in project["dependencies"]}

    assert "pillow" in base


@pytest.mark.full_only
def test_installed_local_model_stack_imports_with_numpy_2():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("bert_score")


def test_full_extra_keeps_local_model_backends():
    full = "\n".join(_project_metadata()["optional-dependencies"]["full"])

    for package in (
        "langchain-huggingface",
        "sentence-transformers",
        "bert-score",
        "torch",
        "llama-cpp-python",
        "peft",
        "gptqmodel",
    ):
        assert package in full


def test_light_excludes_large_optional_feature_stacks():
    project = _project_metadata()
    base = {_requirement_name(dep) for dep in project["dependencies"]}
    light = {
        _requirement_name(dep)
        for dep in project["optional-dependencies"]["light"]
    }

    for package in (
        "mlflow",
        "streamlit",
        "streamlit_option_menu",
        "unstructured",
        "python-pptx",
        "faiss-cpu",
    ):
        assert package not in base
        assert package not in light


def test_feature_extras_are_available_and_full_keeps_them():
    extras = _project_metadata()["optional-dependencies"]
    expected = {
        "ui": {"streamlit", "streamlit_option_menu"},
        "tracking": {"mlflow"},
        "documents": {"unstructured", "python-pptx"},
        "faiss": {"faiss-cpu"},
    }
    full = {_requirement_name(dep) for dep in extras["full"]}

    for extra, packages in expected.items():
        assert {_requirement_name(dep) for dep in extras[extra]} == packages
        assert packages <= full


def test_generated_requirements_match_project_profiles():
    root = Path(__file__).parents[3]
    project = _project_metadata()

    def requirements(path: Path) -> list[str]:
        return [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]

    assert requirements(root / "requirements-light.txt") == project["dependencies"]
    assert requirements(root / "requirements.txt") == [
        *project["dependencies"],
        *project["optional-dependencies"]["full"],
    ]
