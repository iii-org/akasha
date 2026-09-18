"""Installed-package contract for the full dependency profile."""

import importlib

import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.contract,
    pytest.mark.full_only,
]


def test_installed_local_model_stack_imports_with_numpy_2():
    for dependency in (
        "accelerate",
        "bert_score",
        "gptqmodel",
        "torch",
        "transformers",
    ):
        try:
            importlib.import_module(dependency)
        except ImportError as exc:
            pytest.fail(f"full installation is missing {dependency}: {exc}")
