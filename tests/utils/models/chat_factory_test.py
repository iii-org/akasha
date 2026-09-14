"""Behavior tests for the public chat-model factory."""

import os
import sys
from types import ModuleType

import pytest
from langchain_core.messages import HumanMessage

from akasha.helper.handle_objects import _get_env_var
from akasha.utils.models.chat import build_chat_model


@pytest.fixture
def google_adapter(monkeypatch):
    """Replace the optional Google adapter and capture its public arguments."""

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.environment_at_construction = {
                name: os.environ.get(name)
                for name in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")
            }

    module = ModuleType("langchain_google_genai")
    module.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI
    monkeypatch.setitem(sys.modules, "langchain_google_genai", module)
    return FakeChatGoogleGenerativeAI


def test_gemini_defaults_to_developer_api_with_api_key(google_adapter):
    model = build_chat_model(
        "gemini",
        "gemini-2.5-flash",
        {"GEMINI_API_KEY": "developer-key"},
        temperature=0.25,
        max_output_tokens=256,
    )

    assert isinstance(model, google_adapter)
    assert model.kwargs == {
        "model": "gemini-2.5-flash",
        "api_key": "developer-key",
        "temperature": 0.25,
        "max_output_tokens": 256,
    }


def test_gemini_request_config_disables_sdk_afc():
    model = build_chat_model(
        "gemini",
        "gemini-2.5-flash",
        {"GEMINI_API_KEY": "unused"},
    )

    request = model._prepare_request([HumanMessage(content="hello")])

    assert request["config"].automatic_function_calling.disable is True


def test_gemini_uses_vertex_express_mode_when_explicitly_enabled(
    google_adapter, monkeypatch
):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "ambient-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    model = build_chat_model(
        "gemini",
        "gemini-2.5-flash",
        {
            "GEMINI_API_KEY": "vertex-key",
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GOOGLE_CLOUD_PROJECT": "example-project",
            "GOOGLE_CLOUD_LOCATION": "asia-east1",
        },
    )

    assert isinstance(model, google_adapter)
    assert model.kwargs == {
        "model": "gemini-2.5-flash",
        "api_key": "vertex-key",
        "temperature": 0.0,
        "max_output_tokens": 1024,
        "vertexai": True,
        "location": "",
    }
    assert model.environment_at_construction == {
        "GOOGLE_CLOUD_PROJECT": None,
        "GOOGLE_CLOUD_LOCATION": None,
    }
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "ambient-project"
    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "us-central1"


def test_gemini_vertex_express_mode_does_not_require_project(google_adapter):
    model = build_chat_model(
        "gemini",
        "gemini-2.5-flash",
        {
            "GEMINI_API_KEY": "vertex-key",
            "GOOGLE_GENAI_USE_VERTEXAI": "1",
        },
    )

    assert model.kwargs["vertexai"] is True
    assert model.kwargs["location"] == ""


def test_environment_loader_keeps_vertex_express_mode_separate_from_adc(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "yes")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "example-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-east1")

    env = _get_env_var()

    assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "yes"
    assert "GOOGLE_CLOUD_PROJECT" not in env
    assert "GOOGLE_CLOUD_LOCATION" not in env
