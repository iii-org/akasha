import sys
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings

from akasha.helper.base import decide_embedding_type
from akasha.helper import handle_objects

pytestmark = pytest.mark.unit


def test_handle_model_supports_ollama_default_base(monkeypatch):
    captured = {}

    def fake_build(
        provider, model_name, env, temperature, max_output_tokens,
        thinking=False, thinking_budget=None,
    ):
        captured.update(
            provider=provider,
            model_name=model_name,
            env=env,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return object()

    monkeypatch.setattr(handle_objects, "_get_env_var", lambda env_file="": {})
    monkeypatch.setattr(
        "akasha.utils.models.chat.build_chat_model", fake_build
    )

    handle_objects.handle_model(
        "ollama:llama3.1",
        temperature=0.2,
        max_output_tokens=256,
    )

    assert captured["provider"] == "ollama"
    assert captured["env"]["OLLAMA_API_BASE"] == "http://localhost:11434"
    assert captured["model_name"] == "llama3.1"
    assert captured["max_output_tokens"] == 256


def test_handle_model_supports_ollama_custom_base(monkeypatch):
    captured = {}

    def fake_build(
        provider, model_name, env, temperature, max_output_tokens,
        thinking=False, thinking_budget=None,
    ):
        captured.update(provider=provider, model_name=model_name, env=env)
        return object()

    monkeypatch.setattr(
        handle_objects,
        "_get_env_var",
        lambda env_file="": {"OLLAMA_API_BASE": "http://env-host:11434"},
    )
    monkeypatch.setattr(
        "akasha.utils.models.chat.build_chat_model", fake_build
    )

    handle_objects.handle_model("ollama:http://custom-host:11434@qwen3:8b")

    assert captured["provider"] == "ollama"
    assert captured["env"]["OLLAMA_API_BASE"] == "http://custom-host:11434"
    assert captured["model_name"] == "qwen3:8b"


def test_handle_model_requires_ollama_model_name():
    with pytest.raises(ValueError):
        handle_objects.handle_model("ollama:")


def test_handle_embeddings_supports_ollama_default_base(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(handle_objects, "_get_env_var", lambda env_file="": {})
    monkeypatch.setattr(
        handle_objects, "OllamaEmbeddings", FakeEmbeddings, raising=False
    )

    result = handle_objects.handle_embeddings("ollama:nomic-embed-text")

    assert isinstance(result, FakeEmbeddings)
    assert captured == {
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434",
    }


def test_handle_embeddings_supports_ollama_environment_base(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        handle_objects,
        "_get_env_var",
        lambda env_file="": {"OLLAMA_API_BASE": "http://env-host:11434"},
    )
    monkeypatch.setattr(
        handle_objects, "OllamaEmbeddings", FakeEmbeddings, raising=False
    )

    handle_objects.handle_embeddings("ollama:nomic-embed-text")

    assert captured["base_url"] == "http://env-host:11434"


def test_handle_embeddings_supports_ollama_custom_base(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        handle_objects,
        "_get_env_var",
        lambda env_file="": {"OLLAMA_API_BASE": "http://env-host:11434"},
    )
    monkeypatch.setattr(
        handle_objects, "OllamaEmbeddings", FakeEmbeddings, raising=False
    )

    handle_objects.handle_embeddings(
        "ollama:http://custom-host:11434@nomic-embed-text"
    )

    assert captured == {
        "model": "nomic-embed-text",
        "base_url": "http://custom-host:11434",
    }


@pytest.mark.parametrize(
    "embedding_name",
    ["ollama:", "ollama:http://localhost:11434@"],
)
def test_handle_embeddings_requires_ollama_model_name(embedding_name):
    with pytest.raises(ValueError, match="ollama embedding model name is required"):
        handle_objects.handle_embeddings(embedding_name)


def test_handle_embeddings_and_name_canonicalizes_ollama_endpoint(monkeypatch):
    class FakeEmbeddings:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

    monkeypatch.setattr(handle_objects, "_get_env_var", lambda env_file="": {})
    monkeypatch.setattr(
        handle_objects, "OllamaEmbeddings", FakeEmbeddings, raising=False
    )

    _, embedding_name = handle_objects.handle_embeddings_and_name(
        "ollama:http://custom-host:11434@nomic-embed-text"
    )

    assert embedding_name == "ollama:nomic-embed-text"


def test_decide_embedding_type_supports_ollama(monkeypatch):
    class FakeOllamaEmbeddings(Embeddings):
        __module__ = "langchain_ollama"

        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts):
            return [[] for _ in texts]

        def embed_query(self, text):
            return []

    monkeypatch.setitem(
        sys.modules,
        "langchain_ollama",
        SimpleNamespace(OllamaEmbeddings=FakeOllamaEmbeddings),
    )

    embeddings = FakeOllamaEmbeddings(model="nomic-embed-text")

    assert decide_embedding_type(embeddings) == "ollama:nomic-embed-text"


def test_handle_model_supports_explicit_azure_alias(monkeypatch):
    captured = {}

    def fake_build(
        provider, model_name, env, temperature, max_output_tokens,
        thinking=False, thinking_budget=None,
    ):
        captured.update(provider=provider, model_name=model_name, env=env)
        return object()

    monkeypatch.setattr(
        handle_objects,
        "_get_env_var",
        lambda env_file="": {
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_OPENAI_BASE_URL": "https://example.openai.azure.com/openai/v1/",
        },
    )
    monkeypatch.setattr("akasha.utils.models.chat.build_chat_model", fake_build)

    handle_objects.handle_model("azure:my-gpt-deployment")

    assert captured["provider"] == "azure"
    assert captured["model_name"] == "my-gpt-deployment"
    assert captured["env"]["AZURE_OPENAI_API_KEY"] == "azure-key"


def test_handle_embeddings_uses_openai_environment_even_when_azure_is_configured(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        handle_objects,
        "_get_env_var",
        lambda env_file="": {
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_BASE_URL": "https://api.openai.example/v1/",
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_OPENAI_BASE_URL": "https://example.openai.azure.com/openai/v1/",
        },
    )
    monkeypatch.setattr(handle_objects, "OpenAIEmbeddings", FakeEmbeddings)

    handle_objects.handle_embeddings("openai:text-embedding-3-small")

    assert captured == {
        "model": "text-embedding-3-small",
        "api_key": "openai-key",
        "base_url": "https://api.openai.example/v1/",
    }


def test_handle_embeddings_uses_azure_only_for_explicit_azure_alias(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        handle_objects,
        "_get_env_var",
        lambda env_file="": {
            "OPENAI_API_KEY": "openai-key",
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_OPENAI_BASE_URL": "https://example.openai.azure.com/openai/v1/",
        },
    )
    monkeypatch.setattr(handle_objects, "OpenAIEmbeddings", FakeEmbeddings)

    handle_objects.handle_embeddings("azure:embedding-deployment")

    assert captured == {
        "model": "embedding-deployment",
        "api_key": "azure-key",
        "base_url": "https://example.openai.azure.com/openai/v1/",
    }


def test_handle_embeddings_uses_regular_openai_base_url(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        handle_objects,
        "_get_env_var",
        lambda env_file="": {
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_BASE_URL": "https://api.openai.example/v1/",
        },
    )
    monkeypatch.setattr(handle_objects, "OpenAIEmbeddings", FakeEmbeddings)

    handle_objects.handle_embeddings("openai:text-embedding-3-small")

    assert captured["api_key"] == "openai-key"
    assert captured["base_url"] == "https://api.openai.example/v1/"


def test_handle_embeddings_requires_complete_azure_configuration(monkeypatch):
    monkeypatch.setattr(
        handle_objects,
        "_get_env_var",
        lambda env_file="": {"AZURE_OPENAI_API_KEY": "azure-key"},
    )

    with pytest.raises(ValueError, match="AZURE_OPENAI_BASE_URL"):
        handle_objects.handle_embeddings("azure:text-embedding-3-small")
