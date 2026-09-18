"""Live contract checks for the embedding providers used by RAG."""

from __future__ import annotations

import pytest
import yaml
from tests.support.live import load_test_env, require_keys
from tests.support.paths import REPO_ROOT

from akasha.helper.handle_objects import handle_embeddings


MODEL_MANIFEST = REPO_ROOT / "tests" / "config" / "model_manifest.yaml"


def _required_key(provider: str) -> str:
    return {
        "azure": "AZURE_OPENAI_API_KEY",
    }.get(provider, f"{provider.upper()}_API_KEY")


_manifest = yaml.safe_load(MODEL_MANIFEST.read_text(encoding="utf-8"))
PROVIDERS = [
    (item["provider"], item["id"], _required_key(item["provider"]))
    for item in _manifest["embeddings"]
]

pytestmark = [
    pytest.mark.live,
    pytest.mark.contract,
    pytest.mark.requires_api,
    pytest.mark.smoke,
]


@pytest.mark.parametrize("provider,embedding_name,required_key", PROVIDERS)
def test_embedding_provider_returns_vectors(provider, embedding_name, required_key):
    require_keys(required_key)

    print(f"[embedding] {provider}: initialize {embedding_name}", flush=True)
    embeddings = handle_embeddings(embedding_name, env_file=load_test_env())
    vectors = embeddings.embed_documents(["Akasha embedding contract check."])
    print(
        f"[embedding] {provider}: vectors={len(vectors)}, dimension={len(vectors[0])}",
        flush=True,
    )

    assert len(vectors) == 1
    assert len(vectors[0]) > 0
    assert all(isinstance(value, (int, float)) for value in vectors[0])
