"""Live contract checks for the embedding providers used by RAG."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from dotenv import dotenv_values, load_dotenv

from akasha.helper.handle_objects import handle_embeddings


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / "tests" / ".env"
MODEL_MANIFEST = REPO_ROOT / "tests" / "config" / "model_manifest.yaml"
RUN_LIVE = os.getenv("RUN_EMBEDDING_SMOKE", "").lower() in {"1", "true", "yes"}


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
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.smoke,
    pytest.mark.skipif(
        not RUN_LIVE,
        reason="set RUN_EMBEDDING_SMOKE=1 to enable live embedding checks",
    ),
]


@pytest.mark.parametrize("provider,embedding_name,required_key", PROVIDERS)
def test_embedding_provider_returns_vectors(provider, embedding_name, required_key):
    values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    if not (os.getenv(required_key) or values.get(required_key)):
        pytest.skip(f"{required_key} is not configured")
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)

    print(f"[embedding] {provider}: initialize {embedding_name}", flush=True)
    embeddings = handle_embeddings(embedding_name, env_file="")
    vectors = embeddings.embed_documents(["Akasha embedding contract check."])
    print(
        f"[embedding] {provider}: vectors={len(vectors)}, dimension={len(vectors[0])}",
        flush=True,
    )

    assert len(vectors) == 1
    assert len(vectors[0]) > 0
    assert all(isinstance(value, (int, float)) for value in vectors[0])
