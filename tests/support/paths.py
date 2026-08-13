"""Stable repository paths used by tests after test-tree reorganization."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
TEST_ENV_FILE = TESTS_ROOT / ".env"
DATA_ROOT = TESTS_ROOT / "data"
DOCUMENTS_ROOT = DATA_ROOT / "documents"
IMAGES_ROOT = DATA_ROOT / "images"
RAG_DATA_ROOT = DATA_ROOT / "rag"
FIXTURES_ROOT = TESTS_ROOT / "fixtures"
