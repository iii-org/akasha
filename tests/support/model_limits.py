"""Model limits and execution budgets for live tests, not library defaults."""
from functools import lru_cache
import yaml
from tests.support.paths import REPO_ROOT

DEFAULT_TEST_OUTPUT_TOKENS = 8192


@lru_cache(maxsize=1)
def model_settings():
    path = REPO_ROOT / "tests/config/model_manifest.yaml"
    return {item["id"]: item for item in yaml.safe_load(path.read_text(encoding="utf-8"))["models"]}


def output_token_budget(model):
    """Keep the test budget separate from the documented service ceiling."""
    entry = model_settings().get(model, {})
    budget = entry.get("test_max_output_tokens", DEFAULT_TEST_OUTPUT_TOKENS)
    ceiling = entry.get("capabilities", {}).get("output_token_limit")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError(f"Invalid test output budget for {model}: {budget}")
    if ceiling is not None and budget > ceiling:
        raise ValueError(f"Test output budget exceeds documented limit for {model}")
    return budget
