"""Provider-neutral thinking budget normalization."""

import math
from typing import Literal

ThinkingLevel = Literal["low", "medium", "high"]
ThinkingBudget = int | ThinkingLevel | None

THINKING_LEVEL_MULTIPLIERS: dict[ThinkingLevel, float] = {
    "low": 0.5,
    "medium": 1.0,
    "high": 2.0,
}

THINKING_LEVEL_FLOORS: dict[ThinkingLevel, int] = {
    "low": 2048,
    "medium": 4096,
    "high": 8192,
}


def normalize_thinking_budget(
    value: ThinkingBudget,
    *,
    thinking: bool,
    max_output_tokens: int,
) -> int | None:
    """Convert a shared level to a dynamic numeric provider budget."""
    if not thinking or value is None:
        return None
    if isinstance(value, str):
        try:
            level = value  # type: ignore[assignment]
            return max(
                THINKING_LEVEL_FLOORS[level],
                math.ceil(max_output_tokens * THINKING_LEVEL_MULTIPLIERS[level]),
            )
        except KeyError as exc:
            levels = ", ".join(THINKING_LEVEL_MULTIPLIERS)
            raise ValueError(
                f"thinking_budget must be a positive integer or one of: {levels}."
            ) from exc
    if isinstance(value, bool) or value <= 0:
        raise ValueError("thinking_budget must be greater than zero when provided.")
    return value


def normalize_thinking_level(value: ThinkingBudget) -> ThinkingLevel | None:
    """Return an OpenAI-compatible reasoning level when explicitly supplied."""
    if isinstance(value, str):
        if value not in THINKING_LEVEL_MULTIPLIERS:
            levels = ", ".join(THINKING_LEVEL_MULTIPLIERS)
            raise ValueError(f"thinking_budget level must be one of: {levels}.")
        return value  # type: ignore[return-value]
    return None
