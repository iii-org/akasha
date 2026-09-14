"""Consistent errors for dependencies installed through optional extras."""

from importlib import import_module
from types import ModuleType


class OptionalDependencyError(ImportError):
    """An explicitly selected feature is missing its optional dependencies."""

    def __init__(self, feature: str, extra: str) -> None:
        self.feature = feature
        self.extra = extra
        package = f"akasha-terminal[{extra}]"
        super().__init__(
            f"{feature} requires optional dependencies. Install with: "
            f'uv add "{package}" or pip install "{package}".'
        )


def require_optional_dependency(
    module_name: str,
    *,
    feature: str,
    extra: str,
) -> ModuleType:
    """Import an optional dependency or raise an actionable, stable error."""

    try:
        return import_module(module_name)
    except ImportError as exc:
        raise OptionalDependencyError(feature, extra) from exc
