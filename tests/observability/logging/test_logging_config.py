import logging

import pytest

from akasha.utils import logging_config

pytestmark = pytest.mark.unit


def test_configure_logging_reuses_console_handler(monkeypatch):
    root = logging.getLogger()
    original_handlers = list(root.handlers)

    logging_config.configure_logging(verbose=False, keep_logs=False)
    first_handler = logging_config._console_handler
    first_count = sum(1 for handler in root.handlers if handler is first_handler)

    logging_config.configure_logging(verbose=True, keep_logs=False)
    second_handler = logging_config._console_handler
    second_count = sum(1 for handler in root.handlers if handler is second_handler)

    assert first_handler is second_handler
    assert first_count == 1
    assert second_count == 1

    root.handlers = original_handlers


def test_configure_logging_replaces_file_handler_when_path_changes(tmp_path):
    root = logging.getLogger()
    original_handlers = list(root.handlers)

    path_a = tmp_path / "a.log"
    path_b = tmp_path / "b.log"

    logging_config.configure_logging(verbose=False, keep_logs=str(path_a))
    first_handler = logging_config._file_handler
    assert first_handler is not None

    logging_config.configure_logging(verbose=False, keep_logs=str(path_b))
    second_handler = logging_config._file_handler

    assert second_handler is not None
    assert second_handler is not first_handler
    assert logging_config._file_path.endswith("b.log")

    root.handlers = original_handlers
