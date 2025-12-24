import io
import logging
from datetime import datetime

from akasha.utils.logging_config import configure_logging


def _get_console_handler():
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler):
            formatter = getattr(handler, "formatter", None)
            fmt = getattr(formatter, "_fmt", "")
            if "[akasha]" in fmt:
                return handler
    return None


def _get_file_handler():
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            return handler
    return None


def test_verbose_controls_console_output():
    configure_logging(verbose=False, keep_logs=False)
    logger = logging.getLogger("akasha.test")
    logger.setLevel(logging.INFO)
    handler = _get_console_handler()
    assert handler is not None
    stream = io.StringIO()
    original_stream = handler.stream
    handler.stream = stream
    try:
        logger.info("console-hidden")
        assert "console-hidden" not in stream.getvalue()

        configure_logging(verbose=True, keep_logs=False)
        logger.info("console-visible")
        assert "console-visible" in stream.getvalue()
    finally:
        handler.stream = original_stream


def test_keep_logs_writes_file_only(tmp_path):
    log_file = tmp_path / "akasha.log"
    configure_logging(verbose=False, keep_logs=str(log_file))
    logger = logging.getLogger("akasha.test")
    logger.setLevel(logging.INFO)
    handler = _get_console_handler()
    assert handler is not None
    stream = io.StringIO()
    original_stream = handler.stream
    handler.stream = stream
    try:
        logger.warning("file-visible")
        assert "file-visible" not in stream.getvalue()
    finally:
        handler.stream = original_stream

    file_handler = _get_file_handler()
    assert file_handler is not None
    file_handler.flush()
    assert log_file.exists()
    assert "file-visible" in log_file.read_text(encoding="utf-8")


def test_keep_logs_bool_creates_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    configure_logging(verbose=False, keep_logs=True)
    logger = logging.getLogger("akasha.test")
    logger.setLevel(logging.INFO)
    logger.error("default-path")

    log_file = tmp_path / "logs" / f"ak_{datetime.now():%Y%m%d}.log"
    assert log_file.exists()
    assert "default-path" in log_file.read_text(encoding="utf-8")
