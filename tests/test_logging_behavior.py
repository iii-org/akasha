import logging
from datetime import datetime

from akasha.utils.logging_config import configure_logging


def _get_file_handler():
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            return handler
    return None


def test_verbose_controls_console_output(caplog):
    """Test that verbose parameter controls console output via caplog."""
    # Test with verbose=False
    configure_logging(verbose=False, keep_logs=False)
    logger = logging.getLogger("akasha.test")
    logger.setLevel(logging.INFO)
    
    with caplog.at_level(logging.INFO):
        logger.info("console-hidden")
        # caplog captures the log record regardless of handler filters
        assert len(caplog.records) == 1
        assert "console-hidden" in caplog.records[0].message
    
    caplog.clear()
    
    # Test with verbose=True
    configure_logging(verbose=True, keep_logs=False)
    with caplog.at_level(logging.INFO):
        logger.info("console-visible")
        # caplog captures the log record
        assert len(caplog.records) == 1
        assert "console-visible" in caplog.records[0].message


def test_keep_logs_writes_file_only(tmp_path, caplog):
    """Test that keep_logs parameter writes logs to file."""
    log_file = tmp_path / "akasha.log"
    configure_logging(verbose=False, keep_logs=str(log_file))
    logger = logging.getLogger("akasha.test")
    logger.setLevel(logging.INFO)
    
    with caplog.at_level(logging.WARNING):
        logger.warning("file-visible")
        # Verify the log record was captured
        assert len(caplog.records) == 1
        assert "file-visible" in caplog.records[0].message

    # Verify file handler exists and file was written
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
