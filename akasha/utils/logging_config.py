import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

_console_handler = None
_file_handler = None
_console_enabled = True
_file_path: Optional[str] = None
_lock = threading.Lock()


def _is_akasha_record(record: logging.LogRecord) -> bool:
    if record.name.startswith("akasha"):
        return True
    pathname = (getattr(record, "pathname", "") or "").replace("\\", "/").lower()
    return "/akasha/" in pathname or "akasha_package/akasha" in pathname


class _AkashaConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if _is_akasha_record(record):
            return _console_enabled
        return True


class _AkashaOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _is_akasha_record(record)


def configure_logging(
    verbose: bool = True,
    keep_logs: Union[bool, str] = False,
    log_file: Optional[str] = None,
) -> None:
    global _console_handler, _file_handler, _console_enabled, _file_path

    with _lock:
        _console_enabled = bool(verbose)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        console_fmt = "\x1b[32m[akasha]\x1b[0m %(levelname)s %(message)s"
        file_fmt = "[akasha] %(levelname)s %(message)s"
        if verbose and keep_logs:
            console_fmt = "\x1b[32m[akasha]\x1b[0m %(asctime)s %(levelname)s %(message)s"
            file_fmt = "[akasha] %(asctime)s %(levelname)s %(message)s"

        if _console_handler is None:
            _console_handler = logging.StreamHandler()
            _console_handler.setFormatter(logging.Formatter(console_fmt))
            _console_handler.addFilter(_AkashaConsoleFilter())
            root_logger.addHandler(_console_handler)
        else:
            _console_handler.setFormatter(logging.Formatter(console_fmt))

        log_file_path = None
        if isinstance(keep_logs, str) and keep_logs.strip():
            log_file_path = keep_logs
        elif keep_logs:
            log_file_path = log_file
            if not log_file_path:
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                log_file_path = str(log_dir / f"ak_{datetime.now():%Y%m%d}.log")

        if log_file_path:
            log_path = Path(log_file_path).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file_str = str(log_path.resolve())
            if _file_handler is None or log_file_str != _file_path:
                if _file_handler is not None:
                    root_logger.removeHandler(_file_handler)
                    _file_handler.close()
                _file_handler = logging.FileHandler(log_file_str, encoding="utf-8")
                _file_handler.setFormatter(logging.Formatter(file_fmt))
                _file_handler.addFilter(_AkashaOnlyFilter())
                root_logger.addHandler(_file_handler)
                _file_path = log_file_str
            else:
                _file_handler.setFormatter(logging.Formatter(file_fmt))
        else:
            if _file_handler is not None:
                root_logger.removeHandler(_file_handler)
                _file_handler.close()
                _file_handler = None
                _file_path = None
