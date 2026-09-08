"""
shared/logging_config.py

Central logging setup for the invoice pipeline.

Goals:
- One console handler + one rotating file handler, configured once, reused
  by every module (no duplicate handlers, no "why is this printing twice").
- Log level controllable via env var (LOG_LEVEL) without touching code.
- Every log line can be tagged with a thread_id (= source_file / invoice
  being processed), so when you grep logs/pipeline.log for one invoice
  during a batch run, you see its full extraction -> validation ->
  approval trail in order, interleaved correctly with everything else.
- Exceptions always logged with full tracebacks (exc_info=True) at the
  point they're caught, not just re-raised silently.

Usage:
    from shared.logging_config import get_logger, with_thread

    logger = get_logger(__name__)
    logger.info("agent starting")

    # inside a node/function that knows which invoice it's handling:
    tlog = with_thread(logger, thread_id)
    tlog.debug("calling model")
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(os.getenv("PIPELINE_LOG_DIR", "logs"))
_LOG_FILE = _LOG_DIR / "pipeline.log"
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 5

_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-22s | "
    "thread=%(thread_id)s | %(message)s"
)

_configured = False


class _DefaultThreadIdFilter(logging.Filter):
    """Guarantees every record has a thread_id field, even if the caller
    logs through the plain logger instead of a with_thread() adapter."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "thread_id"):
            record.thread_id = "-"
        return True


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("invoice_pipeline")
    root.setLevel(_LOG_LEVEL)
    root.propagate = False  # don't double-log through the bare root logger

    formatter = logging.Formatter(_FORMAT)
    thread_filter = _DefaultThreadIdFilter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_LOG_LEVEL)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(thread_filter)

    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)  # file always keeps full detail
    file_handler.setFormatter(formatter)
    file_handler.addFilter(thread_filter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under invoice_pipeline.<name>, sharing
    the console + rotating-file handlers configured above."""
    _configure_root()
    return logging.getLogger(f"invoice_pipeline.{name}")


def with_thread(logger: logging.Logger, thread_id: str) -> logging.LoggerAdapter:
    """Wrap a logger so every message it emits is tagged with thread_id,
    without having to pass extra={"thread_id": ...} at every call site."""
    return logging.LoggerAdapter(logger, {"thread_id": thread_id})
