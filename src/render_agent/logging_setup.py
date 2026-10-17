from __future__ import annotations

import logging
import re
import sys
import traceback
from datetime import datetime, timezone
from typing import IO

from render_agent.redaction import redact

_STANDARD_ATTRIBUTES = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    | {"message", "asctime", "taskName"}
)
_NEEDS_QUOTING = re.compile(r"[\s\"=]")


class RedactionFilter(logging.Filter):
    """Masks Function URL hosts, AWS key ids, tokens and Authorization values in every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        if record.exc_info is not None:
            record.exc_text = redact("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        for key in set(record.__dict__) - _STANDARD_ATTRIBUTES:
            value = record.__dict__[key]
            if isinstance(value, str):
                record.__dict__[key] = redact(value)
        return True


def _format_value(value: object) -> str:
    text = value if isinstance(value, str) else str(value)
    if text == "" or _NEEDS_QUOTING.search(text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return text


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, timezone.utc)
        parts = [
            f"ts={timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z",
            f"level={record.levelname}",
            f"logger={record.name}",
            f"msg={_format_value(record.getMessage())}",
        ]
        for key in sorted(set(record.__dict__) - _STANDARD_ATTRIBUTES):
            parts.append(f"{key}={_format_value(record.__dict__[key])}")
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            parts.append(f"exc={_format_value(record.exc_text)}")
        return " ".join(parts)


def configure_logging(level: int = logging.INFO, stream: IO[str] | None = None) -> None:
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(KeyValueFormatter())
    handler.addFilter(RedactionFilter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    # httpx logs every request URL at INFO; the Job API URL must never reach a log line.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
