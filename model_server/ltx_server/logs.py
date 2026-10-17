"""One-line key=value logging. Callers pass ids, sizes, timings and outcomes, never prompts."""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime, timezone
from typing import IO

logger = logging.getLogger("ltx_server")
_NEEDS_QUOTING = re.compile(r"[\s\"=]")


def _format(value: object) -> str:
    text = value if isinstance(value, str) else str(value)
    if text == "" or _NEEDS_QUOTING.search(text):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    return text


def event(level: int, name: str, **fields: object) -> None:
    if logger.isEnabledFor(level):
        logger.log(
            level, " ".join([f"event={name}", *(f"{k}={_format(v)}" for k, v in fields.items())])
        )


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        when = datetime.fromtimestamp(record.created, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        return f"ts={when[:-3]}Z level={record.levelname} {record.getMessage()}"


def configure_logging(level: int = logging.INFO, stream: IO[str] | None = None) -> None:
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(_Formatter())
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
