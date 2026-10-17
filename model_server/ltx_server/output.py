"""Atomic output: a clip appears under its final name complete, or not at all."""

from __future__ import annotations

import contextlib
import os
import re
import secrets
from pathlib import Path

# Hidden, in the same directory (so the rename is atomic), and still ending in .mp4 because the
# encoder picks the container from the extension.
TEMP_NAME = re.compile(r"^\.[^/]*\.[0-9a-f]{12}\.partial\.mp4$")


def temp_path_for(final: Path) -> Path:
    return final.with_name(f".{final.stem}.{secrets.token_hex(6)}.partial.mp4")


def commit(temp: Path, final: Path) -> None:
    """Flush the finished temporary file to disk, then rename it over the final name."""
    descriptor = os.open(temp, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temp, final)
    with contextlib.suppress(OSError):  # best effort: make the rename itself durable
        directory = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def remove_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def remove_stale_temp_files(directory: Path) -> int:
    """Delete partial files a killed process left behind. Only our own naming pattern matches."""
    removed = 0
    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if TEMP_NAME.match(entry.name) and entry.is_file() and not entry.is_symlink():
            remove_quietly(entry)
            removed += 1
    return removed
