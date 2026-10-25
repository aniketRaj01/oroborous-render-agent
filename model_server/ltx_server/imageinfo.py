"""Pixel size of a PNG or JPEG from its header, standard library only. For logging; never raises."""

from __future__ import annotations

import struct
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# JPEG start-of-frame markers (those that carry the image size).
JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
READ_LIMIT_BYTES = 1 << 20


def image_size(path: Path) -> tuple[int, int] | None:
    """(width, height), or None when the file is not a PNG/JPEG this can read."""
    try:
        with path.open("rb") as handle:
            data = handle.read(READ_LIMIT_BYTES)
    except OSError:
        return None
    if data.startswith(PNG_SIGNATURE) and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    if data[:2] == b"\xff\xd8":
        position = 2
        while position + 9 < len(data):
            if data[position] != 0xFF:
                position += 1
                continue
            marker = data[position + 1]
            if marker in JPEG_SOF:
                height, width = struct.unpack(">HH", data[position + 5 : position + 9])
                return width, height
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
                position += 1 if marker == 0xFF else 2
                continue
            position += 2 + struct.unpack(">H", data[position + 2 : position + 4])[0]
    return None
