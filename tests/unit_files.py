"""A minimal systemd unit file reader: sections, repeated keys, comments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

Unit = dict[str, dict[str, list[str]]]


def parse_unit(text: str) -> Unit:
    unit: Unit = defaultdict(lambda: defaultdict(list))
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        key, separator, value = line.partition("=")
        if not separator or not section:
            raise ValueError(f"line outside a section or without '=': {raw!r}")
        unit[section][key.strip()].append(value.strip())
    return {name: dict(keys) for name, keys in unit.items()}


def load_unit(path: Path) -> Unit:
    return parse_unit(path.read_text(encoding="utf-8"))
