from __future__ import annotations

from importlib import metadata

FALLBACK_VERSION = "0.0.0-dev"


def package_version() -> str:
    try:
        return metadata.version("render-agent")
    except metadata.PackageNotFoundError:
        return FALLBACK_VERSION


__version__ = package_version()
