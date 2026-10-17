from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

FALLBACK_VERSION = "0.0.0-dev"

_PROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def parse_project_version(pyproject_text: str) -> str | None:
    """The version under [project] only; other tables may have their own `version` keys."""
    project = pyproject_text.partition("[project]")[2].partition("\n[")[0]
    match = _PROJECT_VERSION.search(project)
    return match[1] if match else None


def _version_from_source_tree() -> str | None:
    """A release is run from its exported source tree without being installed, so there is no
    package metadata; read the version from the pyproject.toml shipped next to src/."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        return parse_project_version(pyproject.read_text(encoding="utf-8"))
    except OSError:
        return None


def package_version() -> str:
    try:
        return metadata.version("render-agent")
    except metadata.PackageNotFoundError:
        return _version_from_source_tree() or FALLBACK_VERSION


__version__ = package_version()
