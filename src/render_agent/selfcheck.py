from __future__ import annotations

import importlib
import pkgutil
from typing import TextIO

import render_agent
from render_agent.exit_codes import EXIT_OK, EXIT_UNEXPECTED

RUNTIME_DEPENDENCIES = (
    "boto3",
    "botocore",
    "httpx",
    "pydantic",
    "pydantic_settings",
    "dotenv",
)


def _package_modules() -> list[str]:
    return sorted(
        module.name
        for module in pkgutil.walk_packages(render_agent.__path__, prefix="render_agent.")
        if module.name != "render_agent.__main__"
    )


def run_selfcheck(out: TextIO) -> int:
    """Import the package and every runtime dependency. No settings, no network, no AWS."""
    try:
        modules = [*RUNTIME_DEPENDENCIES, "render_agent", *_package_modules()]
    except Exception as exc:
        out.write(f"selfcheck failed: cannot list the package modules ({type(exc).__name__})\n")
        return EXIT_UNEXPECTED
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            out.write(f"selfcheck failed: cannot import {name} ({type(exc).__name__})\n")
            return EXIT_UNEXPECTED
    out.write(f"render-agent {render_agent.__version__} selfcheck ok\n")
    return EXIT_OK
