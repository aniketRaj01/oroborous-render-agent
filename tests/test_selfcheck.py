from __future__ import annotations

import importlib
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

import render_agent
from render_agent.selfcheck import RUNTIME_DEPENDENCIES, run_selfcheck

REPO = Path(__file__).resolve().parent.parent


def run_module(*args: str, extra_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ}
    if extra_path is not None:
        env["PYTHONPATH"] = f"{extra_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", "render_agent", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=REPO,
    )


def test_selfcheck_passes_and_prints_the_version() -> None:
    out = io.StringIO()
    assert run_selfcheck(out) == 0
    assert out.getvalue() == f"render-agent {render_agent.__version__} selfcheck ok\n"


def test_selfcheck_names_the_failing_import_and_exits_4(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.import_module

    def broken(name: str, package: str | None = None) -> object:
        if name == "httpx":
            raise ImportError("boom")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", broken)
    out = io.StringIO()
    assert run_selfcheck(out) == 4
    assert out.getvalue() == "selfcheck failed: cannot import httpx (ImportError)\n"


def test_selfcheck_covers_every_runtime_dependency_and_package_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    real = importlib.import_module

    def spy(name: str, package: str | None = None) -> object:
        seen.append(name)
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", spy)
    run_selfcheck(io.StringIO())
    assert set(RUNTIME_DEPENDENCIES) <= set(seen)
    for module in ("render_agent.runner", "render_agent.cli", "render_agent.job_api.client"):
        assert module in seen


def test_module_selfcheck_and_version_exit_0() -> None:
    selfcheck = run_module("selfcheck")
    assert selfcheck.returncode == 0 and "selfcheck ok" in selfcheck.stdout
    version = run_module("--version")
    assert version.returncode == 0
    assert version.stdout == f"render-agent {render_agent.__version__}\n"


def test_a_broken_dependency_gives_exit_4_not_a_traceback(tmp_path: Path) -> None:
    (tmp_path / "httpx.py").write_text("raise ImportError('simulated damage')\n")
    result = run_module("selfcheck", extra_path=tmp_path)
    assert result.returncode == 4
    assert result.stdout == "selfcheck failed: cannot import httpx (ImportError)\n"
    assert "Traceback" not in result.stderr


def test_selfcheck_needs_no_settings_and_no_network(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "JOB_API", "AGENT_"))}
    result = subprocess.run(
        [sys.executable, "-m", "render_agent", "selfcheck"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=tmp_path,
    )
    assert result.returncode == 0


def test_unknown_arguments_still_reach_the_normal_cli() -> None:
    assert run_module("selfcheck", "extra").returncode == 2  # argparse rejects it


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('[project]\nname = "x"\nversion = "1.2.3"\n', "1.2.3"),
        (
            '[tool.a]\nversion = "9"\n[project]\nversion = "1.2.3"\n[tool.b]\nversion = "7"\n',
            "1.2.3",
        ),
        ('[project]\nname = "x"\n[tool.b]\nversion = "7"\n', None),
        ("", None),
    ],
)
def test_project_version_is_read_from_the_project_table_only(
    text: str, expected: str | None
) -> None:
    assert render_agent.parse_project_version(text) == expected


def test_the_version_matches_pyproject_in_this_checkout() -> None:
    declared = render_agent.parse_project_version((REPO / "pyproject.toml").read_text())
    assert declared is not None
    assert render_agent.__version__ in {declared, declared.replace("-", ".")}
