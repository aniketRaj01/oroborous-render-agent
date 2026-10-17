"""The scripts that run before any virtualenv exists must stay standard-library only and safe."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = [REPO / "deploy" / "bootstrap.py", REPO / "scripts" / "release.py"]


def tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_only_the_standard_library_is_imported(path: Path) -> None:
    modules: set[str] = set()
    for node in ast.walk(tree(path)):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    assert modules <= set(sys.stdlib_module_names) | {"__future__"}, modules


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_shell_is_ever_used(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "shell=True" not in text and "os.system" not in text and "os.popen" not in text
    for node in ast.walk(tree(path)):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "shell":
                    pytest.fail("a call passes a shell= argument")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_every_subprocess_call_has_a_timeout_and_no_stdin(path: Path) -> None:
    calls = [
        node
        for node in ast.walk(tree(path))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"run", "Popen", "call", "check_call", "check_output"}
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert calls, "expected the script to run subprocesses through subprocess.run"
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "timeout" in keywords, f"line {call.lineno}: subprocess call without a timeout"
        assert "stdin" in keywords, f"line {call.lineno}: subprocess call may wait on stdin"


def test_git_runs_non_interactively_in_the_bootstrap() -> None:
    text = (REPO / "deploy" / "bootstrap.py").read_text(encoding="utf-8")
    assert '"GIT_TERMINAL_PROMPT": "0"' in text


def test_bootstrap_prints_through_one_sanitising_function() -> None:
    text = (REPO / "deploy" / "bootstrap.py").read_text(encoding="utf-8")
    assert "print(" not in text
    writes = [line for line in text.splitlines() if "self._out.write(" in line]
    assert len(writes) == 1 and "sanitize(" in writes[0]
