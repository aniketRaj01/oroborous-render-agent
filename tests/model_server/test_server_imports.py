"""The server shell imports only the standard library; torch and ltx_* only in the binding."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SERVER_DIR = Path(__file__).resolve().parents[2] / "model_server"
PACKAGE = SERVER_DIR / "ltx_server"
BINDING = "ltx_binding.py"
BINDINGS = {BINDING, "resident_binding.py"}
SHELL_MODULES = sorted(p for p in PACKAGE.glob("*.py") if p.name not in BINDINGS)
TOOLS = sorted((SERVER_DIR / "tools").glob("*.py"))


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


@pytest.mark.parametrize("path", [*SHELL_MODULES, *TOOLS], ids=lambda p: p.name)
def test_shell_modules_and_tools_import_only_the_standard_library(path: Path) -> None:
    allowed = set(sys.stdlib_module_names) | {"__future__", "ltx_server"}
    if path.name == "compare_clips.py":
        allowed |= {"av", "numpy"}  # the LTX virtualenv has both; imported inside functions only
    assert imported_modules(path) <= allowed, imported_modules(path) - allowed


def test_torch_and_ltx_are_named_only_in_the_binding_module() -> None:
    forbidden = ("torch", "ltx_core", "ltx_pipelines")
    for path in PACKAGE.glob("*.py"):
        if path.name in BINDINGS:
            continue
        found = [name for name in forbidden if name in imported_modules(path)]
        assert found == [], f"{path.name} imports {found}"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            dynamic = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            )
            assert not dynamic, f"{path.name} imports dynamically"


@pytest.mark.parametrize("name", sorted(BINDINGS))
def test_the_binding_imports_torch_and_ltx_only_inside_functions(name: str) -> None:
    tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not top_level & {"torch", "ltx_core", "ltx_pipelines"}


BLOCKER = """
import importlib.abc, sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        allowed = {"ltx_server", "__future__", *sys.stdlib_module_names, *sys.builtin_module_names}
        if top in allowed:
            return None
        raise ImportError(f"blocked import of {name}")

sys.path.insert(0, SERVER_DIR)
sys.meta_path.insert(0, Blocker())
import ltx_server, ltx_server.config, ltx_server.request, ltx_server.output, ltx_server.probe
import ltx_server.logs, ltx_server.renderer, ltx_server.server, ltx_server.__main__
import ltx_server.ltx_renderer, ltx_server.ltx_binding
import ltx_server.resident_renderer, ltx_server.resident_binding
from ltx_server.config import load_settings
from ltx_server.renderer import FakeRenderer
from ltx_server.server import ServerApp
assert "torch" not in sys.modules and not any(m.startswith("ltx_core") for m in sys.modules)
print("imported-with-everything-else-blocked")
"""


def test_the_whole_package_imports_with_torch_ltx_and_every_third_party_module_blocked() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", BLOCKER.replace("SERVER_DIR", repr(str(SERVER_DIR)))],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=SERVER_DIR,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported-with-everything-else-blocked"


def test_loading_the_ltx_renderer_without_ltx_installed_fails_clearly_not_at_import() -> None:
    code = (
        f"import sys; sys.path.insert(0, {str(SERVER_DIR)!r})\n"
        "from ltx_server.config import Settings\n"
        "from ltx_server.ltx_renderer import LtxRenderer\n"
        "r = LtxRenderer(Settings(ltx_root='/nonexistent'))\n"
        "try:\n    r.load()\nexcept Exception as e:\n    print(type(e).__name__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=SERVER_DIR,
    )
    assert result.returncode == 0 and result.stdout.strip() in {
        "FileNotFoundError",
        "ModuleNotFoundError",
    }
