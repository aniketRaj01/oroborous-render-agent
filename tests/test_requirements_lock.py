from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "requirements.lock"
SCRIPT = REPO / "scripts" / "export_requirements.sh"

DEV_ONLY = ("pytest", "ruff", "mypy", "moto", "pre-commit", "boto3-stubs", "shellcheck-py")
REQUIREMENT = re.compile(r"^([A-Za-z0-9._-]+)==([^\s;\\]+)")


def blocks() -> list[tuple[str, str, str]]:
    """(name, version, block text) for each pinned requirement."""
    result: list[tuple[str, str, str]] = []
    current: list[str] = []
    for line in LOCK.read_text().splitlines():
        if REQUIREMENT.match(line):
            current = [line]
            match = REQUIREMENT.match(line)
            assert match is not None
            result.append((match[1].lower().replace("_", "-"), match[2], ""))
            result[-1] = (*result[-1][:2], line)
        elif current and line.startswith("    "):
            name, version, text = result[-1]
            result[-1] = (name, version, text + "\n" + line)
    return result


def test_every_requirement_is_pinned_with_hashes() -> None:
    pinned = blocks()
    assert len(pinned) >= 10
    for name, _, text in pinned:
        hashes = re.findall(r"--hash=sha256:[0-9a-f]{64}", text)
        assert hashes, f"{name} has no hashes"


def test_the_lock_has_the_runtime_dependencies_and_no_dev_tools_or_the_project() -> None:
    names = {name for name, _, _ in blocks()}
    assert {"boto3", "httpx", "pydantic", "pydantic-settings", "python-dotenv"} <= names
    assert not names & set(DEV_ONLY)
    assert "render-agent" not in names


def test_the_lock_matches_uv_lock_version_for_version() -> None:
    uv_lock = (REPO / "uv.lock").read_text()
    available = {
        (name.lower().replace("_", "-"), version)
        for name, version in re.findall(
            r'^\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', uv_lock, re.MULTILINE
        )
    }
    assert len(available) > 20
    for name, version, _ in blocks():
        assert (name, version) in available, f"{name}=={version} is not in uv.lock"


def test_every_runtime_dependency_of_the_project_is_locked() -> None:
    pyproject = (REPO / "pyproject.toml").read_text()
    dependencies = pyproject.partition("\ndependencies = [")[2].partition("]")[0]
    declared = {
        re.split(r"[<>=!~ \[]", requirement, maxsplit=1)[0].lower().replace("_", "-")
        for requirement in re.findall(r'"([^"]+)"', dependencies)
    }
    assert {"boto3", "httpx", "pydantic", "pydantic-settings"} == declared
    assert declared <= {name for name, _, _ in blocks()}


def test_the_lock_contains_no_urls_or_local_paths() -> None:
    text = LOCK.read_text()
    assert "://" not in text and "file:" not in text and str(Path.home()) not in text


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_the_committed_lock_is_what_uv_exports() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check"], capture_output=True, text=True, timeout=120, cwd=REPO
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_a_drifted_lock_fails_the_check_and_the_script_rewrites_it(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    for name in ("pyproject.toml", "uv.lock", "requirements.lock", "README.md"):
        shutil.copy(REPO / name, tmp_path / name)
    shutil.copy(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    script = tmp_path / "scripts" / SCRIPT.name
    (tmp_path / "requirements.lock").write_text(LOCK.read_text() + "tampered==9.9.9\n")

    drifted = subprocess.run(
        ["bash", str(script), "--check"], capture_output=True, text=True, timeout=120
    )
    assert drifted.returncode == 1 and "out of date" in drifted.stderr
    assert "tampered" in drifted.stdout  # the diff shows what changed

    subprocess.run(["bash", str(script)], check=True, capture_output=True, timeout=120)
    assert (tmp_path / "requirements.lock").read_text() == LOCK.read_text()
    fixed = subprocess.run(
        ["bash", str(script), "--check"], capture_output=True, text=True, timeout=120
    )
    assert fixed.returncode == 0
