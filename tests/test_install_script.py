from __future__ import annotations

import getpass
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from render_agent.config import ConfigError
from render_agent.config_sources import load_settings
from unit_files import load_unit

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
INSTALL = DEPLOY / "install.sh"
USER = getpass.getuser()
TOOLS = (
    "bash",
    "dirname",
    "id",
    "sed",
    "cmp",
    "mktemp",
    "rm",
    "cp",
    "chmod",
    "chown",
    "mkdir",
    "ln",
    "awk",
    "cat",
    "readlink",
    "git",
    "python3",
)


def run(
    *args: str, root: Path | None = None, path: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("SUDO_USER",)}
    if path is not None:
        env["PATH"] = path
    command = ["bash", str(INSTALL), *args]
    if root is not None:
        command += ["--root", str(root)]
    return subprocess.run(command, capture_output=True, text=True, timeout=60, env=env, check=False)


def install(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return run("--service-user", USER, *extra, root=root)


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def tree(root: Path) -> dict[str, tuple[str, int, int]]:
    """Path -> (kind or link target, mode, mtime) for everything under root."""
    state: dict[str, tuple[str, int, int]] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        info = path.lstat()
        kind = (
            f"-> {os.readlink(path)}" if path.is_symlink() else ("dir" if path.is_dir() else "file")
        )
        state[relative] = (kind, stat.S_IMODE(info.st_mode), info.st_mtime_ns)
    return state


def bin_dir(tmp_path: Path, *, without: tuple[str, ...] = ()) -> str:
    """A PATH with only the tools install.sh needs, minus the ones named."""
    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    for tool in TOOLS:
        if tool in without:
            continue
        found = sys.executable if tool == "python3" else shutil.which(tool)
        if found and not (directory / tool).exists():
            (directory / tool).symlink_to(found)
    return str(directory)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "staging"


def test_staging_install_produces_the_expected_tree(root: Path) -> None:
    result = install(root)
    assert result.returncode == 0, result.stderr
    uid = os.getuid()
    for directory, expected in (
        ("opt/render-agent", 0o755),
        ("var/lib/render-agent", 0o750),
        ("etc/render-agent", 0o755),
    ):
        assert (root / directory).is_dir() and mode(root / directory) == expected, directory
        assert (root / directory).stat().st_uid == uid
    bootstrap = root / "opt/render-agent/bootstrap.py"
    assert (
        bootstrap.read_bytes() == (DEPLOY / "bootstrap.py").read_bytes()
        and mode(bootstrap) == 0o755
    )
    for name in (
        "render-agent.service",
        "render-agent-deadman.service",
        "render-agent-deadman.timer",
    ):
        installed = root / "etc/systemd/system" / name
        assert installed.is_file() and mode(installed) == 0o644
        assert "@SERVICE_USER@" not in installed.read_text()
    assert not (root / "run").exists()  # /run/render-agent comes from RuntimeDirectory=


def test_env_files_are_created_from_the_examples_owned_by_the_user_with_mode_0600(
    root: Path,
) -> None:
    install(root)
    for name in ("agent.env", "deploy.env"):
        created = root / "etc/render-agent" / name
        assert created.read_bytes() == (DEPLOY / f"{name}.example").read_bytes()
        assert mode(created) == 0o600 and created.stat().st_uid == os.getuid()


def test_existing_env_files_are_kept_but_their_mode_is_corrected(root: Path) -> None:
    (root / "etc/render-agent").mkdir(parents=True)
    agent_env = root / "etc/render-agent/agent.env"
    agent_env.write_text("AGENT_ID=mine\n")
    agent_env.chmod(0o644)
    install(root)
    assert agent_env.read_text() == "AGENT_ID=mine\n" and mode(agent_env) == 0o600


def test_the_installed_service_unit_runs_as_the_service_user(root: Path) -> None:
    install(root)
    unit = load_unit(root / "etc/systemd/system/render-agent.service")
    assert unit["Service"]["User"] == [USER]
    assert unit["Service"]["ExecStartPre"] == ["/usr/bin/python3 /opt/render-agent/bootstrap.py"]
    assert unit["Install"]["WantedBy"] == ["multi-user.target"]


def test_the_agent_service_is_enabled_and_the_deadman_timer_is_not(root: Path) -> None:
    install(root)
    wants = root / "etc/systemd/system/multi-user.target.wants/render-agent.service"
    assert wants.is_symlink()
    assert os.readlink(wants) == "/etc/systemd/system/render-agent.service"
    assert not (root / "etc/systemd/system/timers.target.wants").exists()
    assert all("deadman" not in str(p) for p in (root / "etc/systemd/system").rglob("*.wants/*"))


def test_enable_deadman_enables_the_timer_only_when_asked(root: Path) -> None:
    result = install(root, "--enable-deadman")
    assert result.returncode == 0
    timer = root / "etc/systemd/system/timers.target.wants/render-agent-deadman.timer"
    assert (
        timer.is_symlink()
        and os.readlink(timer) == "/etc/systemd/system/render-agent-deadman.timer"
    )
    assert "ENABLED" in result.stdout


def test_running_it_again_changes_nothing(root: Path) -> None:
    install(root, "--enable-deadman")
    before = tree(root)
    again = install(root, "--enable-deadman")
    assert again.returncode == 0
    assert tree(root) == before  # contents, modes, link targets and modification times
    assert "already enabled: render-agent.service" in again.stdout
    assert "up to date: /opt/render-agent/bootstrap.py" in again.stdout


def test_a_changed_source_file_is_reinstalled(root: Path) -> None:
    install(root)
    stale = root / "opt/render-agent/bootstrap.py"
    stale.write_text("old")
    install(root)
    assert stale.read_bytes() == (DEPLOY / "bootstrap.py").read_bytes()


def test_dry_run_prints_the_plan_and_changes_nothing(root: Path) -> None:
    result = install(root, "--dry-run")
    assert result.returncode == 0
    assert "would run:" in result.stdout and "render-agent.service" in result.stdout
    assert not root.exists()


def test_dry_run_without_a_root_touches_nothing_on_this_machine(tmp_path: Path) -> None:
    result = run("--service-user", USER, "--dry-run")
    assert result.returncode == 0 and "would run: systemctl daemon-reload" in result.stdout
    assert "would run: systemctl enable render-agent.service" in result.stdout
    assert "enable render-agent-deadman.timer" not in result.stdout


def test_missing_git_is_reported_by_package_name_and_nothing_is_installed(
    tmp_path: Path, root: Path
) -> None:
    result = run("--service-user", USER, root=root, path=bin_dir(tmp_path, without=("git",)))
    assert result.returncode == 1
    assert "missing prerequisites: git" in result.stderr
    assert "apt-get install -y git" in result.stderr and "--install-packages" in result.stderr
    assert not (root / "opt").exists()


def write_fake_python(directory: Path, *, version_ok: bool, venv_ok: bool) -> None:
    (directory / "python3").unlink(missing_ok=True)
    (directory / "python3").write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f'*"version_info >= (3, 10)"*) exit {0 if version_ok else 1} ;;\n'
        '*"print("*) echo 3.8 ;;\n'
        f'*"import venv"*) exit {0 if venv_ok else 1} ;;\n'
        "esac\n"
        "exit 0\n"
    )
    (directory / "python3").chmod(0o755)


def test_missing_venv_module_is_reported(tmp_path: Path, root: Path) -> None:
    path = bin_dir(tmp_path, without=("python3",))
    write_fake_python(Path(path), version_ok=True, venv_ok=False)
    result = run("--service-user", USER, root=root, path=path)
    assert result.returncode == 1 and "missing prerequisites: python3-venv" in result.stderr


def test_an_old_python_is_reported_with_its_version(tmp_path: Path, root: Path) -> None:
    path = bin_dir(tmp_path, without=("python3",))
    write_fake_python(Path(path), version_ok=False, venv_ok=True)
    result = run("--service-user", USER, root=root, path=path)
    assert result.returncode == 1 and "python3 >= 3.10 is required (found 3.8)" in result.stderr


def test_no_python_at_all_lists_both_packages(tmp_path: Path, root: Path) -> None:
    result = run("--service-user", USER, root=root, path=bin_dir(tmp_path, without=("python3",)))
    assert "missing prerequisites: python3 python3-venv" in result.stderr


def test_install_packages_only_prints_apt_get_under_dry_run(tmp_path: Path) -> None:
    result = run(
        "--service-user",
        USER,
        "--dry-run",
        "--install-packages",
        path=bin_dir(tmp_path, without=("git",)),
    )
    assert result.returncode == 0
    assert (
        "would run: apt-get install -y git" in result.stdout
        or "apt-get install -y" in result.stdout
    )


def test_install_packages_cannot_be_combined_with_a_staging_root(root: Path) -> None:
    result = install(root, "--install-packages")
    assert result.returncode == 1 and "cannot be combined with --root" in result.stderr


@pytest.mark.parametrize("name", ["no-such-user-xyz", "Bad User", "1abc", "a;b"])
def test_an_invalid_or_unknown_service_user_is_refused(root: Path, name: str) -> None:
    result = run("--service-user", name, root=root)
    assert (result.returncode == 1 and not root.exists()) or not (root / "opt").exists()


def test_unknown_options_and_help(root: Path) -> None:
    assert run("--surprise").returncode == 1
    helped = run("--help")
    assert helped.returncode == 0 and "--enable-deadman" in helped.stdout


def test_examples_contain_placeholders_only() -> None:
    for name in ("agent.env.example", "deploy.env.example"):
        text = (DEPLOY / name).read_text()
        assert "://" not in text, f"{name} must not contain a URL value"
        assert "lambda-url" not in text and "arn:" not in text


def test_the_agent_env_example_fails_loudly_until_it_is_edited(tmp_path: Path) -> None:
    config = tmp_path / "agent.env"
    shutil.copy(DEPLOY / "agent.env.example", config)
    config.chmod(0o600)
    with pytest.raises(ConfigError) as caught:
        load_settings(
            environ={"RENDER_AGENT_CONFIG_FILE": str(config)},
            cwd=tmp_path,
            ssm_fetch=lambda prefix, region: {},
        )
    assert dict(caught.value.problems)["AGENT_ID"] == "string_pattern_mismatch"


def test_the_deploy_env_example_is_all_comments_so_defaults_apply() -> None:
    from bootstrap import parse_env_file

    assert parse_env_file((DEPLOY / "deploy.env.example").read_text()) == {}


def test_install_script_is_shellcheck_clean() -> None:
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck is not installed (it is a dev dependency)")
    scripts = [INSTALL, DEPLOY.parent / "scripts" / "export_requirements.sh"]
    result = subprocess.run(
        [shellcheck, *map(str, scripts)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout
