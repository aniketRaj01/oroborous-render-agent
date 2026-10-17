from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from unit_files import Unit, load_unit, parse_unit

UNITS = Path(__file__).resolve().parent.parent / "deploy" / "systemd"
SERVICE = UNITS / "render-agent.service"
DEADMAN_SERVICE = UNITS / "render-agent-deadman.service"
DEADMAN_TIMER = UNITS / "render-agent-deadman.timer"

SENSITIVE = [
    r"https?://",
    r"\bi-[0-9a-f]{8,}\b",
    r"\b\d{12}\b",
    r"arn:aws",
    r"AKIA[0-9A-Z]{16}",
    r"\b\d{1,3}(\.\d{1,3}){3}\b",
    r"lambda-url",
    r"(?i)(secret|password|token)\s*=",
]


@pytest.fixture(scope="module")
def service() -> Unit:
    return load_unit(SERVICE)


def only(unit: Unit, section: str, key: str) -> str:
    values = unit[section][key]
    assert len(values) == 1, f"{key} should be set exactly once"
    return values[0]


def test_the_parser_handles_repeated_keys_and_rejects_stray_lines() -> None:
    parsed = parse_unit("[A]\nx=1\nx=2\n# c\n")
    assert parsed == {"A": {"x": ["1", "2"]}}
    with pytest.raises(ValueError, match="outside a section"):
        parse_unit("x=1\n")


def test_ordering_and_dependencies(service: Unit) -> None:
    assert only(service, "Unit", "After").split() == ["network-online.target", "ltx-server.service"]
    wants = only(service, "Unit", "Wants").split()
    assert "network-online.target" in wants and "ltx-server.service" in wants


def test_the_service_runs_as_the_install_time_user(service: Unit) -> None:
    assert only(service, "Service", "User") == "@SERVICE_USER@"


def test_environment_files_are_optional_and_the_version_file_comes_last(service: Unit) -> None:
    assert service["Service"]["EnvironmentFile"] == [
        "-/etc/render-agent/agent.env",
        "-/run/render-agent/version.env",
    ]
    assert only(service, "Service", "RuntimeDirectory") == "render-agent"


def test_the_release_source_tree_is_on_the_python_path(service: Unit) -> None:
    assert only(service, "Service", "Environment") == "PYTHONPATH=/opt/render-agent/current/src"


def test_the_working_directory_keeps_a_stray_dotenv_out(service: Unit) -> None:
    assert only(service, "Service", "WorkingDirectory") == "/var/lib/render-agent"
    assert not any("/opt/render-agent" in v for v in service["Service"]["WorkingDirectory"])


def test_bootstrap_runs_before_the_agent_with_room_for_a_first_venv_build(service: Unit) -> None:
    assert only(service, "Service", "ExecStartPre") == (
        "/usr/bin/python3 /opt/render-agent/bootstrap.py"
    )
    assert only(service, "Service", "ExecStart") == (
        "/opt/render-agent/current-venv/bin/python -m render_agent run"
    )
    start_timeout = int(only(service, "Service", "TimeoutStartSec"))
    assert start_timeout >= 600  # the bootstrap budget is 300 s; a first venv build needs room


def test_restart_policy_cannot_spin_and_skips_hopeless_exit_codes(service: Unit) -> None:
    assert only(service, "Service", "Restart") == "on-failure"
    assert only(service, "Service", "RestartSec") == "30"
    assert only(service, "Service", "RestartPreventExitStatus").split() == ["2", "3", "5"]
    assert int(only(service, "Unit", "StartLimitIntervalSec")) >= 600
    assert 1 <= int(only(service, "Unit", "StartLimitBurst")) <= 10


def test_stop_timeout_is_chosen_not_defaulted(service: Unit) -> None:
    stop_timeout = int(only(service, "Service", "TimeoutStopSec"))
    assert 60 <= stop_timeout <= 300  # not the 90 s default by accident, not a 20 minute render
    comment_block = SERVICE.read_text()
    assert "TimeoutStopSec" in comment_block and "resumes it" in comment_block


def test_hardening_that_is_safe_is_on_and_private_tmp_stays_off(service: Unit) -> None:
    assert only(service, "Service", "NoNewPrivileges") == "yes"
    assert "PrivateTmp" not in service["Service"]
    text = SERVICE.read_text()
    assert "PrivateTmp is deliberately NOT enabled" in text and "model server" in text


def test_the_service_starts_at_boot(service: Unit) -> None:
    assert only(service, "Install", "WantedBy") == "multi-user.target"


def test_the_service_does_not_power_the_machine_off() -> None:
    text = SERVICE.read_text()
    assert "SuccessAction" not in text and "poweroff" not in text


def test_deadman_timer_fires_65_minutes_after_boot_and_is_not_self_enabling() -> None:
    timer = load_unit(DEADMAN_TIMER)
    assert only(timer, "Timer", "OnBootSec") == "65min"
    assert only(timer, "Timer", "Unit") == "render-agent-deadman.service"
    assert only(timer, "Install", "WantedBy") == "timers.target"
    assert "Opt-in" in only(timer, "Unit", "Description") or "opt-in" in only(
        timer, "Unit", "Description"
    )


def test_deadman_service_powers_off_unless_the_opt_out_file_exists() -> None:
    deadman = load_unit(DEADMAN_SERVICE)
    assert only(deadman, "Unit", "ConditionPathExists") == "!/etc/render-agent/no-shutdown"
    assert only(deadman, "Service", "Type") == "oneshot"
    assert only(deadman, "Service", "ExecStart") == "/usr/bin/systemctl poweroff"
    assert "Install" not in deadman  # cannot be enabled on its own


def test_the_deadman_is_documented_as_opt_in() -> None:
    text = DEADMAN_TIMER.read_text()
    assert "Not enabled by default" in text and "no-shutdown" in text and "--enable-deadman" in text


@pytest.mark.parametrize("path", [SERVICE, DEADMAN_SERVICE, DEADMAN_TIMER], ids=lambda p: p.name)
def test_no_secrets_or_real_identifiers_in_unit_files(path: Path) -> None:
    text = path.read_text()
    for pattern in SENSITIVE:
        assert re.search(pattern, text) is None, f"{path.name} matches {pattern}"


BENIGN = ("is not executable", "No such file or directory", "not found", "Failed to load unit")
FATAL = ("Unknown key", "Unknown section", "Unknown lvalue", "outside of section", "Invalid")


def test_systemd_analyze_accepts_the_unit_files(tmp_path: Path) -> None:
    analyzer = shutil.which("systemd-analyze")
    if analyzer is None:
        pytest.skip("systemd-analyze is not installed here; CI runs this check on Linux")
    staged = []
    for path in (SERVICE, DEADMAN_SERVICE, DEADMAN_TIMER):
        target = tmp_path / path.name
        target.write_text(path.read_text().replace("@SERVICE_USER@", "root"))
        staged.append(str(target))
    result = subprocess.run(
        [analyzer, "verify", *staged], capture_output=True, text=True, timeout=60, check=False
    )
    problems = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if any(marker in line for marker in FATAL)
    ]
    assert problems == [], "\n".join(problems)
    unexplained = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if line.strip() and not any(marker in line for marker in BENIGN)
    ]
    assert result.returncode == 0 or unexplained == [], "\n".join(unexplained)
