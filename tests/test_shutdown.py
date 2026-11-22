"""Power off after an idle exit: only the idle exit, only when opted in, never against a guard."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from render_agent import shutdown
from render_agent.runner import RunResult
from rig import Rig, make_rig

HELPER = Path(__file__).resolve().parents[1] / "deploy" / "poweroff-if-requested.sh"


class Requests:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self._fail = fail

    def __call__(self) -> None:
        self.calls += 1
        if self._fail:
            raise PermissionError("denied")


def outcome(rig: Rig, result: RunResult, requests: Requests, guard: Path) -> str:
    return shutdown.request_poweroff_after_idle(
        result,
        settings=rig.settings,
        model_server=rig.model,
        clock=rig.clock,
        should_stop=lambda: rig.stop,
        request=requests,
        guard_file=guard,
    )


@pytest.fixture
def guard(tmp_path: Path) -> Path:
    return tmp_path / "no-shutdown"


def test_flag_off_by_default_never_powers_off(tmp_path: Path, guard: Path) -> None:
    rig, requests = make_rig(tmp_path), Requests()
    result = rig.run()
    assert result.reason == "idle"
    assert outcome(rig, result, requests, guard) == shutdown.DISABLED and requests.calls == 0


def test_idle_exit_with_the_flag_on_waits_the_grace_then_requests_once(
    tmp_path: Path, guard: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rig = make_rig(tmp_path, AUTO_SHUTDOWN_ON_IDLE="true", AUTO_SHUTDOWN_GRACE_SECONDS="7")
    requests = Requests()
    result = rig.run()
    rig.clock.sleeps.clear()
    with caplog.at_level("INFO", logger="render_agent"):
        assert outcome(rig, result, requests, guard) == shutdown.REQUESTED
    assert requests.calls == 1 and rig.clock.sleeps == [7.0]
    assert "idle exit: powering off instance" in caplog.text


def test_guard_file_and_in_flight_work_keep_the_instance_up(tmp_path: Path, guard: Path) -> None:
    rig, requests = make_rig(tmp_path, AUTO_SHUTDOWN_ON_IDLE="true"), Requests()
    result = rig.run()
    guard.touch()
    assert outcome(rig, result, requests, guard) == shutdown.GUARD_FILE
    guard.unlink()
    (rig.scratch / "12-1.mp4").write_bytes(b"x")  # the model server may still be writing it
    assert outcome(rig, result, requests, guard) == shutdown.WORK_IN_FLIGHT
    (rig.scratch / "12-1.mp4").unlink()
    guard_created_during_grace = rig.clock
    guard_created_during_grace.on_sleep = guard.touch
    assert outcome(rig, result, requests, guard) == shutdown.GUARD_FILE
    assert requests.calls == 0


def test_only_the_idle_exit_and_a_clean_stop_never_power_off(tmp_path: Path, guard: Path) -> None:
    rig, requests = make_rig(tmp_path, AUTO_SHUTDOWN_ON_IDLE="true"), Requests()
    rig.stop = True
    result = rig.run()
    assert result.reason == "signal"
    assert outcome(rig, result, requests, guard) == shutdown.NOT_IDLE
    rig.stop = False
    idle = rig.run()
    stop_during_grace = rig.clock
    stop_during_grace.on_sleep = lambda: setattr(rig, "stop", True)
    assert outcome(rig, idle, requests, guard) == shutdown.STOP_REQUESTED
    assert requests.calls == 0


def test_a_failed_request_is_logged_and_does_not_raise(tmp_path: Path, guard: Path) -> None:
    rig = make_rig(tmp_path, AUTO_SHUTDOWN_ON_IDLE="true")
    requests = Requests(fail=True)
    assert outcome(rig, rig.run(), requests, guard) == shutdown.REQUEST_FAILED
    assert requests.calls == 1


def test_a_bootstrap_fallback_keeps_the_instance_up(tmp_path: Path, guard: Path) -> None:
    rig = make_rig(tmp_path, AUTO_SHUTDOWN_ON_IDLE="true", AGENT_BOOTSTRAP_FALLBACK="true")
    requests = Requests()
    assert outcome(rig, rig.run(), requests, guard) == shutdown.BOOTSTRAP_FALLBACK
    assert requests.calls == 0


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_root_helper_powers_off_only_for_a_marker_after_a_normal_exit(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "calls"
    # The helper calls an absolute systemctl path; run a copy that points at a stub instead.
    stub = tmp_path / "systemctl"
    stub.write_text(f'#!/bin/sh\necho "$@" >> {log}\n')
    stub.chmod(0o755)
    script = tmp_path / "helper.sh"
    script.write_text(HELPER.read_text().replace("/usr/bin/systemctl", str(stub)))
    marker, guard = tmp_path / "marker", tmp_path / "guard"

    def run(*, result: str = "success", status: str = "0", has_marker: bool = True) -> None:
        if has_marker:
            marker.write_text("idle\n")
        env = {"PATH": os.environ["PATH"], "SERVICE_RESULT": result, "EXIT_STATUS": status}
        done = subprocess.run(
            ["bash", str(script), str(marker), str(guard)], env=env, check=False, timeout=30
        )
        assert done.returncode == 0 and not marker.exists()

    run(has_marker=False)
    run(result="exit-code", status="2")
    guard.touch()
    run()
    assert not log.exists()
    guard.unlink()
    run()
    assert log.read_text() == "poweroff\n"
