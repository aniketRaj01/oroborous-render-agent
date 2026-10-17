"""The real `python -m ltx_server` process: signals, exit codes, and configuration errors."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from ms_helpers import generate_body, request, wait_until

SERVER_DIR = Path(__file__).resolve().parents[2] / "model_server"


@contextmanager
def server_process(tmp_path: Path, **env: str) -> Iterator[tuple[subprocess.Popen[str], int, Path]]:
    out = tmp_path / "scratch"
    out.mkdir(exist_ok=True)
    environment = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("FAKE_", "LTX_"))},
        "PYTHONPATH": str(SERVER_DIR),
        "RENDERER": "fake",
        "PORT": "0",
        "ALLOWED_OUTPUT_DIR": str(out),
        "SHUTDOWN_GRACE_SECONDS": "5",
        "FAKE_DELAY_SECONDS": "0",
        "PYTHONFAULTHANDLER": "1",
        **env,
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "ltx_server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    try:
        assert process.stdout is not None
        line = process.stdout.readline()
        assert line.startswith("listening on 127.0.0.1:"), line
        yield process, int(line.strip().rsplit(":", 1)[1]), out
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def exit_code(process: subprocess.Popen[str], timeout: float) -> int:
    """The exit status; on a hang, abort the server so faulthandler prints every thread's stack."""
    try:
        return process.wait(timeout)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGABRT)
        _, stderr = process.communicate(timeout=10)
        raise AssertionError(
            f"the server did not exit within {timeout} s; its threads:\n{stderr[-4000:]}"
        ) from None


def ready(port: int) -> bool:
    try:
        return bool(request(port, "GET", "/health", timeout=2)[1]["modelLoaded"])
    except OSError:
        return False


def refused(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    except OSError:
        return True
    return False


def post_in_thread(port: int, out: str, results: list[Any]) -> threading.Thread:
    def work() -> None:
        try:
            results.append(request(port, "POST", "/generate", generate_body(out), timeout=30)[0])
        except OSError as exc:
            results.append(type(exc).__name__)

    thread = threading.Thread(target=work)
    thread.start()
    return thread


def test_sigterm_while_idle_exits_cleanly_and_quickly(tmp_path: Path) -> None:
    with server_process(tmp_path) as (process, port, _):
        assert wait_until(lambda: ready(port))
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        assert exit_code(process, 10) == 0 and time.monotonic() - started < 5
        _, stderr = process.communicate()
    assert "event=signal_received" in stderr and "event=stopped" in stderr


def test_sigterm_during_load_exits_without_waiting_for_the_model(tmp_path: Path) -> None:
    with server_process(tmp_path, FAKE_LOAD_SECONDS="60") as (process, port, _):
        assert not ready(port)
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        assert exit_code(process, 10) == 0 and time.monotonic() - started < 5


def test_a_running_generation_that_ends_within_the_grace_period_completes(tmp_path: Path) -> None:
    with server_process(tmp_path, FAKE_DELAY_SECONDS="1.5", SHUTDOWN_GRACE_SECONDS="10") as (
        process,
        port,
        out,
    ):
        assert wait_until(lambda: ready(port))
        results: list[Any] = []
        target = str(out / "1-1.mp4")
        thread = post_in_thread(port, target, results)
        time.sleep(0.5)  # the fake render is under way
        process.send_signal(signal.SIGTERM)
        assert wait_until(lambda: refused(port), 5)  # no new connections are accepted
        thread.join(15)
        assert exit_code(process, 15) == 0
    assert results == [200] and Path(target).read_bytes()[4:8] == b"ftyp"
    assert sorted(p.name for p in out.iterdir()) == ["1-1.mp4"]


def test_a_generation_that_outlasts_the_grace_period_is_abandoned_and_leaves_no_final_file(
    tmp_path: Path,
) -> None:
    with server_process(tmp_path, FAKE_DELAY_SECONDS="20", SHUTDOWN_GRACE_SECONDS="0.5") as (
        process,
        port,
        out,
    ):
        assert wait_until(lambda: ready(port))
        results: list[Any] = []
        target = str(out / "1-1.mp4")
        thread = post_in_thread(port, target, results)
        time.sleep(0.5)
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        assert exit_code(process, 15) == 0 and time.monotonic() - started < 8
        thread.join(15)
        _, stderr = process.communicate()
    assert not Path(target).exists()
    assert "event=stopped_during_generation" in stderr
    assert results and results[0] != 200


def test_consecutive_ooms_make_the_process_exit_with_a_non_zero_status(tmp_path: Path) -> None:
    with server_process(
        tmp_path, FAKE_FAIL_EVERY="1", FAKE_FAIL_KIND="oom", MAX_CONSECUTIVE_OOMS="2"
    ) as (
        process,
        port,
        out,
    ):
        assert wait_until(lambda: ready(port))
        first = request(port, "POST", "/generate", generate_body(str(out / "1-1.mp4")))
        second = request(port, "POST", "/generate", generate_body(str(out / "2-1.mp4")))
        assert first[0] == second[0] == 500 and second[1]["error"]["code"] == "oom"
        assert exit_code(process, 15) == 75
        _, stderr = process.communicate()
    assert "event=too_many_oom_exiting" in stderr


@pytest.mark.parametrize(
    ("env", "named"),
    [
        ({"HOST": "0.0.0.0"}, "HOST"),
        ({"PORT": "http"}, "PORT"),
        ({"RENDERER": "gpu"}, "RENDERER"),
        ({"OFFLOAD_MODE": "ram"}, "OFFLOAD_MODE"),
        ({"DEFAULT_WIDTH": "100"}, "DEFAULT_WIDTH"),
    ],
)
def test_bad_configuration_exits_2_naming_the_variable_but_not_its_value(
    tmp_path: Path, env: dict[str, str], named: str
) -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": str(SERVER_DIR),
        "RENDERER": "fake",
        "PORT": "0",
        **env,
    }
    result = subprocess.run(
        [sys.executable, "-m", "ltx_server"],
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
        cwd=tmp_path,
    )
    assert result.returncode == 2 and named in result.stderr
    for value in env.values():
        assert f"={value}" not in result.stderr and f" {value}" not in result.stderr.replace(
            named, ""
        )


def test_unexpected_arguments_exit_2(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ltx_server", "--port", "1"],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(SERVER_DIR)},
        cwd=tmp_path,
    )
    assert result.returncode == 2
