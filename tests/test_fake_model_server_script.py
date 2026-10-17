"""Runs scripts/fake_model_server.py on an ephemeral loopback port and drives a clip through it."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from conftest import build_client
from render_agent.model_server.client import create_model_server
from render_agent.runner import Runner
from rig import Rig, make_rig

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fake_model_server.py"


@contextmanager
def fake_server(**env: str) -> Iterator[tuple[int, dict[str, str]]]:
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "FAKE_MODEL_DELAY_SECONDS": "0", **env},
    )
    output: dict[str, str] = {}
    try:
        assert process.stdout is not None
        line = process.stdout.readline()
        assert line.startswith("listening on 127.0.0.1:"), line
        yield int(line.strip().rsplit(":", 1)[1]), output
    finally:
        process.terminate()
        output["stderr"] = process.communicate(timeout=10)[1]


def runner_for(rig: Rig, port: int) -> Runner:
    settings = rig.settings.model_copy(update={"model_server_url": f"http://127.0.0.1:{port}"})
    return Runner(
        settings=settings,
        job_api=build_client(rig.fake_api, rig.clock),
        model_server=create_model_server(settings),
        uploader=rig.uploader,
        clock=rig.clock,
        max_clips=1,
    )


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return make_rig(tmp_path)


def test_one_clip_end_to_end_through_the_script(rig: Rig) -> None:
    job = rig.fake_api.add_plan(1, clips=1).jobs[0]
    with fake_server() as (port, output):
        result = runner_for(rig, port).run(rig.clock.now())
    assert f"generated job {job.job_id}" in output["stderr"]
    assert job.prompt not in output["stderr"]
    assert (result.done, result.exit_code) == (1, 0)
    assert job.status == "done"
    assert rig.uploader.uploads[0].size > 20  # the fake file: ftyp box plus filler
    assert rig.leftover_files() == []


def test_the_script_announces_that_it_is_fake(rig: Rig) -> None:
    with fake_server() as (_, output):
        pass
    assert "FAKE model server" in output["stderr"] and "not playable" in output["stderr"]


def test_injected_failures_surface_as_the_contract_error_codes(rig: Rig) -> None:
    job = rig.fake_api.add_plan(1, clips=1).jobs[0]
    with fake_server(FAKE_MODEL_FAIL_EVERY="1", FAKE_MODEL_FAIL_CODE="oom") as (port, _):
        result = runner_for(rig, port).run(rig.clock.now())
    assert (result.failed, result.done) == (1, 0)
    assert job.last_fail_code == "oom"


def test_health_reports_not_ready_until_the_configured_time(rig: Rig) -> None:
    with fake_server(FAKE_MODEL_READY_AFTER_SECONDS="0.5") as (port, _):
        server = create_model_server(
            rig.settings.model_copy(update={"model_server_url": f"http://127.0.0.1:{port}"})
        )
        assert not server.is_ready()
        deadline = time.monotonic() + 10
        while not server.is_ready() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert server.is_ready()
