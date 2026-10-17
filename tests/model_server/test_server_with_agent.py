"""The agent's real Runner against the new model server (FakeRenderer behind it), unchanged."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import build_client
from ltx_server.renderer import FakeRenderer
from ms_helpers import make_settings, running
from render_agent.model_server.client import create_model_server
from render_agent.runner import Runner, RunResult
from rig import Rig, make_rig


def run_agent(rig: Rig, port: int, max_clips: int | None = None) -> RunResult:
    settings = rig.settings.model_copy(update={"model_server_url": f"http://127.0.0.1:{port}"})
    runner = Runner(
        settings=settings,
        job_api=build_client(rig.fake_api, rig.clock),
        model_server=create_model_server(settings),
        uploader=rig.uploader,
        clock=rig.clock,
        max_clips=max_clips,
    )
    return runner.run(rig.clock.now())


def server_settings(rig: Rig, tmp_path: Path):  # type: ignore[no-untyped-def]
    # The agent and the server share one directory, as on the instance.
    return make_settings(tmp_path, allowed_output_dir=rig.scratch)


def test_three_clips_flow_from_claim_to_complete(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    plan = rig.fake_api.add_plan(1, clips=3)
    renderer = FakeRenderer()
    with running(server_settings(rig, tmp_path), renderer) as server:
        result = run_agent(rig, server.port, max_clips=3)
    assert (result.done, result.failed, result.exit_code) == (3, 0, 0)
    assert [job.status for job in plan.jobs] == ["done", "done", "done"]
    assert renderer.renders == 3 and renderer.after_clips == 3
    assert [r.job_id for r in renderer.requests] == [job.job_id for job in plan.jobs]
    assert len(rig.uploader.uploads) == 3 and all(u.size > 0 for u in rig.uploader.uploads)
    assert rig.leftover_files() == []  # the agent deleted its copies; the server left no temp files
    body = json.loads(rig.fake_api.calls("/v1/complete")[0].body)
    assert body["render"]["width"] == 576 and body["render"]["numFrames"] == 361
    assert body["render"]["hasAudio"] is True and body["upload"]["sizeBytes"] > 0


def test_params_from_the_job_reach_the_renderer(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    (job,) = rig.fake_api.add_plan(1, clips=1).jobs
    job.params = {"width": 320, "height": 576, "numFrames": 121, "fps": 24, "seed": 9}
    renderer = FakeRenderer()
    with running(server_settings(rig, tmp_path), renderer) as server:
        run_agent(rig, server.port, max_clips=1)
    sent = renderer.requests[0]
    assert (sent.width, sent.height, sent.num_frames, sent.fps, sent.seed) == (
        320,
        576,
        121,
        24.0,
        9,
    )
    assert sent.prompt == job.prompt


def test_a_job_without_params_gets_the_server_defaults(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    (job,) = rig.fake_api.add_plan(1, clips=1).jobs
    job.params = None
    renderer = FakeRenderer()
    with running(server_settings(rig, tmp_path), renderer) as server:
        result = run_agent(rig, server.port, max_clips=1)
    assert result.done == 1
    sent = renderer.requests[0]
    assert (sent.width, sent.height, sent.num_frames, sent.fps, sent.seed) == (
        576,
        1024,
        361,
        24.0,
        42,
    )


def test_an_oom_on_one_clip_is_reported_as_oom_and_the_run_continues(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    plan = rig.fake_api.add_plan(1, clips=3)
    renderer = FakeRenderer(script={2: "oom"})
    with running(server_settings(rig, tmp_path), renderer) as server:
        result = run_agent(rig, server.port)
        assert server.app.exit_code == 0  # one OOM does not restart the server
    failed = plan.jobs[1]
    assert failed.last_fail_code == "oom"
    # the failed clip is claimed again after the list drains and renders on the retry
    assert result.failed == 1 and result.done == 3 and result.exit_code == 0
    assert [job.status for job in plan.jobs] == ["done", "done", "done"]
    assert renderer.recovers == 1 and renderer.after_clips == renderer.renders == 4


def test_a_render_error_and_a_missing_output_are_reported_as_render_errors(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    plan = rig.fake_api.add_plan(1, clips=2)
    renderer = FakeRenderer(script={1: "mid_write", 2: "no_output"})
    with running(server_settings(rig, tmp_path), renderer) as server:
        run_agent(rig, server.port)
    assert plan.jobs[0].last_fail_code == "render_error"
    assert plan.jobs[1].last_fail_code == "render_error"
    assert rig.leftover_files() == []


def test_two_ooms_in_a_row_stop_the_server_and_the_agent_stops_cleanly(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    rig.fake_api.add_plan(1, clips=3)
    renderer = FakeRenderer(fail_every=1, fail_kind="oom")
    with running(server_settings(rig, tmp_path), renderer) as server:
        result = run_agent(rig, server.port)
        assert server.app.exit_code == 75
    # the agent's own consecutive-failure rule or the dead server ends the run without a crash
    assert result.exit_code == 5 and result.reason == "consecutive_failures"
