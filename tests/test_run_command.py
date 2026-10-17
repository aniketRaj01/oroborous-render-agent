from __future__ import annotations

import io
import os
import signal
from pathlib import Path

import httpx
import pytest

from conftest import valid_env
from fakes.clock import FakeClock
from fakes.fake_model_server import Reject
from render_agent.__main__ import main
from render_agent.model_server.models import GenerateRequest
from rig import Rig, make_rig


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return make_rig(tmp_path)


def env_for(rig: Rig, **extra: str) -> dict[str, str]:
    return {**valid_env(), "SCRATCH_DIR": str(rig.scratch), **extra}


def run_main(
    rig: Rig, argv: list[str] | None = None, env: dict[str, str] | None = None
) -> tuple[int, str]:
    out = io.StringIO()
    code = main(
        ["run", *(argv or [])],
        environ=env if env is not None else env_for(rig),
        out=out,
        clock=rig.clock,
        transport=rig.fake_api.transport,
        model_server=rig.model,
        uploader=rig.uploader,
    )
    return code, out.getvalue()


def test_exit_0_when_the_work_is_done_and_the_queue_is_idle(
    rig: Rig, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs = rig.fake_api.add_plan(1, clips=2).jobs
    code, _ = run_main(rig)
    assert code == 0 and [j.status for j in jobs] == ["done", "done"]
    err = capsys.readouterr().err
    assert "msg=run_summary" in err and "done=2" in err and "reason=idle" in err
    assert "fake prompt" not in err and "fake-id" not in err


def test_max_clips_option(rig: Rig) -> None:
    jobs = rig.fake_api.add_plan(1, clips=3).jobs
    code, _ = run_main(rig, ["--max-clips", "1"])
    assert code == 0 and [j.status for j in jobs] == ["done", "pending", "pending"]


def test_max_clips_must_be_positive(rig: Rig) -> None:
    with pytest.raises(SystemExit) as caught:
        run_main(rig, ["--max-clips", "0"])
    assert caught.value.code == 2


def test_exit_2_on_configuration_errors_without_values(rig: Rig) -> None:
    code, output = run_main(rig, env={"JOB_API_URL": "http://topsecret1.invalid"})
    assert code == 2 and "AGENT_ID: missing" in output and "topsecret1" not in output
    assert rig.fake_api.requests == []


def test_exit_2_when_the_model_server_url_is_not_loopback(rig: Rig) -> None:
    code, output = run_main(rig, env=env_for(rig, MODEL_SERVER_URL="http://example.invalid:8000"))
    assert code == 2 and "MODEL_SERVER_URL: model_server_not_loopback" in output


def test_exit_2_when_the_scratch_dir_is_unusable(rig: Rig, tmp_path: Path) -> None:
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")
    code, output = run_main(rig, env=env_for(rig, SCRATCH_DIR=str(not_a_dir)))
    assert code == 2 and "SCRATCH_DIR" in output


def test_exit_3_on_a_job_api_auth_error(rig: Rig) -> None:
    rig.fake_api.inject_fault("/v1/claim", httpx.Response(403))
    assert run_main(rig)[0] == 3


def test_exit_4_on_unexpected_errors_with_a_redacted_traceback(
    rig: Rig, capsys: pytest.CaptureFixture[str]
) -> None:
    rig.fake_api.add_plan(1, clips=1)

    def explode(request: GenerateRequest) -> None:
        raise RuntimeError("cannot reach https://leakyid.lambda-url.us-east-1.on.aws now")

    rig.model.on_generate = explode
    code, _ = run_main(rig)
    err = capsys.readouterr().err
    assert code == 4
    assert "run_unexpected_error" in err and "RuntimeError" in err and "Traceback" in err
    assert "leakyid" not in err and "reason=unexpected_error" in err


def test_exit_5_when_the_model_server_never_becomes_ready(rig: Rig) -> None:
    rig.model.not_ready_polls = 10_000
    code, _ = run_main(rig, env=env_for(rig, MODEL_SERVER_READY_TIMEOUT_MINUTES="1"))
    assert code == 5


def test_exit_5_on_consecutive_failures(rig: Rig) -> None:
    for plan_id in range(1, 5):
        rig.fake_api.add_plan(plan_id, clips=1)
    rig.model.default(Reject("render_error"))
    assert run_main(rig)[0] == 5


def test_sigterm_stops_cleanly_restores_handlers_and_leaves_the_job_for_resume(
    rig: Rig, capsys: pytest.CaptureFixture[str]
) -> None:
    job = rig.fake_api.add_plan(1, clips=2).jobs[0]
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)

    def deliver_sigterm(request: GenerateRequest) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    rig.model.on_generate = deliver_sigterm
    code, _ = run_main(rig)
    err = capsys.readouterr().err
    assert code == 0
    assert "signal_received" in err and "signal=SIGTERM" in err and "reason=signal" in err
    assert job.status == "generating" and "/v1/fail" not in [r.path for r in rig.fake_api.requests]
    assert rig.leftover_files() == []
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def test_sigint_is_handled_like_sigterm(rig: Rig) -> None:
    rig.fake_api.add_plan(1, clips=2)
    rig.model.on_generate = lambda request: os.kill(os.getpid(), signal.SIGINT)
    assert run_main(rig)[0] == 0


def test_check_command_is_unchanged(rig: Rig) -> None:
    out = io.StringIO()
    code = main(["check", "--dry-run"], environ=env_for(rig), out=out, clock=FakeClock())
    assert code == 0 and "dry run" in out.getvalue()
