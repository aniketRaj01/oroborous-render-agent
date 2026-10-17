from __future__ import annotations

import io
import os
from pathlib import Path

import httpx
import pytest

from conftest import FAKE_AGENT, FAKE_URL, valid_env
from fakes.clock import FakeClock
from fakes.fake_job_api import FakeJobApi
from render_agent.cli import main

SECRET_URL = "https://leakyid.lambda-url.us-east-1.on.aws"


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def run(
    argv: list[str],
    env: dict[str, str],
    clock: FakeClock,
    transport: httpx.BaseTransport | None = None,
) -> tuple[int, str]:
    out = io.StringIO()
    code = main(argv, environ=env, out=out, clock=clock, transport=transport)
    return code, out.getvalue()


def unsigned_env() -> dict[str, str]:
    return {
        "JOB_API_URL": "http://127.0.0.1:8123",
        "JOB_API_SIGNING": "none",
        "AGENT_ID": FAKE_AGENT,
    }


def test_dry_run_prints_a_redacted_summary_and_calls_nothing(clock: FakeClock) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("dry run must not call the API")

    env = {**valid_env(), "JOB_API_URL": SECRET_URL}
    code, output = run(["check", "--dry-run"], env, clock, httpx.MockTransport(refuse))
    assert code == 0
    assert "leakyid" not in output and SECRET_URL not in output
    for expected in (
        FAKE_AGENT,
        "signing:       sigv4",
        "us-east-1",
        "<id>.lambda-url.us-east-1.on.aws",
        "env",
    ):
        assert expected in output
    assert "dry run" in output


def test_full_check_makes_one_claim_with_max_plans_one_and_prints_the_result(
    clock: FakeClock, fake_api: FakeJobApi
) -> None:
    fake_api.add_plan(253, clips=2)
    fake_api.add_plan(254, clips=2)
    code, output = run(["check", "--lease-minutes", "3"], unsigned_env(), clock, fake_api.transport)
    assert code == 0
    assert len(fake_api.calls("/v1/claim")) == 1
    import json

    body = json.loads(fake_api.calls("/v1/claim")[0].body)
    assert body["maxPlans"] == 1 and body["deadlineAt"] == "2026-10-18T06:03:00Z"
    assert "totals:           plans=1 clips=2" in output
    assert "plan 253" in output and "plan 254" not in output
    assert "position 0" in output and "position 1" in output and "resumed false" in output
    assert "fake prompt" not in output
    assert "fake-clip-bucket" not in output and "plans/253" not in output
    assert "WARNING" in output and FAKE_AGENT in output and "2026-10-18T06:08:00Z" in output
    assert output.rstrip().endswith("immediately.") or "leased to" in output.splitlines()[-1]


def test_check_with_no_work_says_nothing_was_leased(clock: FakeClock, fake_api: FakeJobApi) -> None:
    code, output = run(["check"], unsigned_env(), clock, fake_api.transport)
    assert code == 0 and "Nothing was leased" in output


def test_resumed_clips_are_flagged(clock: FakeClock, fake_api: FakeJobApi) -> None:
    fake_api.add_plan(253, clips=1)
    run(["check"], unsigned_env(), clock, fake_api.transport)
    _, output = run(["check"], unsigned_env(), clock, fake_api.transport)
    assert "resumed true" in output


def test_config_error_exits_2_without_values(clock: FakeClock) -> None:
    code, output = run(["check", "--dry-run"], {"JOB_API_URL": "http://topsecret1.invalid"}, clock)
    assert code == 2
    assert "AGENT_ID: missing" in output and "JOB_API_URL" in output
    assert "topsecret1" not in output


def test_api_errors_exit_3(clock: FakeClock, fake_api: FakeJobApi) -> None:
    fake_api.inject_fault("/v1/claim", httpx.Response(403, json={"Message": "no"}))
    code, output = run(["check"], unsigned_env(), clock, fake_api.transport)
    assert code == 3 and "AuthError" in output and "127.0.0.1" not in output


def test_transient_api_errors_also_exit_3(clock: FakeClock, fake_api: FakeJobApi) -> None:
    fake_api.inject_fault("/v1/claim", *[httpx.Response(503)] * 5)
    code, _ = run(["check"], unsigned_env(), clock, fake_api.transport)
    assert code == 3


def test_unexpected_errors_exit_4_and_show_only_the_type(clock: FakeClock) -> None:
    def explode(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("secret detail leakyid")

    code, output = run(["check"], unsigned_env(), clock, httpx.MockTransport(explode))
    assert code == 4 and "RuntimeError" in output and "leakyid" not in output


def test_lease_minutes_is_range_checked(clock: FakeClock) -> None:
    with pytest.raises(SystemExit) as caught:
        run(["check", "--lease-minutes", "0"], unsigned_env(), clock)
    assert caught.value.code == 2


def test_the_real_environment_is_not_touched_by_tests() -> None:
    assert "JOB_API_URL" not in os.environ or os.environ["JOB_API_URL"] != FAKE_URL
