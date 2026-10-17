from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest

from fakes.fake_model_server import (
    BadFile,
    Garbage,
    Outcome,
    Reject,
    Timeout,
    Unavailable,
    WrongPath,
)
from render_agent.uploader import UploadError
from rig import Rig, envelope, make_rig


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return make_rig(tmp_path)


def job_ids(rig: Rig, plan_index: int = 0) -> list[int]:
    return [job.job_id for job in rig.fake_api.plans[plan_index].jobs]


def routes(rig: Rig) -> list[str]:
    return [request.path for request in rig.fake_api.requests]


# --- startup and claim cadence -----------------------------------------------------------------


def test_waits_for_the_model_server_before_the_first_claim(rig: Rig) -> None:
    rig.model.not_ready_polls = 3
    rig.fake_api.add_plan(1, clips=1)
    seen_at_ready: list[int] = []
    original = rig.model.is_ready

    def spy() -> bool:
        ready = original()
        if ready:
            seen_at_ready.append(len(rig.fake_api.requests))
        return ready

    rig.model.is_ready = spy  # type: ignore[method-assign]
    result = rig.run()
    assert rig.clock.sleeps[:3] == [5.0, 5.0, 5.0]
    assert seen_at_ready == [0]  # no request to the Job API before the model was ready
    assert result.done == 1


def test_model_server_never_ready_exits_5_without_claiming(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, MODEL_SERVER_READY_TIMEOUT_MINUTES="1")
    rig.model.not_ready_polls = 10_000
    result = rig.run()
    assert (result.exit_code, result.reason) == (5, "model_server_not_ready")
    assert rig.fake_api.requests == []


def test_clips_run_in_plan_then_position_order_and_claim_carries_deadline(rig: Rig) -> None:
    rig.fake_api.add_plan(11, clips=2)
    rig.fake_api.add_plan(12, clips=2)
    result = rig.run()
    assert result.done == 4 and result.exit_code == 0
    expected = job_ids(rig, 0) + job_ids(rig, 1)
    assert [r.job_id for r in rig.model.requests] == expected
    import json

    claim = json.loads(rig.fake_api.calls("/v1/claim")[0].body)
    assert claim["deadlineAt"] == "2026-10-18T06:55:00Z" and claim["maxPlans"] == 10


def test_claims_again_immediately_when_the_list_drains(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, MAX_PLANS_PER_CLAIM="1", IDLE_EXIT_MINUTES="1")
    for plan_id in (1, 2, 3):
        rig.fake_api.add_plan(plan_id, clips=1)
    result = rig.run()
    assert result.done == 3
    # three claims that return work, then empty ones until the idle limit
    sequence = [r.path for r in rig.fake_api.requests]
    assert sequence[:3] == ["/v1/claim", "/v1/start", "/v1/complete"]
    assert sequence.count("/v1/claim") == 3 + 2
    assert rig.clock.sleeps == [60.0]  # only the idle wait; no waiting between busy claims


def test_empty_claims_wait_60_seconds_and_exit_after_5_idle_minutes(rig: Rig) -> None:
    result = rig.run()
    assert (result.exit_code, result.reason) == (0, "idle")
    assert len(rig.fake_api.calls("/v1/claim")) == 6
    assert rig.clock.sleeps == [60.0] * 5
    assert result.runtime_seconds == 300


def test_finding_work_resets_the_idle_timer(rig: Rig) -> None:
    def add_work_after_two_waits() -> None:
        if len(rig.clock.sleeps) == 2:
            rig.fake_api.add_plan(1, clips=1)

    rig.clock.on_sleep = add_work_after_two_waits
    result = rig.run()
    assert result.done == 1 and result.reason == "idle"
    # 2 empty claims, 1 with work, then a full 5 idle minutes again (6 empty claims)
    assert len(rig.fake_api.calls("/v1/claim")) == 2 + 1 + 6


def test_a_failing_claim_is_treated_like_an_empty_one(rig: Rig) -> None:
    rig.fake_api.inject_fault("/v1/claim", *[httpx.Response(503)] * 5)
    result = rig.run()
    assert result.reason == "idle" and result.exit_code == 0


# --- deadline ----------------------------------------------------------------------------------


def test_a_clip_that_cannot_finish_before_the_deadline_is_not_started(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, RUN_DEADLINE_MINUTES="10")
    plan = rig.fake_api.add_plan(1, clips=3)
    result = rig.run()
    # clip time = 240 render + 10 upload = 250 s; after one clip the estimate is 250 * 1.25
    assert (result.done, result.skipped, result.reason, result.exit_code) == (2, 1, "deadline", 0)
    assert rig.model.requests[-1].job_id == plan.jobs[1].job_id
    assert len(rig.fake_api.calls("/v1/claim")) == 1  # no claim near the deadline
    assert plan.jobs[2].status == "pending"  # left to lease expiry, never started


def test_nothing_is_claimed_when_not_even_one_default_estimate_fits(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, RUN_DEADLINE_MINUTES="4")
    rig.fake_api.add_plan(1, clips=1)
    result = rig.run()
    assert (result.done, result.reason, result.exit_code) == (0, "deadline", 0)
    assert rig.fake_api.requests == []


def test_estimate_uses_the_mean_of_the_last_five_timings(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, RUN_DEADLINE_MINUTES="55")
    rig.fake_api.add_plan(1, clips=1)
    runner = rig.runner()
    assert runner._clip_estimate_seconds() == 300  # setting, before any timing
    runner._timings.extend([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])  # only the last 5 count
    assert runner._clip_estimate_seconds() == pytest.approx(400.0 * 1.25)


# --- failures ----------------------------------------------------------------------------------

FAILURE_CASES: list[tuple[str, Outcome, str]] = [
    ("oom", Reject("oom", "CUDA out of memory"), "oom"),
    ("render_error", Reject("render_error"), "render_error"),
    ("busy", Reject("busy"), "render_error"),
    ("invalid_request", Reject("invalid_request"), "render_error"),
    ("timeout", Timeout(), "render_error"),
    ("unreachable", Unavailable(), "render_error"),
    ("protocol", Garbage(), "render_error"),
    ("wrong_path", WrongPath(), "render_error"),
    ("missing_file", BadFile("missing"), "verify_failed"),
    ("empty_file", BadFile("empty"), "verify_failed"),
    ("not_an_mp4", BadFile("not_mp4"), "verify_failed"),
]


@pytest.mark.parametrize(
    ("name", "outcome", "code"), FAILURE_CASES, ids=[c[0] for c in FAILURE_CASES]
)
def test_render_and_verify_failures_are_reported_with_the_contract_code(
    rig: Rig, name: str, outcome: Outcome, code: str
) -> None:
    plan = rig.fake_api.add_plan(1, clips=1)
    job = plan.jobs[0]
    rig.model.script(job.job_id, outcome)
    result = rig.run(max_clips=1)
    assert (result.failed, result.done) == (1, 0)
    assert job.last_fail_code == code and job.status == "pending"
    assert rig.uploader.uploads == []  # nothing uploaded for a bad render
    assert "/v1/complete" not in routes(rig)  # never report done on faith
    assert rig.leftover_files() == []


def test_upload_failure_is_reported_as_upload_error_and_nothing_is_completed(rig: Rig) -> None:
    job = rig.fake_api.add_plan(1, clips=1).jobs[0]
    rig.uploader.failures.append(UploadError("S3 request failed (AccessDenied)"))
    result = rig.run(max_clips=1)
    assert result.failed == 1 and job.last_fail_code == "upload_error"
    assert "/v1/complete" not in routes(rig) and rig.leftover_files() == []


def test_one_bad_clip_does_not_stop_the_run(rig: Rig) -> None:
    jobs = rig.fake_api.add_plan(1, clips=3).jobs
    rig.model.script(jobs[0].job_id, Reject("render_error"))
    result = rig.run()
    assert result.failed == 1 and result.done == 3  # the failed clip is retried on re-claim
    assert result.exit_code == 0


def test_three_failures_in_a_row_stop_the_run_with_exit_5(rig: Rig) -> None:
    for plan_id in range(1, 6):
        rig.fake_api.add_plan(plan_id, clips=1)
    rig.model.default(Reject("render_error"))
    result = rig.run()
    assert (result.exit_code, result.reason) == (5, "consecutive_failures")
    assert (result.failed, result.done, result.skipped) == (3, 0, 2)
    assert len(rig.fake_api.calls("/v1/fail")) == 3


def test_a_success_resets_the_consecutive_failure_count(rig: Rig) -> None:
    jobs = [rig.fake_api.add_plan(i, clips=1).jobs[0] for i in range(1, 6)]
    for index in (0, 1, 3, 4):
        rig.model.script(jobs[index].job_id, Reject("render_error"))
    result = rig.run()
    assert result.exit_code == 0 and result.reason == "idle"
    assert result.failed == 4 and result.done == 5


def test_an_unreachable_model_server_counts_towards_the_limit(rig: Rig) -> None:
    for plan_id in range(1, 5):
        rig.fake_api.add_plan(plan_id, clips=1)
    rig.model.default(Unavailable())
    result = rig.run()
    assert result.exit_code == 5 and result.failed == 3


def test_the_prompt_is_never_sent_to_the_job_api_or_logged(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    job = rig.fake_api.add_plan(1, clips=1).jobs[0]
    rig.model.script(job.job_id, Reject("render_error", f"failed to render {job.prompt} badly"))
    with caplog.at_level(logging.DEBUG):
        rig.run(max_clips=1)
    assert job.prompt in rig.model.requests[0].prompt
    assert job.prompt not in (job.last_fail_message or "")
    assert "<prompt>" in (job.last_fail_message or "")
    everything = " ".join(
        r.getMessage() + " " + str(sorted(r.__dict__.items())) for r in caplog.records
    )
    assert job.prompt not in everything


def test_model_server_request_has_only_the_params_the_job_has(rig: Rig) -> None:
    jobs = rig.fake_api.add_plan(1, clips=2).jobs
    jobs[0].params = {"width": 320, "seed": 7}
    jobs[1].params = None
    rig.run()
    first, second = rig.model.requests
    assert first.params.model_dump(exclude_none=True) == {"width": 320, "seed": 7}
    assert second.params.model_dump(exclude_none=True) == {}
    assert first.output_path == str(rig.scratch / f"{jobs[0].job_id}-1.mp4")


# --- start errors ------------------------------------------------------------------------------

START_ERRORS = [
    (409, "max_attempts_reached"),
    (409, "wrong_state"),
    (409, "not_your_lease"),
    (409, "lease_expired"),
    (404, "job_not_found"),
]


@pytest.mark.parametrize(("status", "code"), START_ERRORS)
def test_start_errors_drop_the_job_and_never_fail_it(rig: Rig, status: int, code: str) -> None:
    jobs = rig.fake_api.add_plan(1, clips=2).jobs
    rig.fake_api.inject_fault("/v1/start", envelope(status, code))
    result = rig.run()
    assert result.dropped == 1 and result.done == 1
    assert rig.model.requests[0].job_id == jobs[1].job_id  # first clip was skipped, second ran
    assert "/v1/fail" not in routes(rig)
    # claim returns the dropped job again; it is ignored instead of looping on it
    assert [r.job_id for r in rig.model.requests] == [jobs[1].job_id]


def test_a_persistently_failing_start_does_not_become_a_loop(rig: Rig) -> None:
    rig.fake_api.add_plan(1, clips=1)
    rig.fake_api.inject_fault("/v1/start", *[envelope(409, "wrong_state")] * 50)
    result = rig.run()
    assert result.reason == "idle" and result.dropped == 1
    assert len(rig.fake_api.calls("/v1/start")) == 1


# --- complete errors ---------------------------------------------------------------------------


def test_object_missing_reuploads_once_then_completes(rig: Rig) -> None:
    job = rig.fake_api.add_plan(1, clips=1).jobs[0]
    rig.uploader.skip_registration_times = 1
    result = rig.run(max_clips=1)
    assert result.done == 1 and job.status == "done"
    assert len(rig.uploader.uploads) == 2 and len(rig.fake_api.calls("/v1/complete")) == 2


def test_size_mismatch_reuploads_once_then_completes(rig: Rig) -> None:
    job = rig.fake_api.add_plan(1, clips=1).jobs[0]
    rig.uploader.register_wrong_size_times = 1
    result = rig.run(max_clips=1)
    assert result.done == 1 and job.status == "done" and len(rig.uploader.uploads) == 2


@pytest.mark.parametrize("which", ["skip_registration_times", "register_wrong_size_times"])
def test_a_second_mismatch_fails_the_job_with_upload_error(rig: Rig, which: str) -> None:
    job = rig.fake_api.add_plan(1, clips=1).jobs[0]
    setattr(rig.uploader, which, 2)
    result = rig.run(max_clips=1)
    assert result.failed == 1 and job.last_fail_code == "upload_error"
    assert len(rig.uploader.uploads) == 2 and len(rig.fake_api.calls("/v1/complete")) == 2


def test_key_mismatch_is_a_bug_and_fails_the_job_with_verify_failed(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    job = rig.fake_api.add_plan(1, clips=1).jobs[0]
    rig.fake_api.inject_fault("/v1/complete", envelope(409, "key_mismatch"))
    with caplog.at_level(logging.ERROR):
        result = rig.run(max_clips=1)
    assert result.failed == 1 and job.last_fail_code == "verify_failed"
    assert any(r.getMessage() == "complete_key_mismatch" for r in caplog.records)


@pytest.mark.parametrize("code", ["not_your_lease", "lease_expired", "wrong_state"])
def test_lost_leases_on_complete_are_dropped_without_failing(rig: Rig, code: str) -> None:
    rig.fake_api.add_plan(1, clips=1)
    rig.fake_api.inject_fault("/v1/complete", envelope(409, code))
    result = rig.run(max_clips=1)
    assert result.dropped == 1 and result.failed == 0
    assert "/v1/fail" not in routes(rig) and rig.leftover_files() == []


def test_transient_complete_errors_are_retried_with_backoff_until_it_works(rig: Rig) -> None:
    job = rig.fake_api.add_plan(1, clips=1).jobs[0]
    # the client makes 5 tries per call; the first two calls fail completely
    rig.fake_api.inject_fault("/v1/complete", *[httpx.Response(503)] * 10)
    result = rig.run(max_clips=1)
    assert result.done == 1 and job.status == "done"
    assert 5.0 in rig.clock.sleeps and 10.0 in rig.clock.sleeps
    assert len(rig.uploader.uploads) == 1  # no needless re-upload
    assert rig.leftover_files() == []


def test_complete_retry_gives_up_after_the_budget_and_moves_on(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, COMPLETE_RETRY_BUDGET_MINUTES="1")
    jobs = rig.fake_api.add_plan(1, clips=2).jobs
    rig.fake_api.inject_fault("/v1/complete", *[httpx.Response(503)] * 200)
    result = rig.run()
    assert result.done == 0 and result.dropped == 2 and result.failed == 0
    assert "/v1/fail" not in routes(rig) and rig.leftover_files() == []
    # uploaded but never confirmed: not done, and the next claim by this agent reset them
    assert [j.status for j in jobs] == ["pending", "pending"]


def test_complete_retries_are_bounded_by_the_run_deadline(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, RUN_DEADLINE_MINUTES="10", COMPLETE_RETRY_BUDGET_MINUTES="30")
    rig.fake_api.add_plan(1, clips=1)
    rig.fake_api.inject_fault("/v1/complete", *[httpx.Response(503)] * 500)
    result = rig.run()
    assert result.dropped == 1
    assert rig.clock.now() <= rig.clock.now().replace(hour=6, minute=11, second=0)


def test_complete_carries_the_verified_size_and_render_info(rig: Rig) -> None:
    import json

    job = rig.fake_api.add_plan(1, clips=1).jobs[0]
    rig.run(max_clips=1)
    body = json.loads(rig.fake_api.calls("/v1/complete")[0].body)
    assert body["upload"]["sizeBytes"] == rig.uploader.uploads[0].size > 0
    assert body["upload"]["key"] == job.key
    assert body["render"]["renderSeconds"] == 240.0 and body["render"]["hasAudio"] is True
    assert "costInr" not in body


def test_fail_call_errors_do_not_stop_the_run(rig: Rig) -> None:
    jobs = rig.fake_api.add_plan(1, clips=2).jobs
    rig.model.script(jobs[0].job_id, Reject("render_error"))
    rig.fake_api.inject_fault("/v1/fail", envelope(409, "not_your_lease"))
    result = rig.run()
    assert result.failed == 1 and result.done == 1 and result.exit_code == 0


# --- auth --------------------------------------------------------------------------------------


def test_auth_error_on_claim_exits_3_at_once(rig: Rig) -> None:
    rig.fake_api.inject_fault("/v1/claim", httpx.Response(403, json={"Message": "denied"}))
    result = rig.run()
    assert (result.exit_code, result.reason) == (3, "auth_error")
    assert rig.clock.sleeps == []


def test_auth_error_mid_clip_exits_3_and_cleans_up(rig: Rig) -> None:
    rig.fake_api.add_plan(1, clips=2)
    rig.fake_api.inject_fault("/v1/complete", httpx.Response(403))
    result = rig.run()
    assert (result.exit_code, result.reason) == (3, "auth_error")
    assert result.skipped == 1 and rig.leftover_files() == []


@pytest.mark.parametrize("route", ["/v1/start", "/v1/fail"])
def test_auth_error_on_start_or_fail_exits_3(rig: Rig, route: str) -> None:
    jobs = rig.fake_api.add_plan(1, clips=1).jobs
    if route == "/v1/fail":
        rig.model.script(jobs[0].job_id, Reject("render_error"))
    rig.fake_api.inject_fault(route, httpx.Response(403))
    assert rig.run().exit_code == 3


# --- scratch files, max clips, signals, resume -------------------------------------------------


def test_scratch_file_is_named_after_job_and_attempt_and_always_removed(rig: Rig) -> None:
    seen: list[bool] = []
    jobs = rig.fake_api.add_plan(1, clips=2).jobs
    rig.model.script(jobs[1].job_id, BadFile("not_mp4"))
    rig.model.on_generate = lambda request: seen.append(
        Path(request.output_path).parent == rig.scratch
    )
    rig.run()
    assert all(seen) and rig.leftover_files() == []
    assert rig.model.requests[0].output_path.endswith(f"/{jobs[0].job_id}-1.mp4")


def test_max_clips_stops_after_n_finished_clips(rig: Rig) -> None:
    rig.fake_api.add_plan(1, clips=3)
    result = rig.run(max_clips=2)
    assert (result.done, result.skipped, result.reason, result.exit_code) == (2, 1, "max_clips", 0)


def test_max_clips_counts_failed_clips_as_finished(rig: Rig) -> None:
    jobs = rig.fake_api.add_plan(1, clips=3).jobs
    rig.model.script(jobs[0].job_id, Reject("render_error"))
    result = rig.run(max_clips=1)
    assert (result.failed, result.done, result.reason) == (1, 0, "max_clips")


def test_stop_during_a_render_leaves_the_job_for_resume_and_does_not_fail_it(rig: Rig) -> None:
    jobs = rig.fake_api.add_plan(1, clips=2).jobs

    def stop_mid_render(request: object) -> None:
        rig.stop = True

    rig.model.on_generate = stop_mid_render
    result = rig.run()
    assert (result.exit_code, result.reason) == (0, "signal")
    assert result.done == 0 and result.failed == 0 and result.skipped == 1
    assert "/v1/fail" not in routes(rig) and rig.uploader.uploads == []
    assert rig.leftover_files() == []
    assert jobs[0].status == "generating"  # the same agent id resumes it

    # a restarted agent (same id) gets it back as resumed, attempts kept, and finishes it
    rig.stop = False
    rig.model.on_generate = None
    resumed = rig.run()
    assert resumed.done == 2
    assert jobs[0].status == "done" and jobs[0].attempts == 2


def test_resumed_flag_is_logged_for_restarted_jobs(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    job = rig.fake_api.add_plan(1, clips=1).jobs[0]
    job.claimed_by = rig.settings.agent_id
    job.status = "generating"
    job.attempts = 1
    job.lease_expires_at = rig.clock.now()
    with caplog.at_level(logging.INFO):
        result = rig.run(max_clips=1)
    assert result.done == 1 and job.attempts == 2
    starts = [r for r in caplog.records if r.getMessage() == "clip_start"]
    assert starts[0].__dict__["resumed"] == "true" and starts[0].__dict__["attempt"] == 2


def test_stop_before_the_first_claim_exits_cleanly(rig: Rig) -> None:
    rig.stop = True
    rig.fake_api.add_plan(1, clips=1)
    result = rig.run()
    assert (result.exit_code, result.reason) == (0, "signal")
    assert rig.fake_api.requests == []


def test_stop_while_waiting_for_the_model_server(rig: Rig) -> None:
    rig.model.not_ready_polls = 100
    rig.clock.on_sleep = lambda: setattr(rig, "stop", True)
    result = rig.run()
    assert result.reason == "signal" and rig.fake_api.requests == []


def test_stop_during_the_idle_wait(rig: Rig) -> None:
    rig.clock.on_sleep = lambda: setattr(rig, "stop", True)
    result = rig.run()
    assert result.reason == "signal" and len(rig.fake_api.calls("/v1/claim")) == 1


# --- summary -----------------------------------------------------------------------------------


def test_run_summary_line(rig: Rig, caplog: pytest.LogCaptureFixture) -> None:
    jobs = rig.fake_api.add_plan(1, clips=2).jobs
    rig.model.script(jobs[0].job_id, Reject("oom"))
    with caplog.at_level(logging.INFO):
        rig.run()
    summary = [r for r in caplog.records if r.getMessage() == "run_summary"]
    assert len(summary) == 1
    fields = summary[0].__dict__
    assert (fields["done"], fields["failed"], fields["skipped"]) == (2, 1, 0)
    assert fields["reason"] == "idle" and fields["runtime_s"] > 0


def test_unexpected_errors_propagate_after_logging_a_summary(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    rig.fake_api.add_plan(1, clips=1)

    def explode(request: object) -> None:
        raise RuntimeError("boom")

    rig.model.on_generate = explode
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError):
        rig.run()
    assert any(
        r.getMessage() == "run_summary" and r.__dict__["reason"] == "unexpected_error"
        for r in caplog.records
    )
    assert rig.leftover_files() == []
