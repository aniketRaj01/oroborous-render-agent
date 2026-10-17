from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest
from botocore.credentials import ReadOnlyCredentials

from conftest import FAKE_AGENT, build_client
from fakes.clock import FakeClock
from fakes.fake_job_api import FAKE_BUCKET, FakeJobApi
from render_agent.job_api.client import JobApiClient
from render_agent.job_api.errors import (
    KeyMismatchError,
    LeaseExpiredError,
    MaxAttemptsReachedError,
    NotYourLeaseError,
    ObjectMissingError,
    SizeMismatchError,
    WrongStateError,
)
from render_agent.job_api.models import (
    ClaimResponse,
    CompleteUpload,
    FailErrorCode,
    JobStatus,
    RenderInfo,
)
from render_agent.job_api.signing import SigV4Signer

RENDER = RenderInfo(
    render_seconds=142.6,
    width=576,
    height=1024,
    num_frames=361,
    fps=24,
    duration_seconds=15.04,
    has_audio=True,
)


def body_of(fake_api: FakeJobApi, route: str, index: int = -1) -> dict[str, Any]:
    parsed = json.loads(fake_api.calls(route)[index].body)
    assert isinstance(parsed, dict)
    return parsed


def upload_for(fake_api: FakeJobApi, job_id: int, size: int = 1000) -> CompleteUpload:
    job = fake_api.job(job_id)
    return CompleteUpload(bucket=FAKE_BUCKET, key=job.key, size_bytes=size)


def claim_one(fake_api: FakeJobApi, client: JobApiClient) -> ClaimResponse:
    fake_api.add_plan(253, clips=2)
    return client.claim()


# --- claim -------------------------------------------------------------------------------------


def test_claim_returns_plans_and_clips_in_order(fake_api: FakeJobApi, client: JobApiClient) -> None:
    fake_api.add_plan(254, clips=1)
    fake_api.add_plan(253, clips=2)
    response = client.claim()
    assert response.totals.plans == 2 and response.totals.clips == 3
    assert [p.plan_id for p in response.plans] == [254, 253]  # creation order, not id order
    plan = response.plans[1]
    assert (plan.account_id, plan.clips_total, plan.clips_already_done) == (5, 2, 0)
    assert [c.position for c in plan.clips] == [0, 1]
    clip = plan.clips[0]
    assert clip.upload.key == "plans/253/clip-0.mp4"
    assert clip.upload.region == "us-east-1"
    assert clip.params is not None and clip.params.num_frames == 361 and clip.params.seed == 42
    assert (clip.attempts, clip.resumed) == (0, False)
    assert response.server_time.tzinfo is not None
    assert "fake prompt" not in repr(clip)


def test_claim_request_body_matches_the_contract(
    fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock
) -> None:
    deadline = clock.now() + timedelta(minutes=2, microseconds=123456)
    client.claim(deadline_at=deadline, max_plans=1)
    assert body_of(fake_api, "/v1/claim") == {
        "agentId": FAKE_AGENT,
        "agentVersion": "1.2.3-test",
        "deadlineAt": "2026-10-18T06:02:00Z",
        "maxPlans": 1,
    }
    client.claim()
    assert set(body_of(fake_api, "/v1/claim")) == {"agentId", "agentVersion"}
    assert fake_api.calls("/v1/claim")[0].headers["content-type"] == "application/json"


def test_claim_lease_is_deadline_plus_margin(
    fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock
) -> None:
    fake_api.add_plan(1)
    response = client.claim(deadline_at=clock.now() + timedelta(minutes=2))
    assert response.lease_expires_at == clock.now() + timedelta(minutes=2, seconds=300)
    assert fake_api.job(fake_api.plans[0].jobs[0].job_id).claimed_by == FAKE_AGENT


def test_claim_with_far_or_past_deadline_uses_the_server_default(
    fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock
) -> None:
    for deadline in (clock.now() - timedelta(minutes=1), clock.now() + timedelta(minutes=181)):
        response = client.claim(deadline_at=deadline)
        assert response.lease_expires_at == clock.now() + timedelta(minutes=90, seconds=300)


def test_claim_with_nothing_to_do_still_returns_a_lease_time(client: JobApiClient) -> None:
    response = client.claim()
    assert response.plans == [] and response.totals.clips == 0
    assert response.lease_expires_at > response.server_time


def test_max_plans_caps_whole_plans(fake_api: FakeJobApi, client: JobApiClient) -> None:
    for plan_id in (1, 2, 3):
        fake_api.add_plan(plan_id, clips=2)
    response = client.claim(max_plans=2)
    assert [len(p.clips) for p in response.plans] == [2, 2]


def test_another_agent_cannot_claim_live_leases_but_can_after_expiry(
    fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock
) -> None:
    claim_one(fake_api, client)
    other = build_client(fake_api, clock, agent_id="other-agent")
    assert other.claim().totals.clips == 0
    clock.advance(91 * 60 + 301)
    reclaimed = other.claim()
    assert reclaimed.totals.clips == 2
    assert all(not clip.resumed for clip in reclaimed.plans[0].clips)


def test_same_agent_resume_resets_generating_to_pending_and_keeps_attempts(
    fake_api: FakeJobApi, client: JobApiClient
) -> None:
    first = claim_one(fake_api, client)
    job_id = first.plans[0].clips[0].job_id
    client.start(job_id)
    assert fake_api.job(job_id).status == "generating"
    again = client.claim()  # as after a process restart
    clips = again.plans[0].clips
    assert all(c.resumed for c in clips)
    assert clips[0].attempts == 1
    assert fake_api.job(job_id).status == "pending"


def test_done_clips_are_not_returned_and_counted(
    fake_api: FakeJobApi, client: JobApiClient
) -> None:
    first = claim_one(fake_api, client)
    job_id = first.plans[0].clips[0].job_id
    client.start(job_id)
    fake_api.put_object(fake_api.job(job_id), 1000)
    client.complete(job_id, upload=upload_for(fake_api, job_id), render=RENDER)
    again = client.claim()
    assert again.totals.clips == 1
    assert again.plans[0].clips_already_done == 1 and again.plans[0].clips_total == 2


# --- start -------------------------------------------------------------------------------------


def test_start_counts_an_attempt_and_is_idempotent(
    fake_api: FakeJobApi, client: JobApiClient
) -> None:
    job_id = claim_one(fake_api, client).plans[0].clips[0].job_id
    started = client.start(job_id)
    assert (started.attempt, started.max_attempts, started.already_started) == (1, 3, False)
    repeated = client.start(job_id)
    assert (repeated.attempt, repeated.already_started) == (1, True)
    assert body_of(fake_api, "/v1/start") == {"agentId": FAKE_AGENT, "jobId": job_id}


def test_start_without_a_lease_or_after_expiry(
    fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock
) -> None:
    job_id = claim_one(fake_api, client).plans[0].clips[0].job_id
    stranger = build_client(fake_api, clock, agent_id="other-agent")
    with pytest.raises(NotYourLeaseError):
        stranger.start(job_id)
    clock.advance(24 * 3600)
    with pytest.raises(LeaseExpiredError):
        client.start(job_id)


def test_start_beyond_max_attempts_marks_the_job_failed(
    fake_api: FakeJobApi, client: JobApiClient
) -> None:
    job_id = claim_one(fake_api, client).plans[0].clips[0].job_id
    for _ in range(2):
        client.start(job_id)
        client.fail(job_id, code=FailErrorCode.RENDER_ERROR, message="boom")
    client.start(job_id)  # third attempt; the process dies mid-render
    client.claim()  # restart: same-agent resume resets the job to pending, attempts stay at 3
    with pytest.raises(MaxAttemptsReachedError):
        client.start(job_id)
    assert fake_api.job(job_id).status == "failed"


# --- complete ----------------------------------------------------------------------------------


def test_complete_verifies_and_reports_plan_progress(
    fake_api: FakeJobApi, client: JobApiClient
) -> None:
    clips = claim_one(fake_api, client).plans[0].clips
    done = None
    for clip in clips:
        client.start(clip.job_id)
        fake_api.put_object(fake_api.job(clip.job_id), 6512340)
        done = client.complete(
            clip.job_id,
            upload=upload_for(fake_api, clip.job_id, 6512340),
            render=RENDER,
            cost_inr=8.5,
        )
        assert done.status is JobStatus.DONE and done.storage_key == clip.upload.key
    assert done is not None and done.plan.ready and done.plan.clips_done == 2
    assert body_of(fake_api, "/v1/complete", 0) == {
        "agentId": FAKE_AGENT,
        "jobId": clips[0].job_id,
        "upload": {"bucket": FAKE_BUCKET, "key": clips[0].upload.key, "sizeBytes": 6512340},
        "render": {
            "renderSeconds": 142.6,
            "width": 576,
            "height": 1024,
            "numFrames": 361,
            "fps": 24.0,
            "durationSeconds": 15.04,
            "hasAudio": True,
        },
        "costInr": 8.5,
    }


def test_complete_is_idempotent(fake_api: FakeJobApi, client: JobApiClient) -> None:
    job_id = claim_one(fake_api, client).plans[0].clips[0].job_id
    client.start(job_id)
    fake_api.put_object(fake_api.job(job_id), 1000)
    first = client.complete(job_id, upload=upload_for(fake_api, job_id), render=RENDER)
    second = client.complete(job_id, upload=upload_for(fake_api, job_id), render=RENDER)
    assert first == second
    assert "costInr" not in body_of(fake_api, "/v1/complete")


def test_complete_error_cases(fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock) -> None:
    job_id = claim_one(fake_api, client).plans[0].clips[0].job_id
    with pytest.raises(WrongStateError):  # still pending: never started
        client.complete(job_id, upload=upload_for(fake_api, job_id), render=RENDER)
    client.start(job_id)
    with pytest.raises(ObjectMissingError):
        client.complete(job_id, upload=upload_for(fake_api, job_id), render=RENDER)
    fake_api.put_object(fake_api.job(job_id), 999)
    with pytest.raises(SizeMismatchError):
        client.complete(job_id, upload=upload_for(fake_api, job_id, 1000), render=RENDER)
    with pytest.raises(KeyMismatchError):
        client.complete(
            job_id,
            upload=CompleteUpload(bucket=FAKE_BUCKET, key="plans/1/other.mp4", size_bytes=999),
            render=RENDER,
        )
    stranger = build_client(fake_api, clock, agent_id="other-agent")
    with pytest.raises(NotYourLeaseError):
        stranger.complete(job_id, upload=upload_for(fake_api, job_id, 999), render=RENDER)
    assert fake_api.job(job_id).status == "generating"


# --- fail --------------------------------------------------------------------------------------


def test_fail_retries_until_attempts_are_used(fake_api: FakeJobApi, client: JobApiClient) -> None:
    job_id = claim_one(fake_api, client).plans[0].clips[0].job_id
    expectations = [
        (JobStatus.PENDING, 2, True),
        (JobStatus.PENDING, 1, True),
        (JobStatus.FAILED, 0, False),
    ]
    for status, left, retry in expectations:
        client.start(job_id)
        failed = client.fail(job_id, code=FailErrorCode.OOM, message="CUDA out of memory")
        assert (failed.status, failed.attempts_left, failed.will_retry) == (status, left, retry)
    assert body_of(fake_api, "/v1/fail")["error"] == {
        "code": "oom",
        "message": "CUDA out of memory",
    }


def test_fail_is_idempotent_and_truncates_long_messages(
    fake_api: FakeJobApi, client: JobApiClient
) -> None:
    job_id = claim_one(fake_api, client).plans[0].clips[0].job_id
    client.start(job_id)
    first = client.fail(job_id, code=FailErrorCode.UPLOAD_ERROR, message="x" * 5000)
    second = client.fail(job_id, code=FailErrorCode.UPLOAD_ERROR, message="x" * 5000)
    assert first == second
    assert len(body_of(fake_api, "/v1/fail")["error"]["message"]) == 2000


def test_fail_on_a_job_that_is_not_generating(fake_api: FakeJobApi, client: JobApiClient) -> None:
    job_id = claim_one(fake_api, client).plans[0].clips[0].job_id
    with pytest.raises(WrongStateError):
        client.fail(job_id, code=FailErrorCode.UNKNOWN, message="never started")


# --- signing end to end ------------------------------------------------------------------------


def test_requests_are_accepted_by_a_fake_that_checks_the_signature(
    clock: FakeClock,
) -> None:
    fake = FakeJobApi(clock, require_signature=True)
    signer = SigV4Signer(
        region="us-east-1",
        credentials=lambda: ReadOnlyCredentials("AKIA" + "FAKEFAKEFAKEFAKE", "fake", None),
    )
    signed = build_client(fake, clock, signer=signer)
    assert signed.claim().totals.clips == 0
    unsigned = build_client(fake, clock)
    from render_agent.job_api.errors import AuthError

    with pytest.raises(AuthError):
        unsigned.claim()
