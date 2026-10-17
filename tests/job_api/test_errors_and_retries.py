from __future__ import annotations

import logging

import httpx
import pytest

from conftest import FAKE_URL, build_client
from fakes.clock import FakeClock
from fakes.fake_job_api import FakeJobApi
from render_agent.job_api.client import JobApiClient, RetryPolicy
from render_agent.job_api.errors import (
    AuthError,
    ContractError,
    InvalidRequestError,
    JobApiError,
    JobNotFoundError,
    KeyMismatchError,
    LeaseExpiredError,
    MaxAttemptsReachedError,
    MethodNotAllowedError,
    NotYourLeaseError,
    ObjectMissingError,
    ProtocolError,
    RouteNotFoundError,
    ServerError,
    SizeMismatchError,
    ThrottledError,
    TransientError,
    TransportError,
    UnknownContractError,
    WrongStateError,
)


def envelope(status: int, code: str, message: str = "msg") -> httpx.Response:
    return httpx.Response(
        status, json={"ok": False, "error": {"code": code, "message": message, "details": {}}}
    )


CONTRACT_CASES: list[tuple[int, str, type[JobApiError]]] = [
    (400, "invalid_request", InvalidRequestError),
    (404, "job_not_found", JobNotFoundError),
    (404, "not_found", RouteNotFoundError),
    (405, "method_not_allowed", MethodNotAllowedError),
    (409, "not_your_lease", NotYourLeaseError),
    (409, "lease_expired", LeaseExpiredError),
    (409, "wrong_state", WrongStateError),
    (409, "max_attempts_reached", MaxAttemptsReachedError),
    (409, "key_mismatch", KeyMismatchError),
    (409, "object_missing", ObjectMissingError),
    (409, "size_mismatch", SizeMismatchError),
    (500, "internal", ServerError),
    (409, "some_future_code", UnknownContractError),
]


@pytest.mark.parametrize(("status", "code", "expected"), CONTRACT_CASES)
def test_every_contract_error_code_maps_to_its_exception(
    fake_api: FakeJobApi, client: JobApiClient, status: int, code: str, expected: type[JobApiError]
) -> None:
    fake_api.inject_fault("/v1/start", *[envelope(status, code)] * 10)
    with pytest.raises(expected) as caught:
        client.start(1)
    assert caught.value.code == code and caught.value.http_status == status
    assert isinstance(caught.value, (ContractError, TransientError))


def test_contract_errors_are_not_retried(
    fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock
) -> None:
    for status, code, expected in CONTRACT_CASES:
        if status >= 500:
            continue
        fake_api.inject_fault("/v1/start", envelope(status, code))
        with pytest.raises(expected):
            client.start(1)
    assert len(fake_api.calls("/v1/start")) == sum(1 for s, _, _ in CONTRACT_CASES if s < 500)
    assert clock.sleeps == []


def test_real_fake_api_errors_surface_through_the_client(client: JobApiClient) -> None:
    with pytest.raises(JobNotFoundError):
        client.start(99999)


def test_403_is_an_auth_error_with_a_hint_and_is_not_retried(
    fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock
) -> None:
    fake_api.inject_fault("/v1/claim", httpx.Response(403, json={"Message": "denied"}))
    with pytest.raises(AuthError) as caught:
        client.claim()
    assert "lambda:InvokeFunctionUrl" in str(caught.value)
    assert "lambda-url" not in str(caught.value) and FAKE_URL not in str(caught.value)
    assert len(fake_api.calls("/v1/claim")) == 1 and clock.sleeps == []


def test_429_is_throttling_and_is_retried(
    fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock
) -> None:
    fake_api.inject_fault("/v1/claim", httpx.Response(429, json={"message": "Rate Exceeded."}))
    assert client.claim().totals.clips == 0
    assert len(fake_api.calls("/v1/claim")) == 2 and clock.sleeps == [0.5]


def test_429_after_all_retries_raises_throttled(fake_api: FakeJobApi, client: JobApiClient) -> None:
    fake_api.inject_fault("/v1/claim", *[httpx.Response(429)] * 5)
    with pytest.raises(ThrottledError):
        client.claim()
    assert len(fake_api.calls("/v1/claim")) == 5


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"<html>not json</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"ok": True}),
        httpx.Response(418, content=b"teapot"),
        httpx.Response(404, json={"unexpected": "shape"}),
        httpx.Response(302, headers={"location": "https://elsewhere.invalid/"}),
    ],
)
def test_responses_that_break_the_protocol(
    fake_api: FakeJobApi, client: JobApiClient, response: httpx.Response
) -> None:
    fake_api.inject_fault("/v1/claim", response)
    with pytest.raises(ProtocolError):
        client.claim()
    assert len(fake_api.calls("/v1/claim")) == 1


def test_non_json_body_is_truncated_and_redacted_in_the_message(
    fake_api: FakeJobApi, client: JobApiClient
) -> None:
    host = "leakyid.lambda-url.us-east-1.on.aws"
    body = f"upstream {host} " + "A" * 5000
    fake_api.inject_fault("/v1/claim", httpx.Response(200, content=body.encode()))
    with pytest.raises(ProtocolError) as caught:
        client.claim()
    message = str(caught.value)
    assert "leakyid" not in message and len(message) < 400


def test_contract_violations_in_a_200_name_the_fields_not_the_values(
    fake_api: FakeJobApi, client: JobApiClient
) -> None:
    fake_api.inject_fault(
        "/v1/start", httpx.Response(200, json={"ok": True, "jobId": "secret-looking", "attempt": 1})
    )
    with pytest.raises(ProtocolError) as caught:
        client.start(1)
    assert "secret-looking" not in str(caught.value)
    assert "jobId" in str(caught.value) or "job_id" in str(caught.value)


def test_unknown_extra_response_fields_are_tolerated(
    fake_api: FakeJobApi, client: JobApiClient
) -> None:
    fake_api.inject_fault(
        "/v1/start",
        httpx.Response(
            200,
            json={
                "ok": True,
                "jobId": 1,
                "attempt": 1,
                "maxAttempts": 3,
                "alreadyStarted": False,
                "somethingNew": {"a": 1},
            },
        ),
    )
    assert client.start(1).attempt == 1


# --- retries -----------------------------------------------------------------------------------


def test_transient_failures_are_retried_with_exponential_backoff(
    fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock
) -> None:
    fake_api.inject_fault(
        "/v1/claim",
        httpx.Response(503),
        httpx.ConnectError("boom"),
        envelope(500, "internal"),
        httpx.ReadTimeout("slow"),
    )
    assert client.claim().totals.clips == 0
    assert len(fake_api.calls("/v1/claim")) == 5
    assert clock.sleeps == [0.5, 1.0, 2.0, 4.0]


def test_retries_are_capped(fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock) -> None:
    fake_api.inject_fault("/v1/claim", *[httpx.Response(502)] * 20)
    with pytest.raises(ServerError):
        client.claim()
    assert len(fake_api.calls("/v1/claim")) == 5
    assert clock.sleeps == [0.5, 1.0, 2.0, 4.0]


def test_timeouts_and_connection_errors_raise_transport_error(
    fake_api: FakeJobApi, clock: FakeClock
) -> None:
    client = build_client(fake_api, clock, max_retries=1)
    fake_api.inject_fault("/v1/claim", httpx.ReadTimeout("t"), httpx.ReadTimeout("t"))
    with pytest.raises(TransportError):
        client.claim()
    fake_api.inject_fault("/v1/claim", httpx.ConnectError("c"), httpx.ConnectError("c"))
    with pytest.raises(TransportError):
        client.claim()


def test_transport_error_messages_do_not_leak_the_exception_text(
    fake_api: FakeJobApi, clock: FakeClock
) -> None:
    client = build_client(fake_api, clock, max_retries=0)
    fake_api.inject_fault("/v1/claim", httpx.ConnectError("could not reach leakyhost.invalid"))
    with pytest.raises(TransportError) as caught:
        client.claim()
    assert "leakyhost" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_zero_retries_means_one_attempt(fake_api: FakeJobApi, clock: FakeClock) -> None:
    client = build_client(fake_api, clock, max_retries=0)
    fake_api.inject_fault("/v1/claim", httpx.Response(503))
    with pytest.raises(ServerError):
        client.claim()
    assert clock.sleeps == []


def test_every_call_is_retry_safe_each_route_retries(
    fake_api: FakeJobApi, client: JobApiClient, clock: FakeClock
) -> None:
    from render_agent.job_api.models import CompleteUpload, FailErrorCode, RenderInfo

    render = RenderInfo(
        render_seconds=1,
        width=1,
        height=1,
        num_frames=1,
        fps=1,
        duration_seconds=1,
        has_audio=False,
    )
    upload = CompleteUpload(bucket="b", key="k", size_bytes=1)
    for route in ("/v1/start", "/v1/complete", "/v1/fail"):
        fake_api.inject_fault(route, httpx.Response(503))
    with pytest.raises(JobNotFoundError):
        client.start(99999)
    with pytest.raises(JobNotFoundError):
        client.complete(99999, upload=upload, render=render)
    with pytest.raises(JobNotFoundError):
        client.fail(99999, code=FailErrorCode.UNKNOWN, message="x")
    for route in ("/v1/start", "/v1/complete", "/v1/fail"):
        assert len(fake_api.calls(route)) == 2
    assert len(clock.sleeps) == 3


def test_a_fresh_signature_is_made_for_every_attempt(
    fake_api: FakeJobApi, clock: FakeClock
) -> None:
    seen: list[int] = []

    class CountingSigner:
        def headers_for(self, *, method: str, url: str, body: bytes) -> dict[str, str]:
            seen.append(len(seen))
            return {"content-type": "application/json"}

    client = build_client(fake_api, clock, signer=CountingSigner())
    fake_api.inject_fault("/v1/claim", httpx.Response(503), httpx.Response(503))
    client.claim()
    assert seen == [0, 1, 2]


def test_retry_policy_backoff_cap_and_jitter() -> None:
    no_jitter = RetryPolicy(max_retries=9, base_delay=0.5, max_delay=10.0, jitter=0.0)
    assert [no_jitter.delay_before_retry(n) for n in range(1, 8)] == [0.5, 1, 2, 4, 8, 10, 10]
    low = RetryPolicy(
        max_retries=9, base_delay=1.0, max_delay=10.0, jitter=0.5, random_fraction=lambda: 0.0
    )
    high = RetryPolicy(
        max_retries=9, base_delay=1.0, max_delay=10.0, jitter=0.5, random_fraction=lambda: 1.0
    )
    assert low.delay_before_retry(3) == 4.0 and high.delay_before_retry(3) == 2.0


# --- logging -----------------------------------------------------------------------------------


def test_each_call_logs_route_status_code_duration_and_attempt(
    fake_api: FakeJobApi,
    client: JobApiClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_api.inject_fault("/v1/start", httpx.Response(503), envelope(409, "not_your_lease"))
    with (
        caplog.at_level(logging.INFO, logger="render_agent.job_api.client"),
        pytest.raises(NotYourLeaseError),
    ):
        client.start(1)
    records = [r for r in caplog.records if r.getMessage() == "job_api_call"]
    assert [
        (r.__dict__["attempt"], r.__dict__["status"], r.__dict__["error_code"]) for r in records
    ] == [
        (1, 503, "none"),
        (2, 409, "not_your_lease"),
    ]
    assert all(r.__dict__["route"] == "/v1/start" and "duration_ms" in r.__dict__ for r in records)
    text = " ".join(str(r.__dict__) + r.getMessage() for r in records)
    assert "lambda-url" not in text and "fake-id" not in text


def test_client_repr_does_not_show_the_url(client: JobApiClient) -> None:
    assert "fake-id" not in repr(client) and "lambda-url" not in repr(client)


def test_missing_credentials_are_an_auth_error_and_are_not_retried(
    fake_api: FakeJobApi, clock: FakeClock
) -> None:
    from render_agent.job_api.signing import NoCredentialsError

    class NoCredentialsSigner:
        def headers_for(self, *, method: str, url: str, body: bytes) -> dict[str, str]:
            raise NoCredentialsError("none")

    client = build_client(fake_api, clock, signer=NoCredentialsSigner())
    with pytest.raises(AuthError, match="credentials"):
        client.claim()
    assert fake_api.requests == [] and clock.sleeps == []
