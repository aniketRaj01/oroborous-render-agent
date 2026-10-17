from __future__ import annotations

import pytest

from fakes.clock import FakeClock
from fakes.fake_job_api import FakeJobApi
from render_agent.job_api.client import JobApiClient, RetryPolicy
from render_agent.job_api.signing import RequestSigner, UnsignedSigner

FAKE_URL = "https://fake-id.lambda-url.us-east-1.on.aws"
FAKE_AGENT = "test-agent-1"


def valid_env() -> dict[str, str]:
    return {"JOB_API_URL": FAKE_URL, "AGENT_ID": FAKE_AGENT}


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake_api(clock: FakeClock) -> FakeJobApi:
    return FakeJobApi(clock)


def build_client(
    fake_api: FakeJobApi,
    clock: FakeClock,
    *,
    agent_id: str = FAKE_AGENT,
    signer: RequestSigner | None = None,
    max_retries: int = 4,
) -> JobApiClient:
    import httpx

    return JobApiClient(
        base_url=FAKE_URL,
        agent_id=agent_id,
        agent_version="1.2.3-test",
        http=httpx.Client(transport=fake_api.transport),
        signer=signer or UnsignedSigner(),
        clock=clock,
        retry=RetryPolicy(
            max_retries=max_retries,
            base_delay=0.5,
            max_delay=10.0,
            jitter=0.5,
            random_fraction=lambda: 0.0,
        ),
    )


@pytest.fixture
def client(fake_api: FakeJobApi, clock: FakeClock) -> JobApiClient:
    return build_client(fake_api, clock)
