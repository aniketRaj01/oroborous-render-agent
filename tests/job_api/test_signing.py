from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from botocore.credentials import ReadOnlyCredentials

from conftest import FAKE_URL
from render_agent.job_api.signing import SigV4Signer, UnsignedSigner

URL = FAKE_URL + "/v1/claim"
FAKE_KEY = "AKIA" + "FAKEFAKEFAKEFAKE"


@pytest.fixture(autouse=True)
def fixed_signing_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "botocore.auth.get_current_datetime",
        lambda: datetime(2026, 10, 18, 6, 0, 0, tzinfo=timezone.utc),
    )


def sign(body: bytes, token: str | None = None) -> dict[str, str]:
    signer = SigV4Signer(
        region="us-east-1",
        credentials=lambda: ReadOnlyCredentials(FAKE_KEY, "fake-secret-not-real", token),
    )
    return {k.lower(): v for k, v in signer.headers_for(method="POST", url=URL, body=body).items()}


def signature(headers: dict[str, str]) -> str:
    return headers["authorization"].split("Signature=")[1]


def test_signed_request_carries_authorization_and_payload_hash() -> None:
    body = b'{"agentId":"a"}'
    headers = sign(body)
    assert headers["x-amz-content-sha256"] == hashlib.sha256(body).hexdigest()
    authorization = headers["authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256 ")
    assert f"Credential={FAKE_KEY}/20261018/us-east-1/lambda/aws4_request" in authorization
    signed_headers = authorization.split("SignedHeaders=")[1].split(",")[0].split(";")
    assert "x-amz-content-sha256" in signed_headers
    assert "host" in signed_headers
    assert headers["content-type"] == "application/json"
    assert headers["x-amz-date"] == "20261018T060000Z"


def test_signature_is_deterministic_and_changes_when_the_body_changes() -> None:
    first = sign(b'{"jobId":1}')
    assert signature(first) == signature(sign(b'{"jobId":1}'))
    second = sign(b'{"jobId":2}')
    assert signature(first) != signature(second)
    assert first["x-amz-content-sha256"] != second["x-amz-content-sha256"]


def test_session_token_is_sent_when_present() -> None:
    assert sign(b"{}", token="fake-session-token")["x-amz-security-token"] == "fake-session-token"
    assert "x-amz-security-token" not in sign(b"{}")


def test_unsigned_signer_sends_no_authorization() -> None:
    headers = UnsignedSigner().headers_for(method="POST", url=URL, body=b"{}")
    assert headers == {"content-type": "application/json"}
