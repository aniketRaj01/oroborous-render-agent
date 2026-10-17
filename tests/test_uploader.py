from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, cast

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from moto import mock_aws
from mypy_boto3_s3 import S3Client

from render_agent.uploader import S3Uploader, UploadError, _default_client

REGION: Literal["ap-south-1"] = "ap-south-1"
BUCKET = "fake-test-bucket"
KEY = "plans/253/clip-0.mp4"
CONTENT = b"\x00\x00\x00\x18ftypisom" + b"FAKE" * 1000


@pytest.fixture(autouse=True)
def fake_aws(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "AWS_ACCESS_KEY_ID": "fake",
        "AWS_SECRET_ACCESS_KEY": "fake",
        "AWS_SESSION_TOKEN": "fake",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def s3() -> Iterator[S3Client]:
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        yield client


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "412-1.mp4"
    path.write_bytes(CONTENT)
    return path


class Flaky:
    """Wraps a real (moto) client and injects a fault on one operation."""

    def __init__(
        self,
        inner: S3Client,
        *,
        put: Exception | None = None,
        head: Exception | None = None,
        head_size_delta: int = 0,
    ) -> None:
        self._inner = inner
        self._put, self._head, self._delta = put, head, head_size_delta

    def put_object(self, **kwargs: Any) -> Any:
        if self._put:
            raise self._put
        return self._inner.put_object(**kwargs)

    def head_object(self, **kwargs: Any) -> Any:
        if self._head:
            raise self._head
        response: dict[str, Any] = dict(self._inner.head_object(**kwargs))
        response["ContentLength"] += self._delta
        return response


def uploader_with(client: object) -> S3Uploader:
    return S3Uploader(client_factory=lambda region: cast(S3Client, client))


def upload(uploader: S3Uploader, path: Path, bucket: str = BUCKET) -> int:
    return uploader.upload(path, bucket=bucket, region=REGION, key=KEY)


def client_error(code: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "nope"}}, operation)


def test_upload_succeeds_and_returns_the_verified_size(s3: S3Client, clip: Path) -> None:
    assert upload(S3Uploader(), clip) == len(CONTENT)
    stored = s3.get_object(Bucket=BUCKET, Key=KEY)
    assert stored["Body"].read() == CONTENT
    assert stored["ContentType"] == "video/mp4"


def test_one_client_per_region_is_cached(s3: S3Client, clip: Path) -> None:
    regions: list[str] = []

    def factory(region: str) -> S3Client:
        regions.append(region)
        return boto3.client("s3", region_name=region)

    uploader = S3Uploader(client_factory=factory)
    upload(uploader, clip)
    upload(uploader, clip)
    assert regions == [REGION]


def test_default_client_uses_standard_retries_and_timeouts(s3: S3Client) -> None:
    config: Any = _default_client(REGION).meta.config
    assert config.retries["mode"] == "standard" and config.retries["total_max_attempts"] == 6
    assert config.connect_timeout == 10 and config.read_timeout == 120


def test_missing_bucket_is_an_upload_error(s3: S3Client, clip: Path) -> None:
    with pytest.raises(UploadError, match="NoSuchBucket") as caught:
        upload(S3Uploader(), clip, bucket="bucket-that-does-not-exist")
    assert "bucket-that-does-not-exist" not in str(caught.value)


def test_access_denied_is_an_upload_error(s3: S3Client, clip: Path) -> None:
    flaky = Flaky(s3, put=client_error("AccessDenied", "PutObject"))
    with pytest.raises(UploadError, match="AccessDenied"):
        upload(uploader_with(flaky), clip)


def test_transient_failure_is_an_upload_error(s3: S3Client, clip: Path) -> None:
    flaky = Flaky(s3, put=EndpointConnectionError(endpoint_url="https://leaky.invalid"))
    with pytest.raises(UploadError, match="EndpointConnectionError") as caught:
        upload(uploader_with(flaky), clip)
    assert "leaky" not in str(caught.value)


def test_head_failure_after_a_successful_put_is_an_upload_error(s3: S3Client, clip: Path) -> None:
    flaky = Flaky(s3, head=client_error("403", "HeadObject"))
    with pytest.raises(UploadError):
        upload(uploader_with(flaky), clip)


def test_size_mismatch_after_upload_is_an_upload_error(s3: S3Client, clip: Path) -> None:
    flaky = Flaky(s3, head_size_delta=-1)
    with pytest.raises(UploadError, match="differs"):
        upload(uploader_with(flaky), clip)


def test_empty_or_missing_local_files_are_rejected_before_any_request(
    s3: S3Client, tmp_path: Path
) -> None:
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(UploadError, match="empty"):
        upload(S3Uploader(), empty)
    with pytest.raises(UploadError, match="not readable"):
        upload(S3Uploader(), tmp_path / "missing.mp4")
