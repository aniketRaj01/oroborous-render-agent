from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

CONTENT_TYPE = "video/mp4"


class UploadError(Exception):
    """The clip is not verifiably in S3. Messages carry error codes only, never bucket names."""


class DownloadError(Exception):
    """The object could not be fetched. Messages carry error codes only, never bucket names."""


class Uploader(Protocol):
    def upload(
        self, path: Path, *, bucket: str, region: str, key: str, content_type: str = CONTENT_TYPE
    ) -> int:
        """Upload `path` to the given location and return the verified size in bytes."""
        ...

    def download(self, *, bucket: str, region: str, key: str, dest: Path) -> int:
        """Fetch the object into `dest` (mode 0644, written atomically); return its size."""
        ...


def _default_client(region: str) -> S3Client:
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            retries={"mode": "standard", "max_attempts": 5},
            connect_timeout=10,
            read_timeout=120,
        ),
    )


class S3Uploader:
    """Credentials come from the default chain (the instance role on the GPU machine)."""

    def __init__(self, client_factory: Callable[[str], S3Client] = _default_client) -> None:
        self._client_factory = client_factory
        self._clients: dict[str, S3Client] = {}

    def _client(self, region: str) -> S3Client:
        if region not in self._clients:
            self._clients[region] = self._client_factory(region)
        return self._clients[region]

    def upload(
        self, path: Path, *, bucket: str, region: str, key: str, content_type: str = CONTENT_TYPE
    ) -> int:
        try:
            size = path.stat().st_size
        except OSError:
            raise UploadError("local file is not readable") from None
        if size == 0:
            raise UploadError("local file is empty")
        client = self._client(region)
        try:
            with path.open("rb") as body:
                client.put_object(
                    Bucket=bucket, Key=key, Body=body, ContentLength=size, ContentType=content_type
                )
            remote_size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "unknown")
            raise UploadError(f"S3 request failed ({code})") from None
        except (BotoCoreError, OSError) as exc:
            raise UploadError(f"S3 request failed ({type(exc).__name__})") from None
        if remote_size != size:
            raise UploadError("object size in S3 differs from the local file after upload")
        return size

    def download(self, *, bucket: str, region: str, key: str, dest: Path) -> int:
        client = self._client(region)
        partial = dest.with_name(f"{dest.name}.partial")
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"]
            try:
                with partial.open("wb") as handle:
                    for chunk in body.iter_chunks(chunk_size=1 << 20):
                        handle.write(chunk)
            finally:
                body.close()
            size = partial.stat().st_size
            if size == 0:
                raise DownloadError("downloaded object is empty")
            # The model server reads this file and may run as another user.
            os.chmod(partial, 0o644)
            os.replace(partial, dest)
        except ClientError as exc:
            partial.unlink(missing_ok=True)
            code = exc.response.get("Error", {}).get("Code", "unknown")
            raise DownloadError(f"S3 request failed ({code})") from None
        except (BotoCoreError, OSError) as exc:
            partial.unlink(missing_ok=True)
            raise DownloadError(f"S3 request failed ({type(exc).__name__})") from None
        except DownloadError:
            partial.unlink(missing_ok=True)
            raise
        return size
