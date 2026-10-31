"""Fake uploader that registers the object with the fake Job API's S3 view, like a real upload."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fakes.clock import FakeClock
from fakes.fake_job_api import FakeJobApi
from render_agent.uploader import DownloadError, UploadError


@dataclass(frozen=True)
class UploadRecord:
    bucket: str
    region: str
    key: str
    size: int
    content_type: str = "video/mp4"


class FakeUploader:
    def __init__(
        self, fake_api: FakeJobApi, clock: FakeClock, *, upload_seconds: float = 10.0
    ) -> None:
        self._fake_api = fake_api
        self._clock = clock
        self._upload_seconds = upload_seconds
        self.uploads: list[UploadRecord] = []
        self.failures: list[UploadError] = []
        self.skip_registration_times = 0
        self.register_wrong_size_times = 0
        # (bucket, key) -> bytes, for what download() can fetch; uploads are stored here too.
        self.objects: dict[tuple[str, str], bytes] = {}
        self.downloads: list[tuple[str, str]] = []

    def upload(
        self,
        path: Path,
        *,
        bucket: str,
        region: str,
        key: str,
        content_type: str = "video/mp4",
    ) -> int:
        self._clock.advance(self._upload_seconds)
        if self.failures:
            raise self.failures.pop(0)
        size = path.stat().st_size
        self.uploads.append(UploadRecord(bucket, region, key, size, content_type))
        self.objects[(bucket, key)] = path.read_bytes()
        if self.skip_registration_times > 0:
            self.skip_registration_times -= 1
        elif self.register_wrong_size_times > 0:
            self.register_wrong_size_times -= 1
            self._fake_api.s3_objects[(bucket, key)] = size + 1
        else:
            self._fake_api.s3_objects[(bucket, key)] = size
        return size

    def download(self, *, bucket: str, region: str, key: str, dest: Path) -> int:
        self.downloads.append((bucket, key))
        data = self.objects.get((bucket, key))
        if data is None:
            raise DownloadError("S3 request failed (NoSuchKey)")
        dest.write_bytes(data)
        dest.chmod(0o644)
        return len(data)
