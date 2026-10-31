"""Fake last-frame extractor: writes a plausible PNG instead of running ffmpeg."""

from __future__ import annotations

from pathlib import Path

from render_agent.frames import PNG_SIGNATURE, FrameError

FAKE_PNG = PNG_SIGNATURE + b"fake-frame-data" * 20


class FakeFrameExtractor:
    def __init__(self) -> None:
        self.extracted: list[tuple[str, str]] = []
        self.failures: list[FrameError] = []

    def extract_last_frame(self, video: Path, dest: Path) -> int:
        if self.failures:
            raise self.failures.pop(0)
        self.extracted.append((video.name, dest.name))
        dest.write_bytes(FAKE_PNG)
        dest.chmod(0o644)
        return len(FAKE_PNG)
