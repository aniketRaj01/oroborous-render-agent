from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from render_agent.frames import FfmpegFrameExtractor, FrameError, check_png


def test_a_missing_ffmpeg_is_reported_in_words(tmp_path: Path) -> None:
    extractor = FfmpegFrameExtractor(ffmpeg=str(tmp_path / "no-such-ffmpeg"))
    with pytest.raises(FrameError, match="ffmpeg is required"):
        extractor.extract_last_frame(tmp_path / "clip.mp4", tmp_path / "frame.png")


def test_the_ffmpeg_command_follows_the_agreed_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(command)
        Path(command[-1]).write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 200)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    dest = tmp_path / "frame.png"
    size = FfmpegFrameExtractor().extract_last_frame(tmp_path / "clip.mp4", dest)
    assert size == 208 and dest.stat().st_mode & 0o777 == 0o644
    assert seen[0] == [
        "ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.1", "-i", str(tmp_path / "clip.mp4"),
        "-frames:v", "1", "-update", "1", str(dest),
    ]  # fmt: skip


def test_an_implausible_png_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png at all" * 20)
    with pytest.raises(FrameError):
        check_png(bad)
    bad.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(FrameError):
        check_png(bad)
