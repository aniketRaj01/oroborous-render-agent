from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# A real frame is far above the minimum size of a valid PNG; anything this small is a broken file.
MIN_FRAME_BYTES = 100
MAX_FRAME_BYTES = 50 * 1024 * 1024
FFMPEG_TIMEOUT_SECONDS = 60.0
FFMPEG_MISSING = "ffmpeg is required to extract the last frame of a chained clip but was not found"


class FrameError(Exception):
    """The last frame could not be extracted. Messages never contain paths or prompts."""


class FrameExtractor(Protocol):
    def extract_last_frame(self, video: Path, dest: Path) -> int:
        """Write the last frame of `video` to `dest` as a PNG and return its size in bytes."""
        ...


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def check_png(path: Path) -> int:
    """Size of a plausible PNG file at `path`; raises FrameError otherwise."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            signature = handle.read(len(PNG_SIGNATURE))
    except OSError:
        raise FrameError("frame file is missing") from None
    if not MIN_FRAME_BYTES <= size <= MAX_FRAME_BYTES or signature != PNG_SIGNATURE:
        raise FrameError("frame file is not a plausible PNG")
    return size


class FfmpegFrameExtractor:
    def __init__(self, *, ffmpeg: str = "ffmpeg", timeout_seconds: float = FFMPEG_TIMEOUT_SECONDS):
        self._ffmpeg = ffmpeg
        self._timeout = timeout_seconds

    def extract_last_frame(self, video: Path, dest: Path) -> int:
        # -sseof seeks relative to the end; -update 1 writes a single image, not a sequence.
        command = [
            self._ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-sseof",
            "-0.1",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-update",
            "1",
            str(dest),
        ]
        try:
            subprocess.run(  # noqa: S603 - fixed argument list, no shell
                command,
                check=True,
                capture_output=True,
                timeout=self._timeout,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise FrameError(FFMPEG_MISSING) from None
        except subprocess.TimeoutExpired:
            dest.unlink(missing_ok=True)
            raise FrameError("ffmpeg timed out extracting the last frame") from None
        except (subprocess.CalledProcessError, OSError):
            dest.unlink(missing_ok=True)
            raise FrameError("ffmpeg failed to extract the last frame") from None
        try:
            size = check_png(dest)
            # The model server reads this file; it may run as another user.
            os.chmod(dest, 0o644)
        except (FrameError, OSError) as exc:
            dest.unlink(missing_ok=True)
            raise FrameError(
                str(exc) if isinstance(exc, FrameError) else "frame file unusable"
            ) from None
        return size
