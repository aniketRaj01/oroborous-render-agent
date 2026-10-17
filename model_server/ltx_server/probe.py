from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

PROBE_TIMEOUT_SECONDS = 20.0


def probe_duration(path: Path) -> float | None:
    """Duration in seconds from ffprobe, or None if it is missing or cannot read the file."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            return None
        value = float(completed.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return value if value > 0 else None
