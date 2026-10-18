#!/usr/bin/env python3
"""Compare two mp4 files on the instance: does resident mode render what pipeline mode renders?

    <ltx-venv>/bin/python model_server/tools/compare_clips.py A.mp4 B.mp4 [--samples 5]

Prints, for each file, the frame count, resolution, duration and audio; then for a few evenly spaced
frames the mean absolute difference (0-255 scale) and the PSNR in dB. Uses only `av` and `numpy`,
which the LTX virtualenv has (LTX's own encoder imports both). Exit status: 0 when both files were
read, 1 when they could not be read or differ in frame count or size, 2 on bad arguments. Whether
the numbers count as "close" is the operator's call: see docs/model-server-runbook.md.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClipInfo:
    frames: int
    width: int
    height: int
    duration: float
    has_audio: bool
    sampled: dict[int, Any]


def sample_indices(frames: int, samples: int) -> list[int]:
    if frames <= 0:
        return []
    if samples <= 1 or frames == 1:
        return [0]
    return sorted({round(i * (frames - 1) / (samples - 1)) for i in range(samples)})


def read_clip(path: str, wanted: Sequence[int]) -> ClipInfo:
    import av

    wanted_set = set(wanted)
    sampled: dict[int, Any] = {}
    with av.open(path) as container:
        video = container.streams.video[0]
        has_audio = len(container.streams.audio) > 0
        count = 0
        for frame in container.decode(video):
            if count in wanted_set:
                sampled[count] = frame.to_ndarray(format="rgb24")
            count += 1
        duration = float(container.duration) / 1_000_000 if container.duration else 0.0
        return ClipInfo(count, video.width, video.height, duration, has_audio, sampled)


def frame_count(path: str) -> int:
    import av

    with av.open(path) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))


def difference(a: Any, b: Any) -> tuple[float, float]:
    """(mean absolute difference, PSNR in dB; inf when identical)."""
    import numpy as np

    x, y = a.astype(np.float64), b.astype(np.float64)
    mad = float(np.mean(np.abs(x - y)))
    mse = float(np.mean((x - y) ** 2))
    return mad, (math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse))


def describe(label: str, info: ClipInfo) -> None:
    print(
        f"{label}: {info.frames} frames, {info.width}x{info.height}, "
        f"{info.duration:.2f} s, audio={'yes' if info.has_audio else 'NO'}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two mp4 files frame by frame.")
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--samples", type=int, default=5, help="frames to compare (default 5)")
    args = parser.parse_args(argv)
    if args.samples < 1:
        print("--samples must be at least 1", file=sys.stderr)
        return 2
    try:
        indices = sample_indices(frame_count(args.first), args.samples)
        a = read_clip(args.first, indices)
        b = read_clip(args.second, indices)
    except ImportError:
        print(
            "this tool needs av and numpy: run it with the LTX virtualenv's python", file=sys.stderr
        )
        return 2
    except Exception as exc:  # unreadable or corrupt input: say which kind, never a stack of paths
        print(f"cannot read the clips: {type(exc).__name__}", file=sys.stderr)
        return 1
    describe("A", a)
    describe("B", b)
    if (a.frames, a.width, a.height) != (b.frames, b.width, b.height):
        print("DIFFERENT frame count or size: the clips cannot be compared frame by frame")
        return 1
    print(f"{'frame':>6} {'mean abs diff':>14} {'PSNR dB':>9}")
    worst = 0.0
    for index in indices:
        mad, psnr = difference(a.sampled[index], b.sampled[index])
        worst = max(worst, mad)
        print(f"{index:>6} {mad:>14.3f} {psnr:>9.2f}")
    print(f"largest mean abs diff: {worst:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
