"""Validation of POST /generate bodies, exactly as docs/model-server-contract.md specifies."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ltx_server.config import Settings, valid_dimension, valid_frame_count

ALLOWED_FIELDS = frozenset({"jobId", "prompt", "params", "outputPath", "conditionImagePath"})
OPTIONAL_FIELDS = frozenset({"conditionImagePath"})
ALLOWED_PARAMS = frozenset({"width", "height", "numFrames", "fps", "seed", "conditionStrength"})
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
MAX_SEED = 2**63 - 1


class RequestError(Exception):
    """The request is invalid. The message is safe to send back: it never quotes the input."""


@dataclass(frozen=True)
class GenerateInput:
    job_id: int
    prompt: str = field(repr=False)
    width: int
    height: int
    num_frames: int
    fps: float
    seed: int
    output_path: str
    final_path: Path
    condition_image_path: Path | None = None
    condition_strength: float = 1.0


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def resolve_output_path(raw: str, allowed_dir: Path) -> Path:
    """The real location of `raw`; it must lie inside `allowed_dir` once symlinks are resolved."""
    if "\x00" in raw or not os.path.isabs(raw):
        raise RequestError("outputPath must be an absolute path")
    if not raw.endswith(".mp4") or raw.endswith("/.mp4"):
        raise RequestError("outputPath must end in .mp4")
    root = Path(os.path.realpath(allowed_dir))
    target = Path(os.path.realpath(raw))
    if root not in target.parents:
        raise RequestError("outputPath must be inside the allowed output directory")
    if not target.parent.is_dir():
        raise RequestError("the output directory does not exist")
    if target.exists() and not target.is_file():
        raise RequestError("outputPath is not a regular file")
    return target


def resolve_condition_image_path(raw: str, allowed_dir: Path) -> Path:
    """The real location of the first-frame image; same confinement as the output path."""
    if "\x00" in raw or not os.path.isabs(raw):
        raise RequestError("conditionImagePath must be an absolute path")
    if not raw.lower().endswith(IMAGE_SUFFIXES):
        raise RequestError("conditionImagePath must end in .png, .jpg or .jpeg")
    root = Path(os.path.realpath(allowed_dir))
    target = Path(os.path.realpath(raw))
    if root not in target.parents:
        raise RequestError("conditionImagePath must be inside the allowed output directory")
    if not target.is_file():
        raise RequestError("conditionImagePath does not exist or is not a regular file")
    if not os.access(target, os.R_OK):
        raise RequestError("conditionImagePath is not readable by the server")
    return target


def parse_request(body: bytes, settings: Settings) -> GenerateInput:
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        raise RequestError("the body is not valid JSON") from None
    if not isinstance(data, dict):
        raise RequestError("the body must be a JSON object")
    unknown = sorted(set(data) - ALLOWED_FIELDS)
    if unknown:
        raise RequestError(f"unknown field(s): {', '.join(k[:40] for k in unknown)}")
    missing = sorted(ALLOWED_FIELDS - OPTIONAL_FIELDS - set(data))
    if missing:
        raise RequestError(f"missing field(s): {', '.join(missing)}")

    job_id, prompt, params, output = (
        data["jobId"],
        data["prompt"],
        data["params"],
        data["outputPath"],
    )
    if not _is_int(job_id):
        raise RequestError("jobId must be an integer")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RequestError("prompt must be a non-empty string")
    if not isinstance(params, dict):
        raise RequestError("params must be an object")
    unknown_params = sorted(set(params) - ALLOWED_PARAMS)
    if unknown_params:
        raise RequestError(f"unknown params: {', '.join(k[:40] for k in unknown_params)}")
    if not isinstance(output, str):
        raise RequestError("outputPath must be a string")
    condition_image = data.get("conditionImagePath")
    if condition_image is not None and not isinstance(condition_image, str):
        raise RequestError("conditionImagePath must be a string")

    width = params.get("width", settings.default_width)
    height = params.get("height", settings.default_height)
    num_frames = params.get("numFrames", settings.default_num_frames)
    fps = params.get("fps", settings.default_fps)
    seed = params.get("seed", settings.default_seed)
    strength = params.get("conditionStrength", 1.0)
    if not _is_int(width) or not valid_dimension(width):
        raise RequestError("width must be a positive multiple of 64")
    if not _is_int(height) or not valid_dimension(height):
        raise RequestError("height must be a positive multiple of 64")
    if not _is_int(num_frames) or not valid_frame_count(num_frames, settings.max_frames):
        raise RequestError(f"numFrames must be of the form 8k+1, from 9 to {settings.max_frames}")
    if not _is_number(fps) or not 0 < fps <= 60:
        raise RequestError("fps must be greater than 0 and at most 60")
    if not _is_int(seed) or not 0 <= seed <= MAX_SEED:
        raise RequestError("seed must be a non-negative integer")
    if not _is_number(strength) or not 0 < strength <= 1:
        raise RequestError("conditionStrength must be greater than 0 and at most 1")

    final_path = resolve_output_path(output, settings.allowed_output_dir)
    image_path = (
        resolve_condition_image_path(condition_image, settings.allowed_output_dir)
        if condition_image is not None
        else None
    )
    return GenerateInput(
        job_id=job_id,
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=float(fps),
        seed=seed,
        output_path=output,
        final_path=final_path,
        condition_image_path=image_path,
        condition_strength=float(strength),
    )
