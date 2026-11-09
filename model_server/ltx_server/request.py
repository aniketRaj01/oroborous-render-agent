"""Validation of POST /generate bodies, exactly as docs/model-server-contract.md specifies."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ltx_server.config import Settings, valid_dimension, valid_frame_count
from ltx_server.renderer import ConditionImage

ALLOWED_FIELDS = frozenset(
    {"jobId", "prompt", "params", "outputPath", "conditionImagePath", "conditionImages"}
)
OPTIONAL_FIELDS = frozenset({"conditionImagePath", "conditionImages"})
ALLOWED_IMAGE_KEYS = frozenset({"path", "frameIdx", "strength"})
MAX_CONDITION_IMAGES = 3
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
    # Sorted by frame index. `conditionImagePath` + `conditionStrength` become one entry at frame 0.
    condition_images: tuple[ConditionImage, ...] = ()


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


def resolve_condition_image_path(
    raw: str, allowed_dir: Path, name: str = "conditionImagePath"
) -> Path:
    """The real location of a condition image; same confinement as the output path."""
    if "\x00" in raw or not os.path.isabs(raw):
        raise RequestError(f"{name} must be an absolute path")
    if not raw.lower().endswith(IMAGE_SUFFIXES):
        raise RequestError(f"{name} must end in .png, .jpg or .jpeg")
    root = Path(os.path.realpath(allowed_dir))
    target = Path(os.path.realpath(raw))
    if root not in target.parents:
        raise RequestError(f"{name} must be inside the allowed output directory")
    if not target.is_file():
        raise RequestError(f"{name} does not exist or is not a regular file")
    if not os.access(target, os.R_OK):
        raise RequestError(f"{name} is not readable by the server")
    return target


def parse_condition_images(
    raw: object, allowed_dir: Path, num_frames: int
) -> tuple[ConditionImage, ...]:
    """The `conditionImages` list: at most MAX_CONDITION_IMAGES, distinct frame indexes."""
    if not isinstance(raw, list):
        raise RequestError("conditionImages must be a list")
    if len(raw) > MAX_CONDITION_IMAGES:
        raise RequestError(f"conditionImages takes at most {MAX_CONDITION_IMAGES} entries")
    images: list[ConditionImage] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise RequestError("each conditionImages entry must be an object")
        unknown = sorted(set(entry) - ALLOWED_IMAGE_KEYS)
        if unknown:
            raise RequestError(
                f"unknown conditionImages key(s): {', '.join(k[:40] for k in unknown)}"
            )
        if "path" not in entry or "frameIdx" not in entry:
            raise RequestError("each conditionImages entry needs path and frameIdx")
        path, frame_idx, strength = entry["path"], entry["frameIdx"], entry.get("strength", 1.0)
        if not isinstance(path, str):
            raise RequestError("conditionImages path must be a string")
        # LTX only requires the index to lie inside the clip (frame 0 replaces the first latent,
        # any other index becomes a keyframe at that pixel frame); it does not need multiples of 8.
        if not _is_int(frame_idx) or not 0 <= frame_idx < num_frames:
            raise RequestError(
                "conditionImages frameIdx must be an integer from 0 to numFrames - 1"
            )
        if not _is_number(strength) or not 0 < strength <= 1:
            raise RequestError("conditionImages strength must be greater than 0 and at most 1")
        images.append(
            ConditionImage(
                resolve_condition_image_path(path, allowed_dir, "conditionImages path"),
                frame_idx,
                float(strength),
            )
        )
    if len({image.frame_idx for image in images}) != len(images):
        raise RequestError("conditionImages must not repeat a frameIdx")
    return tuple(sorted(images, key=lambda image: image.frame_idx))


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
    strength = params.get("conditionStrength")
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
    if strength is not None and (not _is_number(strength) or not 0 < strength <= 1):
        raise RequestError("conditionStrength must be greater than 0 and at most 1")
    if "conditionImages" in data and condition_image is not None:
        raise RequestError("send either conditionImagePath or conditionImages, not both")
    if "conditionImages" in data and strength is not None:
        raise RequestError("conditionStrength goes with conditionImagePath; use strength per entry")

    final_path = resolve_output_path(output, settings.allowed_output_dir)
    if "conditionImages" in data:
        images = parse_condition_images(
            data["conditionImages"], settings.allowed_output_dir, num_frames
        )
    elif condition_image is not None:
        images = (
            ConditionImage(
                resolve_condition_image_path(condition_image, settings.allowed_output_dir),
                0,
                float(strength if strength is not None else 1.0),
            ),
        )
    else:
        images = ()
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
        condition_images=images,
    )
