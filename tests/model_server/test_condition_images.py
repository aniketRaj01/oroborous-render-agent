"""conditionImages (several images at different frames) and the conditionImagePath shorthand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ltx_server.ltx_binding import LtxBinding
from ltx_server.renderer import ConditionImage, RenderRequest
from ltx_server.request import MAX_CONDITION_IMAGES, RequestError, parse_request
from ltx_stubs import ImageConditioningInput, install
from ms_helpers import generate_body, make_settings

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64


def parse(tmp_path: Path, **overrides: Any) -> Any:
    settings = make_settings(tmp_path)
    for name in ("anchor.png", "mid.png", "last.jpg"):
        (settings.allowed_output_dir / name).write_bytes(PNG)
    out = str(settings.allowed_output_dir / "412-1.mp4")
    return parse_request(json.dumps(generate_body(out, **overrides)).encode(), settings), settings


def entry(settings_dir: Path, name: str, frame_idx: Any, **extra: Any) -> dict[str, Any]:
    return {"path": str(settings_dir / name), "frameIdx": frame_idx, **extra}


def test_images_at_several_frames_are_parsed_and_sorted_by_frame(tmp_path: Path) -> None:
    out = make_settings(tmp_path).allowed_output_dir
    images = [entry(out, "mid.png", 300, strength=0.5), entry(out, "anchor.png", 0)]
    parsed, _ = parse(tmp_path, conditionImages=images)
    assert [(i.path.name, i.frame_idx, i.strength) for i in parsed.condition_images] == [
        ("anchor.png", 0, 1.0),
        ("mid.png", 300, 0.5),
    ]


def test_the_single_image_shorthand_is_one_entry_at_frame_zero(tmp_path: Path) -> None:
    out = make_settings(tmp_path).allowed_output_dir
    parsed, _ = parse(
        tmp_path,
        conditionImagePath=str(out / "anchor.png"),
        params={"numFrames": 361, "conditionStrength": 0.7},
    )
    assert parsed.condition_images == (ConditionImage(out.resolve() / "anchor.png", 0, 0.7),)
    plain, _ = parse(tmp_path)
    assert plain.condition_images == ()


def rejected(tmp_path: Path, **overrides: Any) -> str:
    with pytest.raises(RequestError) as caught:
        parse(tmp_path, **overrides)
    return str(caught.value)


def test_invalid_condition_image_requests_are_refused_without_quoting_input(
    tmp_path: Path,
) -> None:
    out = make_settings(tmp_path).allowed_output_dir
    good = entry(out, "anchor.png", 0)
    assert "not both" in rejected(
        tmp_path, conditionImagePath=str(out / "anchor.png"), conditionImages=[good]
    )
    assert "conditionStrength" in rejected(
        tmp_path, conditionImages=[good], params={"numFrames": 361, "conditionStrength": 0.5}
    )
    assert "repeat" in rejected(tmp_path, conditionImages=[good, entry(out, "mid.png", 0)])
    cap = [entry(out, "anchor.png", 8 * i) for i in range(MAX_CONDITION_IMAGES + 1)]
    assert "at most" in rejected(tmp_path, conditionImages=cap)
    for frame_idx in (-1, 361, 1.5, True, "0"):
        assert "frameIdx" in rejected(tmp_path, conditionImages=[entry(out, "mid.png", frame_idx)])
    assert "strength" in rejected(tmp_path, conditionImages=[entry(out, "mid.png", 8, strength=0)])
    assert "unknown" in rejected(tmp_path, conditionImages=[{**good, "crf": 1}])
    assert "path and frameIdx" in rejected(tmp_path, conditionImages=[{"path": good["path"]}])
    assert "list" in rejected(tmp_path, conditionImages={"path": "x"})
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    message = rejected(tmp_path, conditionImages=[{"path": str(outside), "frameIdx": 0}])
    assert "inside the allowed output directory" in message and str(tmp_path) not in message
    assert "does not exist" in rejected(tmp_path, conditionImages=[entry(out, "nope.png", 0)])
    assert "conditionImages path must end" in rejected(
        tmp_path, conditionImages=[{"path": str(out / "a.gif"), "frameIdx": 0}]
    )


def test_the_frame_index_may_be_the_last_frame_and_any_alignment(tmp_path: Path) -> None:
    out = make_settings(tmp_path).allowed_output_dir
    parsed, _ = parse(
        tmp_path, conditionImages=[entry(out, "mid.png", 360), entry(out, "anchor.png", 13)]
    )
    assert [i.frame_idx for i in parsed.condition_images] == [13, 360]


def test_the_binding_builds_one_input_per_image_and_none_without(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    binding = LtxBinding(make_settings(tmp_path))

    def make(images: tuple[ConditionImage, ...] = ()) -> RenderRequest:
        return RenderRequest(1, "p", 576, 1024, 361, 24.0, 1, condition_images=images)

    plain = make()
    assert binding._condition_images(plain) == []
    images = (
        ConditionImage(tmp_path / "anchor.png", 0, 1.0),
        ConditionImage(tmp_path / "mid.png", 300, 0.6),
    )
    request = make(images)
    assert binding._condition_images(request) == [
        ImageConditioningInput(str(tmp_path / "anchor.png"), 0, 1.0),
        ImageConditioningInput(str(tmp_path / "mid.png"), 300, 0.6),
    ]
