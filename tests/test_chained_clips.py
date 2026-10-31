"""Chained plans: clip N starts from the last frame of clip N-1."""

from __future__ import annotations

import json
from pathlib import Path

from fakes.fake_frames import FAKE_PNG
from fakes.fake_job_api import FAKE_BUCKET
from fakes.fake_model_server import Reject
from render_agent.frames import FrameError
from render_agent.uploader import UploadError
from rig import Rig, make_rig


def chained_rig(tmp_path: Path, clips: int = 3, **overrides: str) -> tuple[Rig, list[int]]:
    rig = make_rig(tmp_path, **overrides)
    plan = rig.fake_api.add_plan(1, clips=clips)
    rig.fake_api.chain(plan)
    return rig, [job.job_id for job in plan.jobs]


def test_each_clip_starts_from_the_previous_clips_last_frame(tmp_path: Path) -> None:
    rig, ids = chained_rig(tmp_path)
    result = rig.run()
    assert (result.done, result.failed) == (3, 0)
    assert [r.job_id for r in rig.model.requests] == ids
    real_scratch = rig.scratch.resolve()
    assert [r.condition_image_path for r in rig.model.requests] == [
        None,
        str(real_scratch / "p1-0-last.png"),
        str(real_scratch / "p1-1-last.png"),
    ]
    assert rig.uploader.downloads == []  # the frames were still on disk
    frames = [u for u in rig.uploader.uploads if u.content_type == "image/png"]
    assert [u.key for u in frames] == [
        "plans/1/clip-0-last.png",
        "plans/1/clip-1-last.png",
    ]
    assert all(u.content_type == "video/mp4" for u in rig.uploader.uploads if u not in frames)
    plan = rig.fake_api.plans[0]
    assert plan.jobs[0].last_frame_reported == {
        "bucket": FAKE_BUCKET,
        "key": "plans/1/clip-0-last.png",
        "sizeBytes": len(FAKE_PNG),
    }
    assert plan.jobs[2].last_frame_reported is None  # the last clip has no frame to hand on
    assert rig.leftover_files() == []


def test_a_resumed_chain_downloads_the_start_frame_once_then_reuses_local_frames(
    tmp_path: Path,
) -> None:
    rig, _ids = chained_rig(tmp_path)
    plan = rig.fake_api.plans[0]
    plan.jobs[0].status = "done"
    rig.uploader.objects[(FAKE_BUCKET, "plans/1/clip-0-last.png")] = FAKE_PNG
    result = rig.run()
    assert result.done == 2
    assert rig.uploader.downloads == [(FAKE_BUCKET, "plans/1/clip-0-last.png")]
    real_scratch = rig.scratch.resolve()
    assert [r.condition_image_path for r in rig.model.requests] == [
        str(real_scratch / "p1-start-1.png"),
        str(real_scratch / "p1-1-last.png"),
    ]
    assert rig.leftover_files() == []


def test_a_missing_start_frame_fails_the_clip_and_never_renders_it_from_text(
    tmp_path: Path,
) -> None:
    rig, _ids = chained_rig(tmp_path)
    rig.fake_api.plans[0].jobs[0].status = "done"  # its frame object does not exist
    result = rig.run()
    assert result.done == 0 and result.failed >= 1
    assert rig.model.requests == []
    job = rig.fake_api.plans[0].jobs[1]
    assert job.last_fail_code == "render_error"
    assert "start frame" in (job.last_fail_message or "")


def test_later_chained_clips_are_not_started_after_a_failure(tmp_path: Path) -> None:
    rig, ids = chained_rig(tmp_path)
    rig.model.script(ids[1], Reject("render_error"), Reject("render_error"), Reject("render_error"))
    result = rig.run()
    assert result.done == 1  # clip 0
    assert ids[2] not in [r.job_id for r in rig.model.requests]
    assert result.skipped >= 1
    assert rig.fake_api.plans[0].jobs[2].status == "pending"
    assert rig.leftover_files() == []


def test_a_failed_frame_extraction_is_an_upload_error_and_the_clip_is_not_completed(
    tmp_path: Path,
) -> None:
    rig, _ids = chained_rig(tmp_path, clips=2)
    rig.frames.failures.append(FrameError("ffmpeg failed to extract the last frame"))
    result = rig.run()
    job = rig.fake_api.plans[0].jobs[0]
    # reported as failed (the clip was not completed without its frame), then retried and done
    assert result.failed == 1 and job.last_fail_code == "upload_error"
    assert "last frame" in (job.last_fail_message or "")
    assert job.attempts == 2 and result.done == 2


def test_a_frame_upload_failure_is_an_upload_error(tmp_path: Path) -> None:
    rig, _ids = chained_rig(tmp_path, clips=2)
    original = rig.uploader.upload

    def failing_frame_upload(path: Path, **kwargs: str) -> int:
        if kwargs.get("content_type") == "image/png":
            raise UploadError("S3 request failed (AccessDenied)")
        return original(path, **kwargs)

    rig.uploader.upload = failing_frame_upload  # type: ignore[method-assign]
    result = rig.run()
    job = rig.fake_api.plans[0].jobs[0]
    assert result.done == 0 and job.last_fail_code == "upload_error"
    assert job.status != "done"


def test_the_reupload_on_size_mismatch_covers_the_frame(tmp_path: Path) -> None:
    rig, _ids = chained_rig(tmp_path, clips=2)
    rig.uploader.register_wrong_size_times = 2  # the clip and its frame both register wrongly
    result = rig.run()
    assert result.done == 2
    assert [u.content_type for u in rig.uploader.uploads[:4]] == [
        "video/mp4",
        "image/png",
        "video/mp4",
        "image/png",
    ]


def test_clips_without_chain_fields_behave_as_before(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    rig.fake_api.add_plan(1, clips=2)
    result = rig.run()
    assert result.done == 2
    assert all(r.condition_image_path is None for r in rig.model.requests)
    assert all(u.content_type == "video/mp4" for u in rig.uploader.uploads)
    assert rig.frames.extracted == []
    complete_bodies = [json.loads(c.body) for c in rig.fake_api.calls("/v1/complete")]
    assert all("lastFrame" not in body for body in complete_bodies)
