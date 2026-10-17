from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from render_agent.config import ConfigError
from render_agent.scratch import prepare_scratch_dir


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_creates_the_directory_with_mode_0700(tmp_path: Path) -> None:
    target = tmp_path / "a" / "scratch"
    prepare_scratch_dir(target)
    assert target.is_dir() and mode(target) == 0o700


def test_existing_open_directory_is_tightened(tmp_path: Path) -> None:
    target = tmp_path / "scratch"
    target.mkdir()
    os.chmod(target, 0o755)
    prepare_scratch_dir(target)
    assert mode(target) == 0o700


def test_removes_only_stale_clip_files(tmp_path: Path) -> None:
    target = tmp_path / "scratch"
    target.mkdir()
    for name in ("412-1.mp4", "9-12.mp4", "notes.txt", "412.mp4", "x-1.mp4", "412-1.mp4.part"):
        (target / name).write_bytes(b"x")
    (target / "subdir").mkdir()
    prepare_scratch_dir(target)
    assert sorted(p.name for p in target.iterdir()) == [
        "412-1.mp4.part",
        "412.mp4",
        "notes.txt",
        "subdir",
        "x-1.mp4",
    ]


def test_symlinked_or_non_directory_paths_are_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ConfigError) as caught:
        prepare_scratch_dir(link)
    assert caught.value.problems == (("SCRATCH_DIR", "scratch_dir_not_a_directory"),)
    file_path = tmp_path / "file"
    file_path.write_text("x")
    with pytest.raises(ConfigError):
        prepare_scratch_dir(file_path)
