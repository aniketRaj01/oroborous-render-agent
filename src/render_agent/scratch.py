from __future__ import annotations

import logging
import os
import re
import stat
from pathlib import Path

from render_agent.config import ConfigError

logger = logging.getLogger(__name__)

_CLIP_FILE = re.compile(r"^(\d+-\d+\.mp4|p\d+-(\d+-last|start-\d+)\.png)$")


def prepare_scratch_dir(path: Path) -> None:
    """Create the directory 0700 and remove clip files left over from a previous run.

    Only files named <jobId>-<attempt>.mp4, or the frames of chained clips
    (p<planId>-<n>-last.png, p<planId>-start-<n>.png), are removed, so a mistaken SCRATCH_DIR
    cannot delete anything else.
    """
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ConfigError([("SCRATCH_DIR", "scratch_dir_not_a_directory")])
        if info.st_uid != os.getuid():
            raise ConfigError([("SCRATCH_DIR", "scratch_dir_not_owned_by_us")])
        path.chmod(0o700)
        removed = 0
        for entry in path.iterdir():
            if entry.is_file() and not entry.is_symlink() and _CLIP_FILE.match(entry.name):
                entry.unlink()
                removed += 1
    except OSError:
        raise ConfigError([("SCRATCH_DIR", "scratch_dir_unusable")]) from None
    if removed:
        logger.info("scratch_cleaned", extra={"removed_files": removed})
