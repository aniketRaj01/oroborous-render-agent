"""A real git repository on local disk that plays the part of the public origin."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

GIT_ENV = {
    **{k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR")},
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "Test Author",
    "GIT_AUTHOR_EMAIL": "author@example.invalid",
    "GIT_COMMITTER_NAME": "Test Committer",
    "GIT_COMMITTER_EMAIL": "committer@example.invalid",
    "GIT_TERMINAL_PROMPT": "0",
}

LOCK_A = "example-dependency==1.0.0 \\\n    --hash=sha256:" + "a" * 64 + "\n"
LOCK_B = "example-dependency==2.0.0 \\\n    --hash=sha256:" + "b" * 64 + "\n"


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        env=GIT_ENV,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


class Origin:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True)
        subprocess.run(
            ["git", "init", "-q", "-b", "main", str(path)], env=GIT_ENV, check=True, timeout=60
        )
        self.counter = 0

    @property
    def url(self) -> str:
        return f"file://{self.path}"

    def commit(
        self, version: str | None = None, lock: str = LOCK_A, message: str = "change"
    ) -> str:
        """A commit on the current branch. `version` goes into pyproject.toml."""
        self.counter += 1
        if version is not None:
            (self.path / "pyproject.toml").write_text(
                f'[project]\nname = "render-agent"\nversion = "{version}"\n\n'
                '[tool.other]\nversion = "9.9.9"\n'
            )
        (self.path / "requirements.lock").write_text(lock)
        package = self.path / "src" / "render_agent"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(f"# change {self.counter}\n")
        git(self.path, "add", "-A")
        git(self.path, "commit", "-q", "-m", f"{message} {self.counter}")
        return git(self.path, "rev-parse", "HEAD")

    def release(self, version: str, *, lock: str = LOCK_A, tag: str | None = None) -> str:
        """Commit `version` on main and tag it vVERSION (or `tag`). Returns the commit."""
        sha = self.commit(version, lock, message=f"release {version}")
        git(self.path, "tag", "-a", tag or f"v{version}", "-m", f"Release {version}")
        return sha

    def tag(self, name: str, sha: str = "HEAD", *, annotated: bool = True) -> None:
        if annotated:
            git(self.path, "tag", "-a", name, sha, "-m", name)
        else:
            git(self.path, "tag", name, sha)

    def delete_tag(self, name: str) -> None:
        git(self.path, "tag", "-d", name)

    def move_tag(self, name: str, sha: str) -> None:
        git(self.path, "tag", "-f", "-a", name, sha, "-m", name)

    def side_branch_release(self, version: str, *, lock: str = LOCK_A) -> str:
        """A tag on a commit that is not (and never becomes) part of main."""
        git(self.path, "checkout", "-q", "-b", f"side-{version}")
        sha = self.release(version, lock=lock)
        git(self.path, "checkout", "-q", "main")
        return sha
