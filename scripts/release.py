#!/usr/bin/env python3
"""Cut a release: an annotated vX.Y.Z tag on main, pushed alone. Standard library only.

    python scripts/release.py vX.Y.Z [--dry-run] [--remote origin] [--skip-ci-check]

Run on the maintainer's machine from a clean checkout of main. Instances pick the tag up at their
next boot (see docs/instance-runbook.md), so every precondition is checked first and ALL failures
are reported together; nothing is created unless every check passes. `--dry-run` runs the same
read-only checks, prints exactly what it would do, and changes nothing (it does not even fetch).

Preconditions:
  1. the tag is vMAJOR.MINOR.PATCH
  2. the version in pyproject.toml equals the tag without the "v"
  3. the working tree is clean
  4. HEAD is on main and equals the remote's main
  5. requirements.lock is in sync with uv.lock
  6. CI is green for HEAD (via the gh CLI), unless --skip-ci-check
  7. the tag exists neither locally nor on the remote
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

TAG_PATTERN = re.compile(r"v([0-9]{1,6})\.([0-9]{1,6})\.([0-9]{1,6})")
MAIN_BRANCH = "main"
COMMAND_TIMEOUT = 120.0
_PROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
CI_OK = {"success", "skipped"}

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_USAGE = 2


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def __call__(self, argv: Sequence[str], *, cwd: Path, timeout: float) -> CommandResult: ...


def subprocess_runner(argv: Sequence[str], *, cwd: Path, timeout: float) -> CommandResult:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(124, stderr="timed out")
    except OSError as exc:
        return CommandResult(127, stderr=f"cannot run {argv[0]}: {type(exc).__name__}")
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def parse_project_version(pyproject_text: str) -> str | None:
    project = pyproject_text.partition("[project]")[2].partition("\n[")[0]
    match = _PROJECT_VERSION.search(project)
    return match[1] if match else None


class Release:
    def __init__(
        self,
        tag: str,
        *,
        repo: Path,
        remote: str,
        runner: CommandRunner = subprocess_runner,
        skip_ci_check: bool = False,
    ) -> None:
        self.tag = tag
        self.repo = repo
        self.remote = remote
        self._runner = runner
        self._skip_ci_check = skip_ci_check

    def _run(self, *argv: str) -> CommandResult:
        return self._runner(argv, cwd=self.repo, timeout=COMMAND_TIMEOUT)

    def _git(self, *args: str) -> CommandResult:
        return self._run("git", *args)

    def _head(self) -> str:
        return self._git("rev-parse", "HEAD").stdout.strip()

    # --- the preconditions: each returns a problem description, or None when satisfied ---------

    def check_tag_format(self) -> str | None:
        if TAG_PATTERN.fullmatch(self.tag) is None:
            return f"{self.tag!r} is not a release tag (expected vMAJOR.MINOR.PATCH, e.g. v1.2.3)"
        return None

    def check_pyproject_version(self) -> str | None:
        try:
            declared = parse_project_version((self.repo / "pyproject.toml").read_text("utf-8"))
        except OSError:
            return "cannot read pyproject.toml"
        if declared != self.tag[1:]:
            return f"pyproject.toml says version {declared!r}, but the tag is {self.tag}"
        return None

    def check_clean_tree(self) -> str | None:
        status = self._git("status", "--porcelain")
        if status.returncode != 0:
            return "cannot read the git status"
        if status.stdout.strip():
            return "the working tree is not clean (commit, stash or remove your changes)"
        return None

    def check_on_main_and_current(self) -> str | None:
        branch = self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if branch != MAIN_BRANCH:
            return f"HEAD is on {branch!r}, not {MAIN_BRANCH}"
        listed = self._git("ls-remote", self.remote, f"refs/heads/{MAIN_BRANCH}")
        if listed.returncode != 0 or not listed.stdout.strip():
            return f"cannot read {MAIN_BRANCH} on remote {self.remote!r}"
        remote_head = listed.stdout.split()[0]
        if remote_head != self._head():
            return (
                f"local {MAIN_BRANCH} is not the same commit as {self.remote}/{MAIN_BRANCH} "
                "(pull or push first)"
            )
        return None

    def check_lock_in_sync(self) -> str | None:
        script = self.repo / "scripts" / "export_requirements.sh"
        if not script.is_file():
            return "scripts/export_requirements.sh is missing"
        result = self._run("bash", str(script), "--check")
        if result.returncode != 0:
            return (
                "requirements.lock is out of date (run scripts/export_requirements.sh and commit)"
            )
        return None

    def check_ci_is_green(self) -> str | None:
        if self._skip_ci_check:
            return None
        listed = self._run(
            "gh",
            "run",
            "list",
            "--commit",
            self._head(),
            "--json",
            "status,conclusion,name",
            "--limit",
            "50",
        )
        if listed.returncode != 0:
            return (
                "cannot read CI status with gh (is it installed and logged in? or --skip-ci-check)"
            )
        try:
            runs = json.loads(listed.stdout)
        except ValueError:
            return "gh returned something that is not JSON"
        if not isinstance(runs, list) or not runs:
            return "there is no CI run for HEAD yet"
        unfinished = [r for r in runs if isinstance(r, dict) and r.get("status") != "completed"]
        failed = [
            r
            for r in runs
            if isinstance(r, dict)
            and r.get("status") == "completed"
            and r.get("conclusion") not in CI_OK
        ]
        if unfinished:
            return "CI is still running for HEAD"
        if failed:
            return "CI did not pass for HEAD"
        return None

    def check_tag_is_new(self) -> str | None:
        if self._git("rev-parse", "-q", "--verify", f"refs/tags/{self.tag}").returncode == 0:
            return f"the tag {self.tag} already exists locally"
        listed = self._git("ls-remote", "--tags", self.remote, f"refs/tags/{self.tag}")
        if listed.returncode != 0:
            return f"cannot read the tags on remote {self.remote!r}"
        if listed.stdout.strip():
            return f"the tag {self.tag} already exists on {self.remote}"
        return None

    def problems(self) -> list[str]:
        if (bad_format := self.check_tag_format()) is not None:
            return [bad_format]  # every other check builds on a valid tag name
        checks = (
            self.check_pyproject_version,
            self.check_clean_tree,
            self.check_on_main_and_current,
            self.check_lock_in_sync,
            self.check_ci_is_green,
            self.check_tag_is_new,
        )
        return [problem for check in checks if (problem := check()) is not None]

    # --- doing it ------------------------------------------------------------------------------

    def planned_commands(self) -> list[str]:
        return [
            f"git tag -a {self.tag} -m 'Release {self.tag}' <HEAD>",
            f"git push {self.remote} refs/tags/{self.tag}:refs/tags/{self.tag}",
        ]

    def create_and_push(self) -> str | None:
        """Create the annotated tag and push only that tag. Returns a problem or None."""
        head = self._head()
        tagged = self._git("tag", "-a", self.tag, "-m", f"Release {self.tag}", head)
        if tagged.returncode != 0:
            return f"git tag failed: {tagged.stderr.strip()[:200]}"
        pushed = self._git("push", self.remote, f"refs/tags/{self.tag}:refs/tags/{self.tag}")
        if pushed.returncode != 0:
            self._git("tag", "-d", self.tag)  # leave no local tag that the remote does not have
            return f"git push failed, local tag removed: {pushed.stderr.strip()[:200]}"
        return None


def run_release(
    tag: str,
    *,
    repo: Path,
    remote: str,
    dry_run: bool,
    skip_ci_check: bool,
    runner: CommandRunner = subprocess_runner,
    out: TextIO | None = None,
) -> int:
    stream = out if out is not None else sys.stdout
    release = Release(tag, repo=repo, remote=remote, runner=runner, skip_ci_check=skip_ci_check)
    problems = release.problems()
    if problems:
        stream.write(f"release {tag}: blocked, {len(problems)} problem(s):\n")
        for problem in problems:
            stream.write(f"  - {problem}\n")
        return EXIT_BLOCKED
    stream.write(f"release {tag}: all checks passed\n")
    if dry_run:
        stream.write("dry run: nothing was changed. It would run:\n")
        for command in release.planned_commands():
            stream.write(f"  {command}\n")
        return EXIT_OK
    failure = release.create_and_push()
    if failure is not None:
        stream.write(f"release {tag}: {failure}\n")
        return EXIT_BLOCKED
    stream.write(f"release {tag}: tag created and pushed to {remote}\n")
    stream.write("instances pick it up at their next boot\n")
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: CommandRunner = subprocess_runner,
    out: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Cut a release tag of the render agent.")
    parser.add_argument("tag", help="the release tag, vMAJOR.MINOR.PATCH")
    parser.add_argument("--dry-run", action="store_true", help="check and print, change nothing")
    parser.add_argument("--remote", default="origin", help="git remote to push the tag to")
    parser.add_argument(
        "--skip-ci-check", action="store_true", help="do not ask gh whether CI passed for HEAD"
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="repository checkout (default: current directory)",
    )
    args = parser.parse_args(argv)
    return run_release(
        args.tag,
        repo=args.repo,
        remote=args.remote,
        dry_run=args.dry_run,
        skip_ci_check=args.skip_ci_check,
        runner=runner,
        out=out,
    )


if __name__ == "__main__":
    sys.exit(main())
