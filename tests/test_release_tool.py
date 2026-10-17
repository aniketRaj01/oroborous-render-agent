from __future__ import annotations

import io
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import release
from fakes.git_origin import GIT_ENV, git
from release import CommandResult, Release, main, parse_project_version


@dataclass
class Rig:
    """A working checkout whose `origin` is a temporary bare repository. Never the real remote."""

    work: Path
    bare: Path
    lock_ok: bool = True
    gh_result: CommandResult = field(
        default_factory=lambda: CommandResult(
            0, json.dumps([{"status": "completed", "conclusion": "success", "name": "CI"}])
        )
    )
    fail_push: bool = False
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], *, cwd: Path, timeout: float) -> CommandResult:
        self.calls.append(tuple(argv))
        assert cwd == self.work
        if argv[0] == "git":
            if self.fail_push and "push" in argv:
                return CommandResult(1, stderr="remote: permission denied")
            completed = subprocess.run(
                list(argv), cwd=cwd, env=GIT_ENV, capture_output=True, text=True, timeout=timeout
            )
            return CommandResult(completed.returncode, completed.stdout, completed.stderr)
        if argv[0] == "bash":
            return CommandResult(0 if self.lock_ok else 1, stderr="")
        if argv[0] == "gh":
            return self.gh_result
        raise AssertionError(f"unexpected command {argv}")

    def run(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        code = main([*args, "--repo", str(self.work)], runner=self, out=out)
        return code, out.getvalue()

    def remote_refs(self) -> str:
        return git(self.bare, "for-each-ref", "--format=%(refname) %(objecttype) %(objectname)")

    def local_tags(self) -> list[str]:
        return git(self.work, "tag", "--list").splitlines()

    def state(self) -> tuple[str, str, str]:
        return (
            self.remote_refs(),
            git(self.work, "show-ref"),
            git(self.work, "status", "--porcelain"),
        )


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    bare = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(bare)], env=GIT_ENV, check=True
    )
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], env=GIT_ENV, check=True)
    (work / "pyproject.toml").write_text('[project]\nname = "render-agent"\nversion = "1.2.3"\n')
    (work / "scripts").mkdir()
    (work / "scripts" / "export_requirements.sh").write_text("#!/usr/bin/env bash\n")
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "initial")
    git(work, "remote", "add", "origin", str(bare))
    git(work, "push", "-q", "origin", "main")
    return Rig(work, bare)


# --- success and dry run -----------------------------------------------------------------------


def test_a_dry_run_changes_nothing_and_says_what_it_would_do(rig: Rig) -> None:
    before = rig.state()
    code, output = rig.run("v1.2.3", "--dry-run")
    assert code == 0
    assert "all checks passed" in output and "dry run: nothing was changed" in output
    assert (
        "git tag -a v1.2.3" in output
        and "git push origin refs/tags/v1.2.3:refs/tags/v1.2.3" in output
    )
    assert rig.state() == before and rig.local_tags() == []
    mutating = [
        c for c in rig.calls if c[0] == "git" and c[1] in ("tag", "push", "fetch", "commit")
    ]
    assert mutating == []


def test_a_release_creates_exactly_one_annotated_tag_on_the_remote(rig: Rig) -> None:
    head = git(rig.work, "rev-parse", "HEAD")
    refs_before = rig.remote_refs()
    code, output = rig.run("v1.2.3")
    assert code == 0 and "tag created and pushed to origin" in output
    added = set(rig.remote_refs().splitlines()) - set(refs_before.splitlines())
    assert len(added) == 1
    ref, kind, _ = added.pop().split()
    assert ref == "refs/tags/v1.2.3" and kind == "tag"  # an annotated tag object, not a bare ref
    assert git(rig.bare, "rev-parse", "v1.2.3^{commit}") == head
    assert git(rig.bare, "rev-parse", "main") == head
    assert "Release v1.2.3" in git(rig.bare, "tag", "-n", "--list", "v1.2.3")


def test_only_the_tag_is_pushed(rig: Rig) -> None:
    git(rig.work, "branch", "other")
    rig.run("v1.2.3")
    pushes = [c for c in rig.calls if c[0] == "git" and "push" in c]
    assert pushes == [("git", "push", "origin", "refs/tags/v1.2.3:refs/tags/v1.2.3")]
    assert "refs/heads/other" not in rig.remote_refs()


def test_the_checks_do_not_modify_the_repository(rig: Rig) -> None:
    before = rig.state()
    Release("v1.2.3", repo=rig.work, remote="origin", runner=rig).problems()
    assert rig.state() == before


# --- every precondition blocks, and everything is reported together ----------------------------


def assert_blocked(rig: Rig, tag: str, expected: str, *extra: str) -> str:
    before = rig.state()
    code, output = rig.run(tag, *extra)
    assert code == 1 and "blocked" in output and expected in output, output
    assert rig.state() == before and rig.local_tags() == []  # nothing created
    return output


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2", "v1.2.3-rc1", "V1.2.3", "v1.2.3.4", "main"])
def test_the_tag_format_is_checked(rig: Rig, tag: str) -> None:
    assert_blocked(rig, tag, "is not a release tag")


def test_pyproject_version_must_equal_the_tag(rig: Rig) -> None:
    assert_blocked(rig, "v1.2.4", "pyproject.toml says version '1.2.3', but the tag is v1.2.4")


def test_a_dirty_tree_blocks(rig: Rig) -> None:
    (rig.work / "scratch.txt").write_text("uncommitted")
    assert_blocked(rig, "v1.2.3", "working tree is not clean")
    (rig.work / "scratch.txt").unlink()
    (rig.work / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n# edit\n')
    assert_blocked(rig, "v1.2.3", "working tree is not clean")


def test_not_being_on_main_blocks(rig: Rig) -> None:
    git(rig.work, "checkout", "-q", "-b", "feature")
    assert_blocked(rig, "v1.2.3", "HEAD is on 'feature', not main")


def test_a_local_main_ahead_of_the_remote_blocks(rig: Rig) -> None:
    (rig.work / "new.txt").write_text("x")
    git(rig.work, "add", "-A")
    git(rig.work, "commit", "-q", "-m", "unpushed")
    assert_blocked(rig, "v1.2.3", "is not the same commit as origin/main")


def test_a_local_main_behind_the_remote_blocks(rig: Rig, tmp_path: Path) -> None:
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", str(rig.bare), str(other)], env=GIT_ENV, check=True)
    (other / "more.txt").write_text("x")
    git(other, "add", "-A")
    git(other, "commit", "-q", "-m", "someone else pushed")
    git(other, "push", "-q", "origin", "main")
    assert_blocked(rig, "v1.2.3", "is not the same commit as origin/main")


def test_a_stale_lock_file_blocks(rig: Rig) -> None:
    rig.lock_ok = False
    assert_blocked(rig, "v1.2.3", "requirements.lock is out of date")


def test_a_missing_export_script_blocks(rig: Rig) -> None:
    (rig.work / "scripts" / "export_requirements.sh").unlink()
    git(rig.work, "commit", "-q", "-am", "remove the script")
    git(rig.work, "push", "-q", "origin", "main")
    assert_blocked(rig, "v1.2.3", "export_requirements.sh is missing")


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            CommandResult(0, json.dumps([{"status": "completed", "conclusion": "failure"}])),
            "did not pass",
        ),
        (
            CommandResult(0, json.dumps([{"status": "in_progress", "conclusion": ""}])),
            "still running",
        ),
        (
            CommandResult(
                0,
                json.dumps(
                    [
                        {"status": "completed", "conclusion": "success"},
                        {"status": "completed", "conclusion": "cancelled"},
                    ]
                ),
            ),
            "did not pass",
        ),
        (CommandResult(0, "[]"), "no CI run for HEAD yet"),
        (CommandResult(0, "not json"), "not JSON"),
        (CommandResult(127, stderr="cannot run gh"), "cannot read CI status with gh"),
        (CommandResult(1, stderr="not logged in"), "cannot read CI status with gh"),
    ],
)
def test_ci_must_be_green_for_head(rig: Rig, result: CommandResult, expected: str) -> None:
    rig.gh_result = result
    assert_blocked(rig, "v1.2.3", expected)


def test_skipping_the_ci_check_never_calls_gh(rig: Rig) -> None:
    rig.gh_result = CommandResult(1, stderr="would fail")
    code, _ = rig.run("v1.2.3", "--skip-ci-check")
    assert code == 0 and not any(c[0] == "gh" for c in rig.calls)


def test_ci_is_asked_about_the_head_commit(rig: Rig) -> None:
    rig.run("v1.2.3", "--dry-run")
    gh = next(c for c in rig.calls if c[0] == "gh")
    assert gh[gh.index("--commit") + 1] == git(rig.work, "rev-parse", "HEAD")


def test_an_existing_local_tag_blocks(rig: Rig) -> None:
    git(rig.work, "tag", "v1.2.3")
    before = rig.remote_refs()
    code, output = rig.run("v1.2.3")
    assert code == 1 and "already exists locally" in output and rig.remote_refs() == before


def test_an_existing_remote_tag_blocks(rig: Rig) -> None:
    git(rig.work, "tag", "v1.2.3")
    git(rig.work, "push", "-q", "origin", "v1.2.3")
    git(rig.work, "tag", "-d", "v1.2.3")
    code, output = rig.run("v1.2.3")
    assert code == 1 and "already exists on origin" in output and rig.local_tags() == []


def test_every_failing_check_is_reported_in_one_run(rig: Rig) -> None:
    rig.lock_ok = False
    rig.gh_result = CommandResult(0, json.dumps([{"status": "completed", "conclusion": "failure"}]))
    (rig.work / "dirty.txt").write_text("x")
    git(rig.work, "checkout", "-q", "-b", "feature")
    code, output = rig.run("v1.2.4")
    assert code == 1 and "5 problem(s)" in output
    for expected in ("pyproject.toml says", "not clean", "not main", "out of date", "did not pass"):
        assert expected in output


def test_a_failed_push_removes_the_local_tag(rig: Rig) -> None:
    rig.fail_push = True
    code, output = rig.run("v1.2.3")
    assert code == 1 and "git push failed, local tag removed" in output
    assert rig.local_tags() == [] and "refs/tags/v1.2.3" not in rig.remote_refs()


def test_an_unreachable_remote_blocks(rig: Rig) -> None:
    git(rig.work, "remote", "set-url", "origin", str(rig.work.parent / "does-not-exist.git"))
    code, output = rig.run("v1.2.3")
    assert code == 1 and "cannot read main on remote" in output and rig.local_tags() == []


def test_the_real_runner_reports_a_missing_program_instead_of_crashing(tmp_path: Path) -> None:
    result = release.subprocess_runner(["definitely-not-a-program-xyz"], cwd=tmp_path, timeout=5)
    assert result.returncode == 127 and "cannot run" in result.stderr


def test_pyproject_version_parsing_reads_only_the_project_table() -> None:
    text = '[tool.a]\nversion = "9"\n[project]\nversion = "1.2.3"\n[tool.b]\nversion = "7"\n'
    assert parse_project_version(text) == "1.2.3"
