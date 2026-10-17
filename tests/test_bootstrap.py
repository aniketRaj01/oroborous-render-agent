from __future__ import annotations

import errno
import hashlib
import io
import os
import sys
import tarfile
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

import bootstrap
from boot_world import World, lock_n
from bootstrap import (
    Version,
    atomic_symlink,
    load_config,
    parse_env_file,
    parse_project_version,
    parse_tag,
    sanitize,
)
from fakes.clock import FakeClock
from fakes.fake_toolchain import Call, FakeToolchain
from fakes.git_origin import LOCK_A, LOCK_B, Origin, git


@pytest.fixture
def world(tmp_path: Path) -> World:
    clock = FakeClock()
    return World(
        tmp=tmp_path,
        origin=Origin(tmp_path / "origin"),
        clock=clock,
        tools=FakeToolchain(clock),
    )


def sha_of(world: World, tag: str) -> str:
    return git(world.origin.path, "rev-parse", f"{tag}^{{commit}}")


# --- pure helpers ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("v0.0.1", (0, 0, 1)),
        ("v1.2.3", (1, 2, 3)),
        ("v10.20.30", (10, 20, 30)),
        ("v1.0", None),
        ("1.0.0", None),
        ("V1.0.0", None),
        ("v1.0.0-rc1", None),
        ("v1.0.0.1", None),
        ("v1.0.0\n", None),
        ("v1.0.0 ", None),
        ("v\u0661.\u0660.\u0660", None),  # Arabic-Indic digits
        ("v1.0.0/../x", None),
        ("v1234567.0.0", None),
        ("", None),
    ],
)
def test_only_plain_semver_tags_are_release_tags(
    name: str, expected: tuple[int, int, int] | None
) -> None:
    assert parse_tag(name) == expected


def test_semver_ordering_is_numeric_not_lexical() -> None:
    names = ["v0.9.0", "v0.10.0", "v1.0.0", "v0.2.0", "v1.0.10", "v1.0.9"]
    ranked = sorted(names, key=lambda n: parse_tag(n) or (0, 0, 0), reverse=True)
    assert ranked == ["v1.0.10", "v1.0.9", "v1.0.0", "v0.10.0", "v0.9.0", "v0.2.0"]
    assert sorted(names, reverse=True)[0] != ranked[0]  # lexical order would pick wrongly


def test_version_string_is_tag_plus_short_sha() -> None:
    version = Version("v0.2.0", "a" * 40)
    assert version.agent_version == "v0.2.0+" + "a" * 12 and len(version.agent_version) <= 64


def test_sanitize_strips_url_credentials_and_collapses_whitespace() -> None:
    text = "fatal: unable to access 'https://user:s3cr3t@example.invalid/x.git/':\n  timed out"
    cleaned = sanitize(text)
    assert "s3cr3t" not in cleaned and "user:" not in cleaned and "\n" not in cleaned
    assert "example.invalid" in cleaned
    assert len(sanitize("x" * 1000)) == 200


def test_env_file_parsing_does_not_expand_or_execute() -> None:
    parsed = parse_env_file(
        "# comment\nA=1\nexport B = \"two words\"\nC='x'\nD=$(touch /tmp/x)\nbad line\n=nokey\nE=\n"
    )
    assert parsed == {"A": "1", "B": "two words", "C": "x", "D": "$(touch /tmp/x)", "E": ""}


def test_pyproject_version_is_read_from_the_project_table() -> None:
    text = (
        '[tool.a]\nversion = "9"\n[project]\nname = "x"\nversion = "1.2.3"\n'
        '[tool.b]\nversion = "7"\n'
    )
    assert parse_project_version(text) == "1.2.3"
    assert parse_project_version('[tool.b]\nversion = "7"\n') is None


def test_config_environment_beats_the_deploy_env_file(tmp_path: Path) -> None:
    deploy_env = tmp_path / "deploy.env"
    deploy_env.write_text(
        "RENDER_AGENT_HOME=/from/file\nRENDER_AGENT_PIN=v1.0.0\nRENDER_AGENT_PYTHON=py\n"
    )
    config = load_config(
        {"RENDER_AGENT_DEPLOY_ENV": str(deploy_env), "RENDER_AGENT_HOME": "/from/env"}
    )
    assert config.home == Path("/from/env") and config.pin == "v1.0.0" and config.python == "py"
    defaults = load_config({"RENDER_AGENT_DEPLOY_ENV": str(tmp_path / "missing")})
    assert defaults.home == Path("/opt/render-agent") and defaults.total_budget == 300
    assert defaults.version_env == Path("/run/render-agent/version.env")


def test_a_bad_tuning_number_is_ignored_with_a_warning(tmp_path: Path) -> None:
    warnings: list[str] = []
    config = load_config(
        {
            "RENDER_AGENT_DEPLOY_ENV": str(tmp_path / "none"),
            "RENDER_AGENT_BOOTSTRAP_BUDGET_SECONDS": "soon",
            "RENDER_AGENT_PIP_TIMEOUT_SECONDS": "-5",
            "RENDER_AGENT_GIT_TIMEOUT_SECONDS": "30",
        },
        warnings,
    )
    assert config.total_budget == 300 and config.pip_timeout == 900 and config.git_timeout == 30
    assert warnings == [
        "ignoring invalid RENDER_AGENT_BOOTSTRAP_BUDGET_SECONDS",
        "ignoring invalid RENDER_AGENT_PIP_TIMEOUT_SECONDS",
    ]


# --- choosing the release ----------------------------------------------------------------------


def test_first_boot_installs_the_newest_semver_tag(world: World) -> None:
    for version in ("0.9.0", "0.10.0", "1.0.0"):
        world.origin.release(version)
    boot = world.boot()
    assert boot.code == 0 and world.current_tag == "v1.0.0"
    sha = sha_of(world, "v1.0.0")
    assert boot.final == f"bootstrap: running v1.0.0+{sha[:12]} (new; first install)"
    assert world.agent_version_file() == f"AGENT_VERSION=v1.0.0+{sha[:12]}\n"


def test_numeric_order_beats_lexical_order(world: World) -> None:
    world.origin.release("0.9.0")
    world.origin.release("0.10.0")
    world.boot()
    assert world.current_tag == "v0.10.0"


def test_non_semver_and_prerelease_tags_are_ignored(world: World) -> None:
    world.origin.release("0.1.0")
    for name in ("v2.0.0-rc1", "v9.0", "release-9", "V9.9.9", "v9.9.9;x", "9.9.9"):
        world.origin.tag(name)
    boot = world.boot()
    assert world.current_tag == "v0.1.0" and boot.code == 0
    assert not any("rejected" in line for line in boot.lines)  # never even considered


def test_a_pin_overrides_the_newest_tag(world: World) -> None:
    world.origin.release("0.1.0")
    world.origin.release("0.2.0")
    boot = world.boot(pin="v0.1.0")
    assert world.current_tag == "v0.1.0" and "pinned to v0.1.0" in boot.text


@pytest.mark.parametrize(
    "pin", ["../../etc", "main", "v1.0.0\n", "v1.0", "v1.0.0; touch x", "v9.9.9"]
)
def test_an_invalid_or_missing_pin_is_refused_before_any_path_is_used(
    world: World, pin: str
) -> None:
    world.origin.release("0.1.0")
    world.boot()
    before = world.snapshot()
    boot = world.boot(pin=pin)
    assert boot.code == 0 and world.current_tag == "v0.1.0"
    assert "(fallback:" in boot.final
    assert world.snapshot() == before
    assert not (world.tmp / "etc").exists() and not (world.home / "releases" / pin.strip()).exists()


def test_a_tag_on_a_side_branch_is_refused_and_current_keeps_running(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    world.origin.side_branch_release("0.9.0")
    boot = world.boot()
    assert world.current_tag == "v0.1.0" and boot.code == 0
    assert "rejected v0.9.0: commit is not on main" in boot.text
    assert boot.final.endswith("(unchanged)")
    assert "v0.9.0" not in world.releases()


def test_a_tag_whose_pyproject_version_differs_is_refused(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    world.origin.release("0.3.0", tag="v0.4.0")  # pyproject says 0.3.0
    boot = world.boot()
    assert world.current_tag == "v0.1.0"
    assert "rejected v0.4.0: pyproject version does not match the tag" in boot.text


def test_a_moved_tag_is_refused_because_releases_are_immutable(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    new_commit = world.origin.commit("0.1.0")
    world.origin.move_tag("v0.1.0", new_commit)
    boot = world.boot()
    assert world.current_tag == "v0.1.0" and boot.code == 0
    assert "different commit" in boot.text
    assert world.release_of("current") == "v0.1.0"


def test_a_stray_bad_tag_does_not_block_an_older_good_one(world: World) -> None:
    world.origin.release("0.1.0")
    world.origin.side_branch_release("0.5.0")
    boot = world.boot()
    assert world.current_tag == "v0.1.0" and "rejected v0.5.0" in boot.text


def test_deleting_a_bad_tag_rolls_back_to_the_previous_release(world: World) -> None:
    world.origin.release("0.1.0", lock=lock_n(1))
    world.origin.release("0.2.0", lock=lock_n(2))
    world.boot()
    assert world.current_tag == "v0.2.0"
    world.origin.delete_tag("v0.2.0")
    boot = world.boot()
    assert world.current_tag == "v0.1.0" and world.previous_tag == "v0.2.0"
    assert "(new; was v0.2.0)" in boot.final


# --- network and fallback ----------------------------------------------------------------------


def test_unreachable_origin_keeps_running_the_current_version(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    before = world.snapshot()
    boot = world.boot(repo_url="file:///nonexistent/repo.git")
    assert boot.code == 0 and world.current_tag == "v0.1.0"
    assert "fetch failed" in boot.final and "(fallback:" in boot.final
    assert world.agent_version_file().startswith("AGENT_VERSION=v0.1.0+")
    assert world.snapshot() == before


def test_unreachable_origin_and_nothing_installed_exits_non_zero(world: World) -> None:
    boot = world.boot(repo_url="file:///nonexistent/repo.git")
    assert boot.code == 1
    assert boot.final.startswith("bootstrap: no usable version (")
    assert not world.version_env.exists()


def test_nothing_installed_and_no_tags_exits_non_zero(world: World) -> None:
    world.origin.commit("0.0.1")
    boot = world.boot()
    assert boot.code == 1 and "no release tags found" in boot.final


def test_fetch_is_retried_with_backoff_until_the_network_is_up(world: World) -> None:
    world.origin.release("0.1.0")
    world.tools.git_failures = [lambda argv: "fetch" in argv] * 3
    boot = world.boot()
    assert boot.code == 0 and world.current_tag == "v0.1.0"
    assert world.clock.sleeps == [2.0, 4.0, 8.0]
    assert "fetched tags (attempt 4)" in boot.text


def test_fetch_retries_are_bounded(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    world.tools.git_failures = [lambda argv: "fetch" in argv] * 1000
    world.clock.sleeps.clear()
    boot = world.boot(fetch_retry_budget=60.0)
    assert boot.code == 0 and "fetch failed after" in boot.final
    assert sum(world.clock.sleeps) < 60 and boot.elapsed < 60 + 1


def test_a_hanging_fetch_times_out_and_falls_back(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    world.tools.hang = {"git-fetch"}
    boot = world.boot()
    assert boot.code == 0 and "fetch timed out" in boot.final and world.current_tag == "v0.1.0"


def test_an_offline_boot_with_nothing_installed_can_use_a_prefilled_mirror(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    for name in ("current", "current-venv", "previous", "previous-venv"):
        (world.home / name).unlink(missing_ok=True)
    boot = world.boot(repo_url="file:///nonexistent/repo.git")
    assert boot.code == 0 and world.current_tag == "v0.1.0"
    assert "trying the local mirror" in boot.text


def test_a_damaged_mirror_is_recreated(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    (world.home / "repo" / "HEAD").write_text("garbage")
    world.origin.release("0.2.0", lock=lock_n(2))
    boot = world.boot()
    assert "mirror is damaged" in boot.text and world.current_tag == "v0.2.0"


# --- environments (venvs) ----------------------------------------------------------------------


def test_the_environment_is_reused_while_the_lock_is_unchanged(world: World) -> None:
    world.origin.release("0.1.0", lock=LOCK_A)
    world.boot()
    assert world.tools.count("pip-install") == 1
    world.origin.release("0.1.1", lock=LOCK_A)
    boot = world.boot()
    assert world.current_tag == "v0.1.1"
    assert world.tools.count("pip-install") == 1 and world.tools.count("venv") == 1
    assert "reusing environment" in boot.text and len(world.venvs()) == 1


def test_a_changed_lock_builds_a_new_environment(world: World) -> None:
    world.origin.release("0.1.0", lock=LOCK_A)
    world.origin.release("0.2.0", lock=LOCK_B)
    world.boot()
    assert world.tools.count("pip-install") == 1  # only the newest was prepared
    world.origin.release("0.3.0", lock=lock_n(3))
    world.boot()
    assert world.tools.count("pip-install") == 2 and len(world.venvs()) == 2


def test_dependencies_install_with_hashes_binaries_only_and_no_shell(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    install = next(c for c in world.tools.calls if c.kind == "pip-install")
    assert install.argv[1:4] == ("-m", "pip", "install")
    for flag in ("--require-hashes", "--only-binary=:all:", "--no-deps"):
        assert flag in install.argv
    assert install.argv[install.argv.index("-r") + 1].endswith("releases/v0.1.0/requirements.lock")
    assert any(c.kind == "pip-check" for c in world.tools.calls)
    for call in world.tools.calls:
        assert isinstance(call.argv, tuple) and call.argv[0] not in ("sh", "bash", "/bin/sh")
        assert "-c" not in call.argv[:2]
        assert 0 < call.timeout <= 900


def test_every_external_call_has_a_timeout_within_the_budget(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot(total_budget=100.0)
    assert world.tools.calls and all(0 < c.timeout <= 100 for c in world.tools.calls)


def test_git_never_prompts_and_ignores_user_configuration(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    for call in (c for c in world.tools.calls if c.kind == "git"):
        assert call.env["GIT_TERMINAL_PROMPT"] == "0"
        assert (
            call.env["GIT_CONFIG_GLOBAL"] == os.devnull and call.env["GIT_CONFIG_NOSYSTEM"] == "1"
        )


def test_a_failed_installer_leaves_no_environment_and_keeps_current(world: World) -> None:
    world.origin.release("0.1.0", lock=LOCK_A)
    world.boot()
    world.origin.release("0.2.0", lock=LOCK_B)
    world.tools.pip_fails = True
    boot = world.boot()
    assert boot.code == 0 and world.current_tag == "v0.1.0"
    assert "install dependencies failed" in boot.final
    assert len(world.venvs()) == 1 and list((world.home / ".tmp").iterdir()) == []
    assert world.agent_version_file().startswith("AGENT_VERSION=v0.1.0+")


def test_a_failed_venv_creation_leaves_nothing_behind(world: World) -> None:
    world.origin.release("0.1.0")
    world.tools.venv_fails = True
    boot = world.boot()
    assert boot.code == 1 and "create environment failed" in boot.text
    assert world.venvs() == [] and list((world.home / ".tmp").iterdir()) == []


# --- selfcheck gate and switching --------------------------------------------------------------


def test_a_failing_selfcheck_never_replaces_the_working_version(world: World) -> None:
    world.origin.release("0.1.0", lock=LOCK_A)
    world.boot()
    pair = (world.link("current"), world.link("current-venv"))
    world.origin.release("0.1.1", lock=LOCK_A)
    world.tools.selfcheck_fails = {"0.1.1"}
    boot = world.boot()
    assert boot.code == 0 and world.current_tag == "v0.1.0"
    assert (world.link("current"), world.link("current-venv")) == pair
    assert world.link("previous") is None
    assert "selfcheck failed (selfcheck failed: cannot import httpx" in boot.final
    assert world.agent_version_file().startswith("AGENT_VERSION=v0.1.0+")


def test_a_selfcheck_that_reports_another_version_is_refused(world: World) -> None:
    world.origin.release("0.1.0")
    world.tools.selfcheck_prints_version = {"0.1.0": "0.0.9"}
    boot = world.boot()
    assert boot.code == 1 and "different version than the tag" in boot.text


def test_previous_points_at_the_old_pair_after_a_switch(world: World) -> None:
    world.origin.release("0.1.0", lock=LOCK_A)
    world.boot()
    old = (world.link("current"), world.link("current-venv"))
    world.origin.release("0.2.0", lock=LOCK_B)
    boot = world.boot()
    assert world.current_tag == "v0.2.0" and world.previous_tag == "v0.1.0"
    assert (world.link("previous"), world.link("previous-venv")) == old
    assert world.link("current") == "releases/v0.2.0" and world.link("current-venv") is not None
    assert "(new; was v0.1.0)" in boot.final
    assert (world.home / "current" / "RELEASE").read_text().startswith("tag=v0.2.0\ncommit=")


def test_the_symlink_switch_is_atomic_for_a_concurrent_reader(tmp_path: Path) -> None:
    for name in ("one", "two"):
        (tmp_path / "releases" / name).mkdir(parents=True)
        (tmp_path / "releases" / name / "RELEASE").write_text(name)
    link = tmp_path / "current"
    atomic_symlink(link, "releases/one")
    stop = threading.Event()
    failures: list[str] = []
    reads = 0

    def reader() -> None:
        nonlocal reads
        while not stop.is_set():
            try:
                content = (link / "RELEASE").read_text()
                if content not in ("one", "two"):
                    failures.append(content)
                reads += 1
            except OSError as exc:
                # macOS intermittently answers EINVAL while a path component is being replaced
                # by rename(2) (observed: never ENOENT, never wrong content). Linux, which the
                # instance and CI run, must see no error at all.
                if not (sys.platform == "darwin" and exc.errno == errno.EINVAL):
                    failures.append(f"{type(exc).__name__}: {exc.strerror}")

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for index in range(3000):
            atomic_symlink(link, "releases/two" if index % 2 == 0 else "releases/one")
    finally:
        stop.set()
        thread.join()
    assert failures == [] and reads > 100
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_version_file_is_exactly_the_agent_version_setting_format(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    content = world.agent_version_file()
    sha = sha_of(world, "v0.1.0")
    assert content == f"AGENT_VERSION=v0.1.0+{sha[:12]}\n"
    assert len(content.removeprefix("AGENT_VERSION=").strip()) <= 64
    assert world.version_env.stat().st_mode & 0o777 == 0o644


def test_an_unwritable_version_file_is_a_warning_not_a_failure(
    world: World, tmp_path: Path
) -> None:
    world.origin.release("0.1.0")
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed")
    boot = world.boot(version_env=blocker / "version.env")
    assert boot.code == 0 and "cannot write the version file" in boot.text


def test_the_release_directory_holds_source_without_git_metadata(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    release = world.home / "releases" / "v0.1.0"
    assert (release / "requirements.lock").is_file() and (
        release / "src/render_agent/__init__.py"
    ).is_file()
    assert not (release / ".git").exists()
    assert (release / "RELEASE").read_text() == f"tag=v0.1.0\ncommit={sha_of(world, 'v0.1.0')}\n"


# --- archive safety ----------------------------------------------------------------------------


class EvilArchive(FakeToolchain):
    def __init__(self, clock: FakeClock, build: str) -> None:
        super().__init__(clock)
        self.build = build

    def __call__(self, argv: Sequence[str], **kwargs):  # type: ignore[no-untyped-def]
        if "archive" in argv:
            self.calls.append(Call(tuple(argv), kwargs["timeout"], dict(kwargs["env"])))
            output = next(a for a in argv if a.startswith("--output=")).split("=", 1)[1]
            with tarfile.open(output, "w") as tar:
                good = tarfile.TarInfo("requirements.lock")
                good.size = 0
                tar.addfile(good, io.BytesIO(b""))
                if self.build == "traversal":
                    evil = tarfile.TarInfo("../escaped.txt")
                elif self.build == "absolute":
                    evil = tarfile.TarInfo("/tmp/escaped-absolute.txt")
                else:
                    evil = tarfile.TarInfo("link")
                    evil.type = tarfile.SYMTYPE
                    evil.linkname = "/etc/passwd"
                evil.size = 0
                tar.addfile(evil, io.BytesIO(b""))
            from bootstrap import CommandResult

            return CommandResult(0)
        return super().__call__(argv, **kwargs)


@pytest.mark.parametrize("build", ["traversal", "absolute", "symlink"])
def test_unsafe_archive_entries_are_rejected_and_nothing_escapes(world: World, build: str) -> None:
    world.origin.release("0.1.0")
    world.tools = EvilArchive(world.clock, build)
    boot = world.boot()
    assert boot.code == 1
    assert "unsafe path" in boot.text or "link or special file" in boot.text
    assert not (world.tmp / "escaped.txt").exists() and not (world.home / "escaped.txt").exists()
    assert not Path("/tmp/escaped-absolute.txt").exists()
    assert world.releases() == [] and list((world.home / ".tmp").iterdir()) == []


# --- pruning -----------------------------------------------------------------------------------


def test_pruning_keeps_three_releases_plus_previous_and_the_environments_they_use(
    world: World,
) -> None:
    for n in range(1, 6):
        world.origin.release(f"0.{n}.0", lock=lock_n(n))
        world.boot()
    assert world.releases() == ["v0.3.0", "v0.4.0", "v0.5.0"] and len(world.venvs()) == 3
    boot = world.boot(pin="v0.3.0")  # roll back; previous becomes v0.5.0
    assert world.current_tag == "v0.3.0" and world.previous_tag == "v0.5.0"
    assert world.releases() == ["v0.3.0", "v0.4.0", "v0.5.0"]
    assert "pruned" not in boot.text


def test_a_rolled_back_release_outside_the_newest_three_is_kept_while_in_use(world: World) -> None:
    for n in range(1, 6):
        world.origin.release(f"0.{n}.0", lock=lock_n(n))
    world.boot(pin="v0.1.0")
    world.boot()  # newest again: previous is v0.1.0
    assert world.current_tag == "v0.5.0" and world.previous_tag == "v0.1.0"
    assert "v0.1.0" in world.releases() and len(world.releases()) == 2


def test_an_environment_still_used_by_a_kept_release_survives_pruning(world: World) -> None:
    for n in range(1, 6):
        world.origin.release(f"0.{n}.0", lock=LOCK_A if n in (1, 5) else lock_n(n))
        world.boot()
    assert "v0.1.0" not in world.releases()  # pruned ...
    shared_environment = hashlib.sha256(LOCK_A.encode()).hexdigest()[:12]
    assert shared_environment in world.venvs()  # ... but v0.5.0 still uses its environment
    assert len(world.venvs()) == 3 and world.current_tag == "v0.5.0"


def test_stale_temporary_files_from_a_killed_run_are_removed(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    stale = world.home / ".tmp" / "venv-999-1"
    (stale / "bin").mkdir(parents=True)
    (world.home / ".tmp" / "archive-999-2.tar").write_text("x")
    world.boot()
    assert list((world.home / ".tmp").iterdir()) == []


# --- robustness --------------------------------------------------------------------------------


def test_running_twice_with_no_new_tag_changes_nothing(world: World) -> None:
    world.origin.release("0.1.0")
    first = world.boot()
    snapshot = world.snapshot()
    calls_before = len(world.tools.calls)
    second = world.boot()
    assert world.snapshot() == snapshot
    assert second.final.endswith("(unchanged)") and first.final.endswith("first install)")
    non_git = [c for c in world.tools.calls[calls_before:] if c.kind != "git"]
    assert non_git == []


def test_a_damaged_current_is_replaced_by_the_previous_pair(world: World) -> None:
    world.origin.release("0.1.0", lock=LOCK_A)
    world.origin.release("0.2.0", lock=LOCK_B)
    world.boot(pin="v0.1.0")
    world.boot()  # current v0.2.0, previous v0.1.0
    current_venv = (world.link("current-venv") or "").split("/")[-1]
    (world.home / "venvs" / current_venv / "bin" / "python").unlink()
    boot = world.boot(repo_url="file:///nonexistent/repo.git")
    assert boot.code == 0 and world.current_tag == "v0.1.0"
    assert "restored the previous one" in boot.final


def test_a_half_switched_pair_is_repaired_on_the_next_boot(world: World) -> None:
    world.origin.release("0.1.0", lock=LOCK_A)
    world.origin.release("0.2.0", lock=LOCK_B)
    world.boot(pin="v0.1.0")
    world.boot()
    # simulate a crash between the two symlink switches: new release, old environment
    atomic_symlink(world.home / "current-venv", world.link("previous-venv") or "")
    boot = world.boot()
    assert boot.code == 0 and "(unchanged)" not in boot.final
    assert world.current_tag == "v0.2.0"
    assert world.link("current-venv") != world.link("previous-venv")


def test_leftover_release_directory_without_a_valid_release_file_is_replaced(world: World) -> None:
    world.origin.release("0.1.0")
    (world.home / "releases" / "v0.1.0").mkdir(parents=True)
    (world.home / "releases" / "v0.1.0" / "junk").write_text("partial")
    boot = world.boot()
    assert boot.code == 0 and world.current_tag == "v0.1.0"
    assert not (world.home / "releases" / "v0.1.0" / "junk").exists()
    assert "removing a damaged leftover" in boot.text


def test_the_time_budget_aborts_a_hanging_step_and_falls_back(world: World) -> None:
    world.origin.release("0.1.0", lock=LOCK_A)
    world.boot()
    world.origin.release("0.2.0", lock=LOCK_B)
    world.tools.hang = {"pip-install"}
    boot = world.boot(total_budget=300.0)
    assert boot.code == 0 and world.current_tag == "v0.1.0"
    assert "time budget exceeded during install dependencies" in boot.final
    assert boot.elapsed <= 300 + 1
    assert len(world.venvs()) == 1 and list((world.home / ".tmp").iterdir()) == []


def test_a_hanging_selfcheck_falls_back_too(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    world.origin.release("0.1.1")
    world.tools.hang = {"selfcheck"}
    boot = world.boot()
    assert world.current_tag == "v0.1.0" and "selfcheck timed out" in boot.final


def test_a_tiny_budget_never_hangs_the_boot(world: World) -> None:
    world.origin.release("0.1.0")
    world.boot()
    boot = world.boot(total_budget=0.0000001)
    assert boot.code == 0 and world.current_tag == "v0.1.0"


def test_a_second_bootstrap_does_not_wait_forever_or_corrupt_state(
    world: World, tmp_path: Path
) -> None:
    world.origin.release("0.1.0", lock=LOCK_A)
    world.boot()
    world.origin.release("0.2.0", lock=LOCK_B)

    entered, proceed = threading.Event(), threading.Event()

    def block_in_selfcheck(call: Call) -> None:
        if call.kind == "selfcheck":
            entered.set()
            assert proceed.wait(30)

    world.tools.on_call = block_in_selfcheck
    first_result: list[int] = []
    first = threading.Thread(target=lambda: first_result.append(world.boot().code))
    first.start()
    assert entered.wait(30)

    clock = FakeClock()
    second_world = World(tmp_path, world.origin, clock, FakeToolchain(clock))
    snapshot = second_world.snapshot()
    second = second_world.boot(lock_wait=2.0)
    assert second.code == 0 and "another bootstrap is running" in second.text
    assert second.final.startswith("bootstrap: running v0.1.0+") and "(fallback:" in second.final
    assert second_world.snapshot() == snapshot  # it changed nothing
    assert second_world.tools.count("pip-install") == 0 and second_world.tools.count("git") == 0

    proceed.set()
    first.join(30)
    assert first_result == [0] and world.current_tag == "v0.2.0"


def test_output_never_contains_credentials_from_the_repository_url(world: World) -> None:
    secret_url = "https://user:s3cr3t@example.invalid/org/repo.git"
    world.origin.release("0.1.0")
    world.boot()
    world.tools.git_failures = [lambda argv: "fetch" in argv] * 20
    original = world.tools.__call__

    def leaky(argv: Sequence[str], **kwargs):  # type: ignore[no-untyped-def]
        result = original(argv, **kwargs)
        if result.returncode == 128:
            from bootstrap import CommandResult

            return CommandResult(
                128, stderr=f"fatal: unable to access '{secret_url}/': Could not resolve host\n"
            )
        return result

    world.tools = leaky  # type: ignore[assignment]
    boot = world.boot(repo_url=secret_url, fetch_retry_budget=5.0)
    assert boot.code == 0 and "s3cr3t" not in boot.text and "user:" not in boot.text


# --- the command line --------------------------------------------------------------------------


def test_main_runs_a_bootstrap_from_the_environment(world: World) -> None:
    world.origin.release("0.1.0")
    out = io.StringIO()
    code = bootstrap.main(
        [],
        environ={
            "RENDER_AGENT_REPO_URL": world.origin.url,
            "RENDER_AGENT_HOME": str(world.home),
            "RENDER_AGENT_VERSION_ENV": str(world.version_env),
            "RENDER_AGENT_DEPLOY_ENV": str(world.tmp / "absent.env"),
            "RENDER_AGENT_BOOTSTRAP_BUDGET_SECONDS": "oops",
            "PATH": os.environ["PATH"],
        },
        runner=world.tools,
        clock=world.clock,
        out=out,
    )
    assert code == 0 and "ignoring invalid RENDER_AGENT_BOOTSTRAP_BUDGET_SECONDS" in out.getvalue()
    assert world.current_tag == "v0.1.0"


def test_main_rejects_unexpected_arguments() -> None:
    assert bootstrap.main(["--surprise"]) == 2


# --- AC2: a simulated history of boots ---------------------------------------------------------


def test_full_history_of_boots(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    """v0.1.0 installed, v0.1.1 broken, an outage, v0.2.0 with a changed lock, later releases."""

    def step(title: str, **overrides: object) -> tuple[str, str]:
        boot = world.boot(**overrides)
        print(f"\n== {title}")
        for line in boot.lines:
            print(f"   {line}")
        return boot.final, world.current_tag or "-"

    world.origin.release("0.1.0", lock=LOCK_A)
    final, tag = step("boot 1: first install of the newest tag, v0.1.0")
    assert tag == "v0.1.0" and "first install" in final and world.tools.count("pip-install") == 1

    world.origin.release("0.1.1", lock=LOCK_A)
    world.tools.selfcheck_fails = {"0.1.1"}
    installs = world.tools.count("pip-install")
    final, tag = step("boot 2: v0.1.1 is tagged but fails its self-check")
    assert tag == "v0.1.0" and "(fallback: selfcheck failed" in final
    assert world.tools.count("pip-install") == installs  # unchanged lock: no reinstall

    final, tag = step("boot 3: same state again, still on v0.1.0")
    assert tag == "v0.1.0" and "(fallback:" in final

    final, tag = step("boot 4: GitHub is unreachable", repo_url="file:///nonexistent/repo.git")
    assert (
        tag == "v0.1.0"
        and "fetch failed" in final
        and world.agent_version_file().startswith("AGENT_VERSION=v0.1.0+")
    )

    world.origin.release("0.2.0", lock=LOCK_B)
    world.tools.selfcheck_fails = set()
    final, tag = step("boot 5: v0.2.0 arrives with a changed requirements.lock")
    assert tag == "v0.2.0" and "(new; was v0.1.0)" in final
    assert world.tools.count("pip-install") == installs + 1 and world.previous_tag == "v0.1.0"
    assert "v0.1.0" in world.releases()  # never lost until pruned

    final, tag = step("boot 6: nothing new")
    assert tag == "v0.2.0" and final.endswith("(unchanged)")
    assert world.tools.count("pip-install") == installs + 1

    world.origin.release("0.3.0", lock=LOCK_B)
    final, tag = step("boot 7: v0.3.0, same lock as v0.2.0, so no reinstall")
    assert tag == "v0.3.0" and world.tools.count("pip-install") == installs + 1
    assert world.releases() == ["v0.1.0", "v0.1.1", "v0.2.0", "v0.3.0"][
        1:
    ] and "pruned release v0.1.0" in "\n".join(world.boots[-1].lines)

    world.origin.release("0.4.0", lock=LOCK_B)
    final, tag = step("boot 8: v0.4.0; the oldest releases and unused environments are pruned")
    assert tag == "v0.4.0" and world.releases() == ["v0.2.0", "v0.3.0", "v0.4.0"]
    assert len(world.venvs()) == 1  # only the second lock's environment is still used
    assert all(b.code == 0 for b in world.boots)
    capsys.readouterr()
