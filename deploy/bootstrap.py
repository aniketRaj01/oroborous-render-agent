#!/usr/bin/env python3
"""Boot-time release bootstrap for the render agent. Standard library only.

Runs as the service user before the agent starts (systemd ExecStartPre). It makes sure the
newest deliberately released version (a git tag vX.Y.Z whose commit is on main) is installed
reproducibly and self-checked, switches to it atomically, and records which version will run.
A bad tag, a GitHub outage or a missing network never stops the agent from starting on the
version that already works: every failure while preparing a new version falls back to the
current one. It exits non-zero only when there is no usable version at all.

Layout under RENDER_AGENT_HOME (default /opt/render-agent):

  repo/                bare mirror of the public repository (fetched, never executed from)
  releases/<tag>/      exported source of a tag, plus a RELEASE file (tag, commit)
  venvs/<hash>/        one virtualenv per sha256-12 of requirements.lock
  current -> releases/<tag>       current-venv -> venvs/<hash>        (atomic switches)
  previous, previous-venv         the last pair that was current

Configuration comes from the environment or the file named by RENDER_AGENT_DEPLOY_ENV
(default /etc/render-agent/deploy.env); the environment wins. See deploy/deploy.env.example.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, TextIO

DEFAULT_REPO_URL = "https://github.com/aniketRaj01/oroborous-render-agent.git"
DEFAULT_HOME = "/opt/render-agent"
DEFAULT_DEPLOY_ENV = "/etc/render-agent/deploy.env"
DEFAULT_VERSION_ENV = "/run/render-agent/version.env"
MAIN_BRANCH = "main"
KEEP_RELEASES = 3
MAX_AGENT_VERSION_CHARS = 64
SHORT_SHA_CHARS = 12
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024

# Matched with fullmatch only: `$` would accept a trailing newline, and \d accepts non-ASCII digits.
TAG_PATTERN = re.compile(r"v([0-9]{1,6})\.([0-9]{1,6})\.([0-9]{1,6})")
VENV_NAME = re.compile(r"[0-9a-f]{12}")
_PROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
_URL_USERINFO = re.compile(r"(://)[^/@\s]+@")

PASSTHROUGH_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
)


# --- small helpers -----------------------------------------------------------------------------


def sanitize(text: str, limit: int = 200) -> str:
    """One line, URL credentials removed, truncated: safe to print."""
    return _URL_USERINFO.sub(r"\1<redacted>@", " ".join(text.split()))[:limit]


def parse_tag(name: str) -> tuple[int, int, int] | None:
    match = TAG_PATTERN.fullmatch(name)
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def parse_project_version(pyproject_text: str) -> str | None:
    project = pyproject_text.partition("[project]")[2].partition("\n[")[0]
    match = _PROJECT_VERSION.search(project)
    return match[1] if match else None


def parse_env_file(text: str) -> dict[str, str]:
    """KEY=VALUE lines. No expansion and nothing is executed."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def atomic_symlink(link: Path, target: str) -> None:
    """Point `link` at `target` so that readers see the old target or the new one, never none."""
    temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    os.symlink(target, temporary)
    os.replace(temporary, link)


def atomic_write(path: Path, text: str, mode: int = 0o644) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.chmod(mode)
    os.replace(temporary, path)


# --- configuration -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    repo_url: str = DEFAULT_REPO_URL
    home: Path = Path(DEFAULT_HOME)
    python: str = "python3"
    pin: str | None = None
    version_env: Path = Path(DEFAULT_VERSION_ENV)
    total_budget: float = 300.0
    fetch_retry_budget: float = 120.0
    fetch_backoff_initial: float = 2.0
    fetch_backoff_max: float = 30.0
    lock_wait: float = 60.0
    git_timeout: float = 120.0
    venv_timeout: float = 120.0
    pip_timeout: float = 900.0
    selfcheck_timeout: float = 60.0


def load_config(environ: Mapping[str, str], warnings: list[str] | None = None) -> Config:
    """Environment first, then the deploy env file. A bad number is ignored with a warning:
    a typo in a tuning knob must not stop the boot."""
    merged = dict(environ)
    deploy_env = Path(environ.get("RENDER_AGENT_DEPLOY_ENV", DEFAULT_DEPLOY_ENV))
    try:
        file_values = parse_env_file(deploy_env.read_text(encoding="utf-8"))
    except OSError:
        file_values = {}
    for key, value in file_values.items():
        merged.setdefault(key, value)

    defaults = Config()

    def number(name: str, default: float) -> float:
        raw = merged.get(name)
        if raw is None or raw == "":
            return default
        try:
            parsed = float(raw)
        except ValueError:
            parsed = -1.0
        if parsed <= 0:
            if warnings is not None:
                warnings.append(f"ignoring invalid {name}")
            return default
        return parsed

    return Config(
        repo_url=merged.get("RENDER_AGENT_REPO_URL") or defaults.repo_url,
        home=Path(merged.get("RENDER_AGENT_HOME") or defaults.home),
        python=merged.get("RENDER_AGENT_PYTHON") or defaults.python,
        pin=merged.get("RENDER_AGENT_PIN") or None,
        version_env=Path(merged.get("RENDER_AGENT_VERSION_ENV") or defaults.version_env),
        total_budget=number("RENDER_AGENT_BOOTSTRAP_BUDGET_SECONDS", defaults.total_budget),
        fetch_retry_budget=number("RENDER_AGENT_FETCH_RETRY_SECONDS", defaults.fetch_retry_budget),
        lock_wait=number("RENDER_AGENT_LOCK_WAIT_SECONDS", defaults.lock_wait),
        git_timeout=number("RENDER_AGENT_GIT_TIMEOUT_SECONDS", defaults.git_timeout),
        pip_timeout=number("RENDER_AGENT_PIP_TIMEOUT_SECONDS", defaults.pip_timeout),
    )


# --- collaborators -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class CommandRunner(Protocol):
    def __call__(
        self, argv: Sequence[str], *, cwd: Path | None, env: Mapping[str, str], timeout: float
    ) -> CommandResult: ...


def subprocess_runner(
    argv: Sequence[str], *, cwd: Path | None, env: Mapping[str, str], timeout: float
) -> CommandResult:
    """Run one command: argument list (never a shell), explicit environment, hard timeout."""
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(returncode=124, timed_out=True)
    except OSError as exc:
        return CommandResult(returncode=127, stderr=f"cannot run {argv[0]}: {type(exc).__name__}")
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class StepFailed(Exception):
    """A step failed; the reason is safe to print. The boot falls back to the current version."""


class BudgetExceeded(StepFailed):
    pass


class LockBusy(Exception):
    pass


@dataclass(frozen=True)
class Version:
    tag: str
    commit: str

    @property
    def agent_version(self) -> str:
        return f"{self.tag}+{self.commit[:SHORT_SHA_CHARS]}"


class _Lock:
    def __init__(self, handle: int) -> None:
        self._handle = handle

    def close(self) -> None:
        os.close(self._handle)  # closing the descriptor releases the flock


# --- the bootstrap -----------------------------------------------------------------------------


class Bootstrap:
    def __init__(
        self,
        config: Config,
        *,
        runner: CommandRunner = subprocess_runner,
        clock: Clock | None = None,
        out: TextIO | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config
        self._runner = runner
        self._clock = clock or SystemClock()
        self._out = out if out is not None else sys.stdout
        self._environ = os.environ if environ is None else environ
        self._deadline = self._clock.monotonic() + config.total_budget
        home = config.home
        self._repo = home / "repo"
        self._releases = home / "releases"
        self._venvs = home / "venvs"
        self._tmp = home / ".tmp"
        self._current = home / "current"
        self._current_venv = home / "current-venv"
        self._previous = home / "previous"
        self._previous_venv = home / "previous-venv"
        self._tmp_counter = 0

    # --- output and process plumbing -----------------------------------------------------------

    def _log(self, message: str) -> None:
        self._out.write(f"bootstrap: {sanitize(message, 400)}\n")
        self._out.flush()

    def _base_env(self) -> dict[str, str]:
        env = {k: self._environ[k] for k in PASSTHROUGH_ENV if k in self._environ}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        return env

    def _git_env(self) -> dict[str, str]:
        return {
            **self._base_env(),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }

    def _remaining(self) -> float:
        return self._deadline - self._clock.monotonic()

    def _timeout(self, step_timeout: float) -> float:
        remaining = self._remaining()
        if remaining <= 0:
            raise BudgetExceeded("time budget exceeded")
        return min(step_timeout, remaining)

    def _run(
        self,
        argv: Sequence[str],
        *,
        what: str,
        timeout: float,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> CommandResult:
        result = self._runner(
            argv,
            cwd=cwd,
            env=env if env is not None else self._base_env(),
            timeout=self._timeout(timeout),
        )
        if result.timed_out:
            if self._remaining() <= 0:
                raise BudgetExceeded(f"time budget exceeded during {what}")
            raise StepFailed(f"{what} timed out")
        return result

    def _git(self, *args: str, what: str, check: bool = True) -> CommandResult:
        result = self._run(
            ["git", "-C", str(self._repo), *args],
            what=what,
            timeout=self._config.git_timeout,
            env=self._git_env(),
        )
        if check and result.returncode != 0:
            raise StepFailed(f"{what} failed ({self._last_line(result)})")
        return result

    @staticmethod
    def _last_line(result: CommandResult) -> str:
        """The most telling line of a failed command's output (git's `fatal:` line if any)."""
        lines = [line for line in (result.stderr or result.stdout).splitlines() if line.strip()]
        if not lines:
            return f"exit status {result.returncode}"
        telling = [line for line in lines if line.startswith(("fatal:", "error:", "ERROR:"))]
        return sanitize((telling or lines)[-1] if not telling else telling[0])

    def _temp_path(self, prefix: str) -> Path:
        self._tmp_counter += 1
        return self._tmp / f"{prefix}-{os.getpid()}-{self._tmp_counter}"

    # --- entry point ---------------------------------------------------------------------------

    def run(self) -> int:
        try:
            self._config.home.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._log(f"no usable version (cannot create the home directory: {type(exc).__name__})")
            return 1
        try:
            lock = self._acquire_lock()
        except LockBusy:
            self._log("another bootstrap is running; using the current version without changes")
            return self._finish("fallback", "another bootstrap holds the lock", locked=False)
        try:
            return self._run_locked()
        except Exception as exc:  # a bug here must never stop the boot
            self._log(f"unexpected {type(exc).__name__}; using the current version")
            return self._finish("fallback", f"unexpected {type(exc).__name__}", locked=False)
        finally:
            lock.close()

    def _acquire_lock(self) -> _Lock:
        try:
            handle = os.open(self._config.home / ".bootstrap.lock", os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            raise LockBusy from None
        give_up = self._clock.monotonic() + self._config.lock_wait
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return _Lock(handle)
            except BlockingIOError:
                if self._clock.monotonic() < give_up and self._remaining() > 0:
                    self._clock.sleep(0.5)
                    continue
            except OSError:
                pass
            os.close(handle)
            raise LockBusy from None

    def _run_locked(self) -> int:
        status, reason = "unchanged", ""
        try:
            for directory in (self._releases, self._venvs, self._tmp):
                directory.mkdir(parents=True, exist_ok=True)
            self._clear_tmp()
            status, reason = self._upgrade()
        except StepFailed as exc:
            status, reason = "fallback", str(exc)
        except OSError as exc:
            status, reason = "fallback", f"filesystem error while preparing ({type(exc).__name__})"
        return self._finish(status, reason, locked=True)

    def _clear_tmp(self) -> None:
        for entry in self._tmp.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)

    # --- choosing and preparing a release ------------------------------------------------------

    def _upgrade(self) -> tuple[str, str]:
        """Prepare and switch to the chosen release: (status, detail), or raise StepFailed."""
        try:
            self._fetch()
        except StepFailed as exc:
            if self._read_pair() is not None:
                raise
            self._log(f"{exc}; nothing is installed yet, so trying the local mirror")
        version = self._choose()
        if self._is_current(version):
            self._log(f"{version.tag} is already current")
            return "unchanged", ""
        release_dir = self._export(version)
        venv_dir = self._ensure_venv(release_dir)
        self._selfcheck(version, release_dir, venv_dir)
        old = self._read_pair()
        self._switch(release_dir, venv_dir)
        self._log(f"switched to {version.agent_version}")
        return "new", f"was {old[0].tag}" if old else "first install"

    def _fetch(self) -> None:
        self._ensure_repo()
        started = self._clock.monotonic()
        backoff = self._config.fetch_backoff_initial
        attempt = 0
        self._git("config", "remote.origin.url", self._config.repo_url, what="configure origin")
        while True:
            attempt += 1
            try:
                result = self._git(
                    "fetch",
                    "--prune",
                    "--no-tags",
                    "origin",
                    f"+refs/heads/{MAIN_BRANCH}:refs/remotes/origin/{MAIN_BRANCH}",
                    "+refs/tags/*:refs/tags/*",
                    what="fetch",
                    check=False,
                )
                failure = None if result.returncode == 0 else self._last_line(result)
            except BudgetExceeded:
                raise
            except StepFailed as exc:  # a timed-out fetch is a failed attempt like any other
                failure = str(exc)
            if failure is None:
                self._log(f"fetched tags (attempt {attempt})")
                return
            waited = self._clock.monotonic() - started
            if waited + backoff >= self._config.fetch_retry_budget or self._remaining() <= backoff:
                raise StepFailed(f"fetch failed after {attempt} attempt(s) ({failure})")
            self._log(f"fetch attempt {attempt} failed; retrying in {backoff:g}s")
            self._clock.sleep(backoff)
            backoff = min(backoff * 2, self._config.fetch_backoff_max)

    def _ensure_repo(self) -> None:
        if self._repo.exists():
            check = self._run(
                ["git", "-C", str(self._repo), "rev-parse", "--is-bare-repository"],
                what="check mirror",
                timeout=self._config.git_timeout,
                env=self._git_env(),
            )
            if check.returncode == 0 and check.stdout.strip() == "true":
                return
            self._log("the repository mirror is damaged; recreating it")
            shutil.rmtree(self._repo, ignore_errors=True)
        init = self._run(
            ["git", "init", "--bare", str(self._repo)],
            what="create mirror",
            timeout=self._config.git_timeout,
            env=self._git_env(),
        )
        if init.returncode != 0:
            raise StepFailed(f"create mirror failed ({self._last_line(init)})")

    def _choose(self) -> Version:
        if self._config.pin is not None:
            pinned = self._config.pin
            if parse_tag(pinned) is None:
                raise StepFailed("RENDER_AGENT_PIN is not a valid release tag name")
            self._log(f"pinned to {pinned}")
            return self._evaluate(pinned)
        tags = self._git(
            "for-each-ref", "--format=%(refname:strip=2)", "refs/tags/", what="list tags"
        ).stdout.split()
        ranked = sorted(
            ((parsed, name) for name in tags if (parsed := parse_tag(name)) is not None),
            reverse=True,
        )
        if not ranked:
            raise StepFailed("no release tags found")
        for _, name in ranked:
            try:
                version = self._evaluate(name)
            except StepFailed as exc:
                self._log(f"rejected {name}: {exc}")
                continue
            self._log(f"newest acceptable release is {name}")
            return version
        raise StepFailed("no acceptable release tag")

    def _evaluate(self, tag: str) -> Version:
        """Checks that need only the mirror. `tag` must already match TAG_PATTERN."""
        if parse_tag(tag) is None:
            raise StepFailed("not a release tag name")
        resolved = self._git(
            "rev-parse",
            "--verify",
            "--quiet",
            f"refs/tags/{tag}^{{commit}}",
            what="resolve tag",
            check=False,
        )
        commit = resolved.stdout.strip()
        if resolved.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise StepFailed("tag does not exist")
        ancestor = self._git(
            "merge-base",
            "--is-ancestor",
            commit,
            f"refs/remotes/origin/{MAIN_BRANCH}",
            what="check ancestry",
            check=False,
        )
        if ancestor.returncode == 1:
            raise StepFailed(f"commit is not on {MAIN_BRANCH}")
        if ancestor.returncode != 0:
            raise StepFailed(f"cannot verify the commit is on {MAIN_BRANCH}")
        shown = self._git("show", f"{commit}:pyproject.toml", what="read pyproject", check=False)
        declared = parse_project_version(shown.stdout) if shown.returncode == 0 else None
        if declared != tag[1:]:
            raise StepFailed("pyproject version does not match the tag")
        version = Version(tag, commit)
        if len(version.agent_version) > MAX_AGENT_VERSION_CHARS:
            raise StepFailed("version string is longer than the agent's 64 character limit")
        existing = self._read_release(self._releases / tag)
        if existing is not None and existing.commit != commit:
            raise StepFailed("tag now points at a different commit than the installed release")
        return version

    def _export(self, version: Version) -> Path:
        final = self._releases / version.tag
        if self._read_release(final) == version:
            self._log(f"{version.tag} is already exported")
            return final
        if final.exists() or final.is_symlink():
            if self._is_referenced(final):
                raise StepFailed("the installed release directory is damaged")
            self._log(f"removing a damaged leftover release directory for {version.tag}")
            shutil.rmtree(final, ignore_errors=True)
        archive = self._temp_path("archive").with_suffix(".tar")
        staging = self._temp_path("release")
        try:
            self._git(
                "archive", "--format=tar", f"--output={archive}", version.commit, what="export"
            )
            staging.mkdir()
            self._extract(archive, staging)
            if not (staging / "requirements.lock").is_file():
                raise StepFailed("the release has no requirements.lock")
            if not (staging / "src" / "render_agent" / "__init__.py").is_file():
                raise StepFailed("the release has no render_agent package")
            (staging / "RELEASE").write_text(
                f"tag={version.tag}\ncommit={version.commit}\n", encoding="utf-8"
            )
            os.rename(staging, final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            archive.unlink(missing_ok=True)
        self._log(f"exported {version.tag}")
        return final

    @staticmethod
    def _extract(archive: Path, destination: Path) -> None:
        root = destination.resolve()
        try:
            with tarfile.open(archive, "r:") as tar:
                members = tar.getmembers()
                if sum(m.size for m in members if m.isfile()) > MAX_ARCHIVE_BYTES:
                    raise StepFailed("the release archive is too large")
                for member in members:
                    parts = PurePosixPath(member.name).parts
                    if (
                        member.name.startswith("/")
                        or ".." in parts
                        or "\x00" in member.name
                        or "\\" in member.name
                    ):
                        raise StepFailed("the release archive contains an unsafe path")
                    if not (member.isfile() or member.isdir()):
                        raise StepFailed("the release archive contains a link or special file")
                    target = root.joinpath(*parts)
                    if root != target and root not in target.resolve().parents:
                        raise StepFailed("the release archive escapes its directory")
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        raise StepFailed("the release archive is unreadable")
                    with source, target.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
        except (tarfile.TarError, OSError) as exc:
            raise StepFailed(f"cannot unpack the release archive ({type(exc).__name__})") from None

    def _ensure_venv(self, release_dir: Path) -> Path:
        lock_hash = self._lock_hash(release_dir)
        final = self._venvs / lock_hash
        if final.is_dir():
            self._log(f"reusing environment {lock_hash} (requirements.lock unchanged)")
            return final
        staging = self._temp_path("venv")
        try:
            created = self._run(
                [self._config.python, "-m", "venv", str(staging)],
                what="create environment",
                timeout=self._config.venv_timeout,
            )
            if created.returncode != 0:
                raise StepFailed(f"create environment failed ({self._last_line(created)})")
            venv_python = str(staging / "bin" / "python")
            installed = self._run(
                [
                    venv_python,
                    "-m",
                    "pip",
                    "install",
                    "--require-hashes",
                    "--only-binary=:all:",
                    "--no-deps",
                    "--disable-pip-version-check",
                    "--no-input",
                    "-r",
                    str(release_dir / "requirements.lock"),
                ],
                what="install dependencies",
                timeout=self._config.pip_timeout,
            )
            if installed.returncode != 0:
                raise StepFailed(f"install dependencies failed ({self._last_line(installed)})")
            checked = self._run(
                [venv_python, "-m", "pip", "check"],
                what="check dependencies",
                timeout=self._config.venv_timeout,
            )
            if checked.returncode != 0:
                raise StepFailed(f"dependency check failed ({self._last_line(checked)})")
            os.rename(staging, final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        self._log(f"built environment {lock_hash}")
        return final

    def _is_referenced(self, release_dir: Path) -> bool:
        for link in (self._current, self._previous):
            if link.is_symlink() and Path(os.readlink(link)).name == release_dir.name:
                return True
        return False

    @staticmethod
    def _lock_hash(release_dir: Path) -> str:
        digest = hashlib.sha256((release_dir / "requirements.lock").read_bytes()).hexdigest()
        return digest[:12]

    def _selfcheck(self, version: Version, release_dir: Path, venv_dir: Path) -> None:
        env = {
            **self._base_env(),
            "PYTHONPATH": str(release_dir / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = self._run(
            [str(venv_dir / "bin" / "python"), "-m", "render_agent", "selfcheck"],
            what="selfcheck",
            timeout=self._config.selfcheck_timeout,
            env=env,
        )
        expected = f"render-agent {version.tag[1:]} selfcheck ok"
        if result.returncode != 0:
            detail = sanitize(result.stdout) or self._last_line(result)
            raise StepFailed(f"selfcheck failed ({detail})")
        if result.stdout.strip() != expected:
            raise StepFailed("selfcheck reported a different version than the tag")
        self._log(f"selfcheck passed for {version.tag}")

    # --- the pair: reading, validating, switching ----------------------------------------------

    @staticmethod
    def _read_release(release_dir: Path) -> Version | None:
        try:
            lines = (release_dir / "RELEASE").read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        values = dict(line.split("=", 1) for line in lines if "=" in line)
        tag, commit = values.get("tag", ""), values.get("commit", "")
        if parse_tag(tag) is None or not re.fullmatch(r"[0-9a-f]{40}", commit):
            return None
        return Version(tag, commit)

    def _pair_version(self, release_link: Path, venv_link: Path) -> Version | None:
        """The version of a pair if it is complete and consistent, else None."""
        if not (release_link.is_symlink() and venv_link.is_symlink()):
            return None
        release_dir, venv_dir = release_link.resolve(), venv_link.resolve()
        version = self._read_release(release_dir)
        if version is None or release_dir.name != version.tag:
            return None
        if not (venv_dir / "bin" / "python").exists():
            return None
        try:
            if venv_dir.name != self._lock_hash(release_dir):
                return None
        except OSError:
            return None
        return version

    def _read_pair(self) -> tuple[Version, str] | None:
        version = self._pair_version(self._current, self._current_venv)
        if version is None:
            return None
        return version, os.readlink(self._current_venv)

    def _is_current(self, version: Version) -> bool:
        pair = self._read_pair()
        return pair is not None and pair[0] == version

    def _switch(self, release_dir: Path, venv_dir: Path) -> None:
        old_release = os.readlink(self._current) if self._current.is_symlink() else None
        old_venv = os.readlink(self._current_venv) if self._current_venv.is_symlink() else None
        new_release, new_venv = f"releases/{release_dir.name}", f"venvs/{venv_dir.name}"
        if (
            self._pair_version(self._current, self._current_venv) is not None
            and old_release is not None
            and old_venv is not None
            and (old_release, old_venv) != (new_release, new_venv)
        ):
            atomic_symlink(self._previous_venv, old_venv)
            atomic_symlink(self._previous, old_release)
        atomic_symlink(self._current_venv, new_venv)
        atomic_symlink(self._current, new_release)

    # --- finishing -----------------------------------------------------------------------------

    def _usable_current(self) -> tuple[Version | None, str]:
        """The pair that will run. A damaged current is replaced by previous when possible."""
        version = self._pair_version(self._current, self._current_venv)
        if version is not None:
            return version, ""
        previous = self._pair_version(self._previous, self._previous_venv)
        if previous is None:
            return None, ""
        atomic_symlink(self._current_venv, os.readlink(self._previous_venv))
        atomic_symlink(self._current, os.readlink(self._previous))
        return previous, "the current version was unusable; restored the previous one"

    def _finish(self, status: str, reason: str, *, locked: bool) -> int:
        try:
            version, restored = (
                self._usable_current()
                if locked
                else (self._pair_version(self._current, self._current_venv), "")
            )
        except OSError as exc:
            self._log(f"no usable version (cannot inspect the installation: {type(exc).__name__})")
            return 1
        if version is None:
            self._log(f"no usable version ({reason or 'nothing is installed'})")
            return 1
        if restored:
            status, reason = "fallback", f"{reason}; {restored}" if reason else restored
        self._write_version_env(version)
        if locked:
            self._prune()
        suffix = {"new": f"new; {reason}", "unchanged": "unchanged"}.get(
            status, f"fallback: {reason}"
        )
        self._log(f"running {version.agent_version} ({suffix})")
        return 0

    def _write_version_env(self, version: Version) -> None:
        try:
            self._config.version_env.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(self._config.version_env, f"AGENT_VERSION={version.agent_version}\n")
        except OSError as exc:
            self._log(f"warning: cannot write the version file ({type(exc).__name__})")

    def _prune(self) -> None:
        try:
            self._prune_unchecked()
        except OSError as exc:
            self._log(f"warning: pruning failed ({type(exc).__name__})")

    def _prune_unchecked(self) -> None:
        names = sorted(
            (
                (parsed, entry.name)
                for entry in self._releases.iterdir()
                if entry.is_dir() and (parsed := parse_tag(entry.name)) is not None
            ),
            reverse=True,
        )
        keep_releases = {name for _, name in names[:KEEP_RELEASES]}
        keep_venvs: set[str] = set()
        for link in (self._current, self._previous):
            if link.is_symlink():
                keep_releases.add(Path(os.readlink(link)).name)
        for link in (self._current_venv, self._previous_venv):
            if link.is_symlink():
                keep_venvs.add(Path(os.readlink(link)).name)
        for name in keep_releases:
            try:
                keep_venvs.add(self._lock_hash(self._releases / name))
            except OSError:
                continue
        for _, name in names:
            if name not in keep_releases:
                shutil.rmtree(self._releases / name, ignore_errors=True)
                self._log(f"pruned release {name}")
        for entry in self._venvs.iterdir():
            if entry.is_dir() and VENV_NAME.fullmatch(entry.name) and entry.name not in keep_venvs:
                shutil.rmtree(entry, ignore_errors=True)
                self._log(f"pruned environment {entry.name}")


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    runner: CommandRunner = subprocess_runner,
    clock: Clock | None = None,
    out: TextIO | None = None,
) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        sys.stderr.write("usage: bootstrap.py   (configure with environment variables)\n")
        return 2
    environment = os.environ if environ is None else environ
    stream = out if out is not None else sys.stdout
    warnings: list[str] = []
    config = load_config(environment, warnings)
    for warning in warnings:
        stream.write(f"bootstrap: warning: {warning}\n")
    return Bootstrap(config, runner=runner, clock=clock, out=stream, environ=environment).run()


if __name__ == "__main__":
    sys.exit(main())
