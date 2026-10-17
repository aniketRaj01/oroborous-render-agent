"""Command runner for bootstrap tests: real git, pretend venv/pip/selfcheck."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bootstrap import CommandResult, subprocess_runner
from fakes.clock import FakeClock


@dataclass(frozen=True)
class Call:
    argv: tuple[str, ...]
    timeout: float
    env: Mapping[str, str]

    @property
    def kind(self) -> str:
        if self.argv[0] == "git":
            return "git"
        tail = self.argv[1:]
        if tail[:2] == ("-m", "venv"):
            return "venv"
        if tail[:3] == ("-m", "pip", "install"):
            return "pip-install"
        if tail[:3] == ("-m", "pip", "check"):
            return "pip-check"
        if tail == ("-m", "render_agent", "selfcheck"):
            return "selfcheck"
        return "other"


@dataclass
class FakeToolchain:
    clock: FakeClock
    calls: list[Call] = field(default_factory=list)
    selfcheck_fails: set[str] = field(default_factory=set)  # versions, e.g. "0.1.1"
    selfcheck_prints_version: dict[str, str] = field(default_factory=dict)
    pip_fails: bool = False
    venv_fails: bool = False
    hang: set[str] = field(default_factory=set)  # call kinds that never return
    git_failures: list[Callable[[Sequence[str]], bool]] = field(default_factory=list)
    on_call: Callable[[Call], None] | None = None

    def count(self, kind: str) -> int:
        return sum(1 for call in self.calls if call.kind == kind)

    def __call__(
        self, argv: Sequence[str], *, cwd: Path | None, env: Mapping[str, str], timeout: float
    ) -> CommandResult:
        call = Call(tuple(argv), timeout, dict(env))
        self.calls.append(call)
        if self.on_call is not None:
            self.on_call(call)
        if call.kind in self.hang or (
            call.kind == "git" and "fetch" in argv and "git-fetch" in self.hang
        ):
            self.clock.advance(timeout)
            return CommandResult(124, timed_out=True)
        if call.kind == "git":
            for should_fail in list(self.git_failures):
                if should_fail(argv):
                    self.git_failures.remove(should_fail)
                    return CommandResult(128, stderr="fatal: unable to access the remote")
            return subprocess_runner(argv, cwd=cwd, env=env, timeout=timeout)
        if call.kind == "venv":
            if self.venv_fails:
                return CommandResult(1, stderr="Error: ensurepip is not available")
            target = Path(argv[3])
            (target / "bin").mkdir(parents=True)
            (target / "bin" / "python").write_text("#!/bin/sh\n")
            (target / "pyvenv.cfg").write_text("home = /usr/bin\n")
            return CommandResult(0)
        if call.kind == "pip-install":
            if self.pip_fails:
                return CommandResult(1, stderr="ERROR: THESE PACKAGES DO NOT MATCH THE HASHES")
            venv = Path(argv[0]).parent.parent
            lock = Path(argv[argv.index("-r") + 1])
            (venv / "installed-from").write_text(hashlib.sha256(lock.read_bytes()).hexdigest())
            return CommandResult(0)
        if call.kind == "pip-check":
            return CommandResult(0, stdout="No broken requirements found.\n")
        if call.kind == "selfcheck":
            release = Path(env["PYTHONPATH"]).parent
            tag = dict(
                line.split("=", 1) for line in (release / "RELEASE").read_text().splitlines()
            )["tag"]
            version = tag[1:]
            if version in self.selfcheck_fails:
                return CommandResult(
                    4, stdout="selfcheck failed: cannot import httpx (ImportError)\n"
                )
            printed = self.selfcheck_prints_version.get(version, version)
            return CommandResult(0, stdout=f"render-agent {printed} selfcheck ok\n")
        raise AssertionError(f"unexpected command: {argv}")
