"""A throwaway installation directory, a local origin repository and a fake toolchain."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path

from bootstrap import Bootstrap, Config
from fakes.clock import FakeClock
from fakes.fake_toolchain import FakeToolchain
from fakes.git_origin import Origin

SHA_LENGTH = 12


def lock_n(n: int) -> str:
    """A distinct requirements.lock per number, so each gets its own environment."""
    return f"example-dependency=={n}.0.0 \\\n    --hash=sha256:{n:064d}\n"


@dataclass
class Boot:
    code: int
    text: str
    elapsed: float

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()

    @property
    def final(self) -> str:
        return self.lines[-1]


@dataclass
class World:
    tmp: Path
    origin: Origin
    clock: FakeClock
    tools: FakeToolchain
    boots: list[Boot] = field(default_factory=list)

    @property
    def home(self) -> Path:
        return self.tmp / "opt" / "render-agent"

    @property
    def version_env(self) -> Path:
        return self.tmp / "run" / "version.env"

    def config(self, **overrides: object) -> Config:
        values: dict[str, object] = {
            "repo_url": self.origin.url,
            "home": self.home,
            "version_env": self.version_env,
            "python": "python3",
        }
        values.update(overrides)
        return Config(**values)  # type: ignore[arg-type]

    def bootstrap(self, **overrides: object) -> Bootstrap:
        out = io.StringIO()
        self._out = out
        return Bootstrap(
            self.config(**overrides),
            runner=self.tools,
            clock=self.clock,
            out=out,
            environ={"PATH": os.environ["PATH"], "HOME": str(self.tmp)},
        )

    def boot(self, **overrides: object) -> Boot:
        started = self.clock.monotonic()
        bootstrap = self.bootstrap(**overrides)
        code = bootstrap.run()
        result = Boot(code, self._out.getvalue(), self.clock.monotonic() - started)
        self.boots.append(result)
        return result

    # --- inspection ----------------------------------------------------------------------------

    def link(self, name: str) -> str | None:
        path = self.home / name
        return os.readlink(path) if path.is_symlink() else None

    def release_of(self, link: str) -> str | None:
        release = self.home / link / "RELEASE"
        if not release.exists():
            return None
        return dict(line.split("=", 1) for line in release.read_text().splitlines())["tag"]

    @property
    def current_tag(self) -> str | None:
        return self.release_of("current")

    @property
    def previous_tag(self) -> str | None:
        return self.release_of("previous")

    def releases(self) -> list[str]:
        return sorted(p.name for p in (self.home / "releases").iterdir())

    def venvs(self) -> list[str]:
        return sorted(p.name for p in (self.home / "venvs").iterdir())

    def agent_version_file(self) -> str:
        return self.version_env.read_text()

    def snapshot(self) -> dict[str, str]:
        """Everything under home except the lock file and the repository mirror."""
        state: dict[str, str] = {}
        for path in sorted(self.home.rglob("*")):
            relative = path.relative_to(self.home)
            if relative.parts[0] in ("repo", ".bootstrap.lock"):
                continue
            if path.is_symlink():
                state[str(relative)] = f"-> {os.readlink(path)}"
            elif path.is_file():
                state[str(relative)] = f"file {path.stat().st_size}"
            else:
                state[str(relative)] = "dir"
        return state
