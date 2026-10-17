"""The renderer interface the server shell depends on, and a dependency-free fake."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

FTYP_BOX = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
FILLER = b"FAKE-NOT-A-VIDEO\n" * 64


class RendererOutOfMemory(Exception):
    """GPU or host memory ran out. Renderers translate their library's error into this."""


@dataclass(frozen=True)
class RenderRequest:
    job_id: int
    prompt: str = field(repr=False)
    width: int
    height: int
    num_frames: int
    fps: float
    seed: int


@dataclass(frozen=True)
class RenderResult:
    width: int
    height: int
    num_frames: int
    fps: float
    has_audio: bool


class Renderer(Protocol):
    def load(self) -> None:
        """Build the model once. Slow; runs in a background thread while the server answers."""
        ...

    def warm_up(self) -> None:
        """One cheap generation so the first real clip does not pay first-use costs."""
        ...

    def render(self, request: RenderRequest, tmp_path: Path) -> RenderResult:
        """Write the finished clip to `tmp_path` (a .mp4 name). Raise RendererOutOfMemory on OOM."""
        ...

    def after_clip(self) -> None:
        """Called after every clip, success or failure: free scratch memory, keep the weights."""
        ...

    def recover(self) -> None:
        """Called after an out-of-memory failure, before the next request."""
        ...


@dataclass
class FakeRenderer:
    """No dependencies. Writes a tiny file with an mp4 `ftyp` box; NOT a playable video.

    Failure kinds: oom, render_error, mid_write (half a file is written, then it fails),
    no_output (returns without writing anything).
    """

    load_seconds: float = 0.0
    warmup_seconds: float = 0.0
    render_seconds: float = 0.0
    fail_every: int = 0
    fail_kind: str = "render_error"
    script: dict[int, str] = field(default_factory=dict)  # call number (1-based) -> failure kind
    load_gate: threading.Event | None = None
    warmup_gate: threading.Event | None = None
    render_gate: threading.Event | None = None
    render_started: threading.Event | None = None
    sleep: Callable[[float], None] = time.sleep
    loads: int = 0
    warm_ups: int = 0
    renders: int = 0
    after_clips: int = 0
    recovers: int = 0
    requests: list[RenderRequest] = field(default_factory=list)

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> FakeRenderer:
        def number(name: str, default: float) -> float:
            return float(environ.get(name, default))

        return cls(
            load_seconds=number("FAKE_LOAD_SECONDS", 0.0),
            warmup_seconds=number("FAKE_WARMUP_SECONDS", 0.0),
            render_seconds=number("FAKE_DELAY_SECONDS", 1.0),
            fail_every=int(number("FAKE_FAIL_EVERY", 0)),
            fail_kind=environ.get("FAKE_FAIL_KIND", "render_error"),
        )

    def load(self) -> None:
        self.loads += 1
        if self.load_gate is not None:
            self.load_gate.wait(30)
        self.sleep(self.load_seconds)

    def warm_up(self) -> None:
        self.warm_ups += 1
        if self.warmup_gate is not None:
            self.warmup_gate.wait(30)
        self.sleep(self.warmup_seconds)

    def render(self, request: RenderRequest, tmp_path: Path) -> RenderResult:
        self.renders += 1
        self.requests.append(request)
        if self.render_started is not None:
            self.render_started.set()
        if self.render_gate is not None:
            self.render_gate.wait(30)
        self.sleep(self.render_seconds)
        kind = self.script.get(self.renders)
        if kind is None and self.fail_every and self.renders % self.fail_every == 0:
            kind = self.fail_kind
        if kind == "oom":
            raise RendererOutOfMemory("fake out of memory")
        if kind == "render_error":
            raise RuntimeError("fake render failure")
        if kind == "no_output":
            return self._result(request)
        if kind == "mid_write":
            tmp_path.write_bytes(FTYP_BOX + FILLER[:100])
            raise RuntimeError("fake crash while writing")
        tmp_path.write_bytes(FTYP_BOX + FILLER)
        return self._result(request)

    @staticmethod
    def _result(request: RenderRequest) -> RenderResult:
        return RenderResult(request.width, request.height, request.num_frames, request.fps, True)

    def after_clip(self) -> None:
        self.after_clips += 1

    def recover(self) -> None:
        self.recovers += 1
