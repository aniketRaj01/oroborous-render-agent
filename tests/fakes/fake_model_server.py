"""In-process fake of the model server, with scripted outcomes per job."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fakes.clock import FakeClock
from render_agent.model_server.client import (
    ModelServerProtocolError,
    ModelServerRejected,
    ModelServerTimeout,
    ModelServerUnavailable,
)
from render_agent.model_server.models import GenerateRequest, GenerateResponse, RenderReport

FTYP_FILE = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"FAKE" * 16


@dataclass(frozen=True)
class Ok:
    pass


@dataclass(frozen=True)
class Reject:
    code: str
    message: str = "scripted failure"


@dataclass(frozen=True)
class Timeout:
    pass


@dataclass(frozen=True)
class Unavailable:
    pass


@dataclass(frozen=True)
class Garbage:
    pass


@dataclass(frozen=True)
class BadFile:
    kind: str  # "missing" | "empty" | "not_mp4"


@dataclass(frozen=True)
class WrongPath:
    pass


Outcome = Ok | Reject | Timeout | Unavailable | Garbage | BadFile | WrongPath


class FakeModelServer:
    def __init__(
        self, clock: FakeClock, *, not_ready_polls: int = 0, render_seconds: float = 240.0
    ) -> None:
        self.clock = clock
        self.not_ready_polls = not_ready_polls
        self.render_seconds = render_seconds
        self.requests: list[GenerateRequest] = []
        self.polls = 0
        self.on_generate: Callable[[GenerateRequest], None] | None = None
        self._script: dict[int, deque[Outcome]] = defaultdict(deque)
        self._default: Outcome = Ok()

    def script(self, job_id: int, *outcomes: Outcome) -> None:
        self._script[job_id].extend(outcomes)

    def default(self, outcome: Outcome) -> None:
        self._default = outcome

    def is_ready(self) -> bool:
        self.polls += 1
        return self.polls > self.not_ready_polls

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.requests.append(request)
        if self.on_generate is not None:
            self.on_generate(request)
        queue = self._script[request.job_id]
        outcome = queue.popleft() if queue else self._default
        self.clock.advance(self.render_seconds if isinstance(outcome, Ok) else 20.0)
        path = Path(request.output_path)
        match outcome:
            case Reject(code, message):
                raise ModelServerRejected(code, message, 500)
            case Timeout():
                raise ModelServerTimeout("generate timed out")
            case Unavailable():
                raise ModelServerUnavailable("model server unreachable (ConnectError)")
            case Garbage():
                raise ModelServerProtocolError("HTTP 200 body does not match the contract")
            case BadFile(kind):
                if kind == "empty":
                    path.write_bytes(b"")
                elif kind == "not_mp4":
                    path.write_bytes(b"this is not an mp4 file at all")
            case WrongPath():
                path = path.with_name("elsewhere.mp4")
            case _:
                path.write_bytes(FTYP_FILE)
        return GenerateResponse(
            ok=True,
            output_path=str(path),
            render=RenderReport(
                render_seconds=self.render_seconds,
                width=request.params.width or 576,
                height=request.params.height or 1024,
                num_frames=request.params.num_frames or 361,
                fps=request.params.fps or 24,
                duration_seconds=15.04,
                has_audio=True,
            ),
        )
