"""HTTP shell: GET /health and POST /generate on loopback, exactly per the model server contract.

Standard library only. The renderer (a fake, or the LTX binding) is injected.
"""

from __future__ import annotations

import json
import logging
import signal
import socket
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import NamedTuple, cast

from ltx_server.config import LOOPBACK_HOSTS, Settings
from ltx_server.logs import event, phase_fields
from ltx_server.output import commit, remove_quietly, remove_stale_temp_files, temp_path_for
from ltx_server.probe import probe_duration
from ltx_server.renderer import Renderer, RendererOutOfMemory, RenderRequest
from ltx_server.request import GenerateInput, RequestError, parse_request

EXIT_LOAD_FAILED = 3
EXIT_TOO_MANY_OOMS = 75
READ_TIMEOUT_SECONDS = 30.0
OOM_ERROR_NAMES = frozenset({"OutOfMemoryError"})  # torch.cuda.OutOfMemoryError, matched by name


class Response(NamedTuple):
    status: int
    body: dict[str, object]
    after_send: Callable[[], None] | None = None


def error_response(status: int, code: str, message: str) -> Response:
    return Response(status, {"ok": False, "error": {"code": code, "message": message}})


def is_out_of_memory(exc: BaseException) -> bool:
    return (
        isinstance(exc, (RendererOutOfMemory, MemoryError)) or type(exc).__name__ in OOM_ERROR_NAMES
    )


def _where(exc: BaseException) -> str:
    """The innermost frame as file:line:function. The exception text can quote the prompt."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return "unknown"
    last = frames[-1]
    return f"{Path(last.filename).name}:{last.lineno}:{last.name}"


class ServerApp:
    def __init__(
        self,
        settings: Settings,
        renderer: Renderer,
        *,
        probe: Callable[[Path], float | None] = probe_duration,
    ) -> None:
        self.settings = settings
        self.renderer = renderer
        self._probe = probe
        self.exit_code = 0
        self.port = 0
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._stop_requested = threading.Event()
        self._generation = threading.Lock()
        self._state = threading.Lock()
        self._in_flight = 0
        self._idle = threading.Condition()
        self._consecutive_ooms = 0
        self._httpd: _HttpServer | None = None
        self._serve_thread: threading.Thread | None = None
        self._load_thread: threading.Thread | None = None

    # --- lifecycle -----------------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and not self._stopping.is_set()

    def start(self) -> int:
        """Bind, start serving, and start loading the model in the background. Returns the port."""
        if self.settings.host not in LOOPBACK_HOSTS:
            raise ValueError("the server only binds to a loopback address")
        removed = remove_stale_temp_files(self.settings.allowed_output_dir)
        if removed:
            event(logging.INFO, "stale_temp_removed", files=removed)
        if not self.settings.allowed_output_dir.is_dir():
            event(logging.WARNING, "output_dir_missing")
        self._httpd = _HttpServer(self.settings.host, self.settings.port, self)
        self.port = int(self._httpd.server_address[1])
        self._serve_thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        self._serve_thread.start()
        self._load_thread = threading.Thread(target=self._load, name="model-load", daemon=True)
        self._load_thread.start()
        event(logging.INFO, "listening", host=self.settings.host, port=self.port)
        return self.port

    def _load(self) -> None:
        started = time.monotonic()
        try:
            self.renderer.load()
            loaded = time.monotonic()
            event(logging.INFO, "model_loaded", seconds=round(loaded - started, 1))
            self.renderer.warm_up()
            event(logging.INFO, "warm_up_done", seconds=round(time.monotonic() - loaded, 1))
        except Exception as exc:
            event(logging.ERROR, "load_failed", error_type=type(exc).__name__, where=_where(exc))
            self.request_stop(EXIT_LOAD_FAILED)
            return
        self._ready.set()
        event(logging.INFO, "ready", seconds=round(time.monotonic() - started, 1))

    def request_stop(self, code: int) -> None:
        """Ask the server to stop. The first non-zero code wins."""
        with self._state:
            if code != 0 and self.exit_code == 0:
                self.exit_code = code
        self._stop_requested.set()

    def wait(self) -> int:
        """Block until a stop is requested, then shut down cleanly and return the exit code."""
        while not self._stop_requested.wait(0.5):
            pass
        return self.shutdown()

    def shutdown(self) -> int:
        self._stopping.set()
        if self._httpd is not None:
            self._httpd.shutdown()  # stop accepting; running handlers continue
            self._httpd.server_close()
        grace = self.settings.shutdown_grace_seconds
        if self._wait_until_idle(grace):
            event(logging.INFO, "stopped", exit_code=self.exit_code)
        else:
            event(logging.WARNING, "stopped_during_generation", grace_seconds=grace)
        return self.exit_code

    @contextmanager
    def in_flight(self) -> Iterator[None]:
        """Count a /generate request from its start until its response has been written."""
        with self._idle:
            self._in_flight += 1
        try:
            yield
        finally:
            with self._idle:
                self._in_flight -= 1
                self._idle.notify_all()

    def _wait_until_idle(self, grace: float) -> bool:
        deadline = time.monotonic() + grace
        with self._idle:
            while self._in_flight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
        return True

    @property
    def can_render(self) -> bool:
        return self._ready.is_set() and not self._stopping.is_set()

    def run(self, on_started: Callable[[int], None] | None = None) -> int:
        port = self.start()
        if on_started is not None:
            on_started(port)

        def on_signal(signum: int, frame: FrameType | None) -> None:
            event(logging.INFO, "signal_received", signal=signal.Signals(signum).name)
            self.request_stop(0)

        previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            return self.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    # --- requests ------------------------------------------------------------------------------

    def health(self) -> Response:
        return Response(200, {"ok": True, "modelLoaded": self.ready})

    def generate(self, body: bytes) -> Response:
        if self._stopping.is_set():
            return error_response(503, "render_error", "the server is shutting down")
        if not self._ready.is_set():
            return error_response(503, "render_error", "model is still loading")
        try:
            request = parse_request(body, self.settings)
        except RequestError as exc:
            event(logging.INFO, "generate_rejected", reason=str(exc))
            return error_response(400, "invalid_request", str(exc))
        if not self._generation.acquire(blocking=False):
            event(logging.INFO, "generate_busy", job_id=request.job_id)
            return error_response(409, "busy", "a generation is already running")
        try:
            return self._render(request)
        finally:
            self._generation.release()

    def _render(self, request: GenerateInput) -> Response:
        event(
            logging.INFO,
            "generate_start",
            job_id=request.job_id,
            width=request.width,
            height=request.height,
            frames=request.num_frames,
            fps=request.fps,
            seed=request.seed,
            **(
                {
                    "condition_image": str(request.condition_image_path),
                    "condition_strength": request.condition_strength,
                }
                if request.condition_image_path is not None
                else {}
            ),
        )
        temp = temp_path_for(request.final_path)
        started = time.monotonic()
        try:
            result = self.renderer.render(
                RenderRequest(
                    job_id=request.job_id,
                    prompt=request.prompt,
                    width=request.width,
                    height=request.height,
                    num_frames=request.num_frames,
                    fps=request.fps,
                    seed=request.seed,
                    condition_image_path=request.condition_image_path,
                    condition_strength=request.condition_strength,
                ),
                temp,
            )
            render_seconds = time.monotonic() - started
            if not temp.is_file() or temp.stat().st_size == 0:
                raise RuntimeError("the renderer produced no output")
            duration = self._probe(temp) or result.num_frames / result.fps
            size = temp.stat().st_size
            commit(temp, request.final_path)
        except Exception as exc:
            return self._failure(request, exc, time.monotonic() - started)
        finally:
            remove_quietly(temp)
            self._after_clip()
        with self._state:
            self._consecutive_ooms = 0
        event(
            logging.INFO,
            "generate_done",
            job_id=request.job_id,
            render_s=round(render_seconds, 1),
            duration_s=round(duration, 2),
            bytes=size,
            **phase_fields(result.phase_seconds),
        )
        return Response(
            200,
            {
                "ok": True,
                "outputPath": request.output_path,
                "render": {
                    "renderSeconds": round(render_seconds, 2),
                    "width": result.width,
                    "height": result.height,
                    "numFrames": result.num_frames,
                    "fps": result.fps,
                    "durationSeconds": round(duration, 2),
                    "hasAudio": result.has_audio,
                },
            },
        )

    def _failure(self, request: GenerateInput, exc: Exception, elapsed: float) -> Response:
        oom = is_out_of_memory(exc)
        # Only the exception's type and location are reported: its text can quote the prompt.
        event(
            logging.ERROR,
            "generate_failed",
            job_id=request.job_id,
            kind="oom" if oom else "render_error",
            error_type=type(exc).__name__,
            where=_where(exc),
            elapsed_s=round(elapsed, 1),
        )
        with self._state:
            self._consecutive_ooms = self._consecutive_ooms + 1 if oom else 0
            exit_now = oom and self._consecutive_ooms >= self.settings.max_consecutive_ooms
        if not oom:
            return error_response(500, "render_error", f"render failed ({type(exc).__name__})")
        self._recover()
        after_send: Callable[[], None] | None = None
        if exit_now:
            event(logging.ERROR, "too_many_oom_exiting", count=self._consecutive_ooms)
            after_send = lambda: self.request_stop(EXIT_TOO_MANY_OOMS)  # noqa: E731
        return Response(500, error_response(500, "oom", "out of memory").body, after_send)

    def _after_clip(self) -> None:
        try:
            self.renderer.after_clip()
        except Exception as exc:
            event(logging.WARNING, "after_clip_failed", error_type=type(exc).__name__)

    def _recover(self) -> None:
        try:
            self.renderer.recover()
        except Exception as exc:
            event(logging.WARNING, "recover_failed", error_type=type(exc).__name__)


class _HttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str, port: int, app: ServerApp) -> None:
        if ":" in host:
            self.address_family = socket.AF_INET6
        self.app = app
        super().__init__((host, port), _Handler)


class _Handler(BaseHTTPRequestHandler):
    timeout = READ_TIMEOUT_SECONDS
    server_version = "ltx-server"
    sys_version = ""

    @property
    def app(self) -> ServerApp:
        return cast(_HttpServer, self.server).app

    def log_message(self, format: str, *args: object) -> None:
        pass  # request lines are not logged; the app logs its own events

    def _send(self, response: Response) -> None:
        payload = _json(response.body)
        try:
            self.send_response(response.status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        except OSError:
            event(logging.WARNING, "client_went_away")
        finally:
            if response.after_send is not None:
                response.after_send()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(self.app.health())
        elif self.path == "/generate":
            self._send(error_response(405, "invalid_request", "use POST for /generate"))
        else:
            self._send(error_response(404, "invalid_request", "unknown route"))

    def do_POST(self) -> None:
        if self.path == "/health":
            self._send(error_response(405, "invalid_request", "use GET for /health"))
            return
        if self.path != "/generate":
            self._send(error_response(404, "invalid_request", "unknown route"))
            return
        try:
            with self.app.in_flight():
                self._send(self._generate())
        except Exception as exc:
            event(logging.ERROR, "handler_failed", error_type=type(exc).__name__, where=_where(exc))
            self._send(error_response(500, "render_error", "internal error"))

    def _generate(self) -> Response:
        if self.headers.get("Transfer-Encoding"):
            return error_response(400, "invalid_request", "chunked bodies are not supported")
        raw = self.headers.get("Content-Length", "")
        if not raw.isdigit():
            return error_response(400, "invalid_request", "a Content-Length header is required")
        length = int(raw)
        if length > self.app.settings.max_body_bytes:
            return error_response(413, "invalid_request", "the request body is too large")
        if not self.app.can_render:
            return self.app.generate(b"")  # answers 503 without reading a body that may be large
        body = self.rfile.read(length)
        if len(body) != length:
            return error_response(400, "invalid_request", "the request body is incomplete")
        return self.app.generate(body)

    def _method_not_allowed(self) -> None:
        self._send(error_response(405, "invalid_request", "method not allowed"))

    do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _method_not_allowed


def _json(body: dict[str, object]) -> bytes:
    return json.dumps(body).encode("utf-8")
