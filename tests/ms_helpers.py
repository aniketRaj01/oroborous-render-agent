"""Helpers for the model server tests: settings, a loopback server on a free port, raw HTTP."""

from __future__ import annotations

import http.client
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ltx_server.config import Settings
from ltx_server.renderer import Renderer
from ltx_server.server import ServerApp


@dataclass
class Running:
    app: ServerApp
    port: int
    out: Path

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def path(self, name: str = "412-1.mp4") -> str:
        return str(self.out / name)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    out = tmp_path / "scratch"
    out.mkdir(exist_ok=True)
    base = Settings(port=0, renderer="fake", allowed_output_dir=out, shutdown_grace_seconds=2.0)
    return replace(base, **overrides)


def wait_until(condition: Any, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


@contextmanager
def running(
    settings: Settings, renderer: Renderer, *, wait_ready: bool = True
) -> Iterator[Running]:
    app = ServerApp(settings, renderer, probe=lambda path: None)
    port = app.start()
    handle = Running(app, port, settings.allowed_output_dir)
    try:
        if wait_ready:
            assert wait_until(lambda: app.ready), "the server did not become ready"
        yield handle
    finally:
        app.request_stop(0)
        app.shutdown()


def request(
    port: int,
    method: str,
    path: str,
    body: bytes | str | dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> tuple[int, Any, bytes]:
    """One raw HTTP request. Returns (status, parsed JSON or None, raw body)."""
    if isinstance(body, dict):
        body = json.dumps(body)
    payload = body.encode() if isinstance(body, str) else body
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, body=payload, headers=headers or {})
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    return response.status, parsed, raw


def generate_body(output_path: str, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "jobId": 412,
        "prompt": "a calm lake",
        "params": {"width": 576, "height": 1024, "numFrames": 361, "fps": 24, "seed": 42},
        "outputPath": output_path,
    }
    body.update(overrides)
    return body
