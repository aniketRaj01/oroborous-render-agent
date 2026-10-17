"""Fake model server for manual runs of the render agent. NOT a video model.

Implements the HTTP contract in docs/model-server-contract.md with the standard library only.
It "renders" by writing a tiny file that starts with a valid-looking mp4 `ftyp` box followed by
filler bytes. The output is NOT a playable video and must never be treated as one.

Binds to 127.0.0.1 only. Environment variables control its behaviour:

  FAKE_MODEL_DELAY_SECONDS        pretend render time per clip (default 1)
  FAKE_MODEL_READY_AFTER_SECONDS  report modelLoaded=false for this long after start (default 0)
  FAKE_MODEL_FAIL_EVERY           fail every Nth generate request, 0 = never (default 0)
  FAKE_MODEL_FAIL_CODE            error code used for injected failures:
                                  oom | render_error | invalid_request | busy (default render_error)

Usage: python scripts/fake_model_server.py [--port 8000]   (--port 0 picks a free port and
prints it on the first line of standard output).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
FTYP_BOX = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
FILLER = b"FAKE-NOT-A-VIDEO\n" * 64
STATUS = {"oom": 500, "render_error": 500, "invalid_request": 400, "busy": 409}
DEFAULTS = {"width": 576, "height": 1024, "numFrames": 361, "fps": 24}


class State:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.delay = float(os.environ.get("FAKE_MODEL_DELAY_SECONDS", "1"))
        self.ready_after = float(os.environ.get("FAKE_MODEL_READY_AFTER_SECONDS", "0"))
        self.fail_every = int(os.environ.get("FAKE_MODEL_FAIL_EVERY", "0"))
        self.fail_code = os.environ.get("FAKE_MODEL_FAIL_CODE", "render_error")
        self.requests = 0
        self.busy = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    state: State

    def log_message(self, format: str, *args: Any) -> None:
        pass  # never log request lines: they could carry paths

    def _send(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, code: str, message: str) -> None:
        self._send(STATUS[code], {"ok": False, "error": {"code": code, "message": message}})

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(404, {"ok": False, "error": {"code": "invalid_request", "message": "no"}})
            return
        loaded = time.monotonic() - self.state.started >= self.state.ready_after
        self._send(200, {"ok": True, "modelLoaded": loaded})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._error("invalid_request", "unknown route")
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            request = json.loads(self.rfile.read(length))
            job_id, params = request["jobId"], request["params"]
            output = Path(request["outputPath"])
            valid = (
                isinstance(request["prompt"], str)
                and isinstance(params, dict)
                and isinstance(job_id, int)
                and output.is_absolute()
                and output.suffix == ".mp4"
            )
        except (ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            self._error("invalid_request", "malformed request")
            return
        if not self.state.busy.acquire(blocking=False):
            self._error("busy", "one generation at a time")
            return
        try:
            self.state.requests += 1
            fail_every = self.state.fail_every
            if fail_every and self.state.requests % fail_every == 0:
                self._error(self.state.fail_code, f"injected failure for job {job_id}")
                return
            time.sleep(self.state.delay)
            output.write_bytes(FTYP_BOX + FILLER)
            settings = {**DEFAULTS, **params}
            self._send(
                200,
                {
                    "ok": True,
                    "outputPath": str(output),
                    "render": {
                        "renderSeconds": self.state.delay,
                        "width": settings["width"],
                        "height": settings["height"],
                        "numFrames": settings["numFrames"],
                        "fps": settings["fps"],
                        "durationSeconds": settings["numFrames"] / settings["fps"],
                        "hasAudio": False,
                    },
                },
            )
            print(f"generated job {job_id}", file=sys.stderr, flush=True)
        except OSError:
            self._error("render_error", "could not write the output file")
        finally:
            self.state.busy.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake model server (not a video model).")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    Handler.state = State()
    server = ThreadingHTTPServer((HOST, args.port), Handler)
    print(f"listening on {HOST}:{server.server_address[1]}", flush=True)
    print(
        "NOTE: this is a FAKE model server. Its output files are not playable videos.",
        file=sys.stderr,
        flush=True,
    )
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
