"""Memory hygiene, the out-of-memory restart policy, and the promise never to leak the prompt."""

from __future__ import annotations

import io
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from ltx_server import logs
from ltx_server.renderer import FakeRenderer, RenderRequest, RenderResult
from ltx_server.server import EXIT_TOO_MANY_OOMS, ServerApp
from ms_helpers import generate_body, make_settings, request, running, wait_until

SENTINEL = "SENTINEL-PROMPT-7f3a9c-a purple elephant on a bicycle"


@contextmanager
def captured_logs() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    logs.configure_logging(logging.DEBUG, stream)
    try:
        yield stream
    finally:
        logs.configure_logging(logging.INFO)


def post(port: int, path: str, **overrides: object) -> tuple[int, dict]:  # type: ignore[type-arg]
    status, body, _ = request(port, "POST", "/generate", generate_body(path, **overrides))
    return status, body


# --- after_clip and recover ---------------------------------------------------------------------


def test_after_clip_runs_after_every_clip_success_or_failure(tmp_path: Path) -> None:
    renderer = FakeRenderer(script={2: "render_error", 3: "mid_write", 4: "oom"})
    with running(make_settings(tmp_path, max_consecutive_ooms=9), renderer) as server:
        for index in range(1, 5):
            post(server.port, server.path(f"{index}-1.mp4"))
            assert renderer.after_clips == index
    assert renderer.renders == 4 and renderer.recovers == 1  # recover only after the oom


def test_after_clip_is_not_called_for_requests_that_never_reach_the_renderer(
    tmp_path: Path,
) -> None:
    gate, started = threading.Event(), threading.Event()
    renderer = FakeRenderer(render_gate=gate, render_started=started)
    with running(make_settings(tmp_path), renderer) as server:
        first = threading.Thread(target=post, args=(server.port, server.path("1-1.mp4")))
        first.start()
        assert started.wait(10)
        assert post(server.port, server.path("2-1.mp4"))[0] == 409  # busy
        assert post(server.port, server.path("3-1.mp4"), jobId="x")[0] == 400  # invalid
        assert renderer.after_clips == 0  # the running clip has not finished yet
        gate.set()
        first.join(10)
        assert renderer.after_clips == 1


def test_a_failing_after_clip_does_not_break_the_response(tmp_path: Path) -> None:
    class Flaky(FakeRenderer):
        def after_clip(self) -> None:
            super().after_clip()
            raise RuntimeError("cuda cache failure")

    with running(make_settings(tmp_path), Flaky()) as server:
        assert post(server.port, server.path())[0] == 200


# --- consecutive out-of-memory restarts --------------------------------------------------------


def test_one_oom_is_answered_and_the_server_keeps_running(tmp_path: Path) -> None:
    renderer = FakeRenderer(script={1: "oom"})
    with running(make_settings(tmp_path), renderer) as server:
        status, body = post(server.port, server.path())
        assert status == 500 and body["error"]["code"] == "oom"
        assert server.app.exit_code == 0 and post(server.port, server.path("2-1.mp4"))[0] == 200


def test_two_ooms_in_a_row_answer_then_request_a_clean_exit(tmp_path: Path) -> None:
    renderer = FakeRenderer(script={1: "oom", 2: "oom"})
    app = ServerApp(make_settings(tmp_path), renderer, probe=lambda path: None)
    port = app.start()
    assert wait_until(lambda: app.ready)
    first = post(port, str(tmp_path / "scratch" / "1-1.mp4"))
    assert first[1]["error"]["code"] == "oom" and app.exit_code == 0
    second = post(port, str(tmp_path / "scratch" / "2-1.mp4"))
    assert second[0] == 500 and second[1]["error"]["code"] == "oom"  # the request IS answered
    assert wait_until(lambda: app.exit_code == EXIT_TOO_MANY_OOMS)
    assert app.wait() == EXIT_TOO_MANY_OOMS != 0  # the exit path, without exiting this process
    assert renderer.recovers == 2


def test_a_success_or_other_failure_resets_the_oom_count(tmp_path: Path) -> None:
    renderer = FakeRenderer(script={1: "oom", 2: "render_error", 3: "oom", 4: "oom"})
    with running(make_settings(tmp_path), renderer) as server:
        for index in (1, 2, 3):
            post(server.port, server.path(f"{index}-1.mp4"))
        assert server.app.exit_code == 0  # oom, other, oom: never two in a row
        post(server.port, server.path("4-1.mp4"))
        assert wait_until(lambda: server.app.exit_code == EXIT_TOO_MANY_OOMS)


def test_the_oom_limit_is_configurable(tmp_path: Path) -> None:
    renderer = FakeRenderer(fail_every=1, fail_kind="oom")
    with running(make_settings(tmp_path, max_consecutive_ooms=3), renderer) as server:
        for index in (1, 2):
            post(server.port, server.path(f"{index}-1.mp4"))
        assert server.app.exit_code == 0
        post(server.port, server.path("3-1.mp4"))
        assert wait_until(lambda: server.app.exit_code == EXIT_TOO_MANY_OOMS)


@pytest.mark.parametrize(
    "error", [MemoryError("host memory"), type("OutOfMemoryError", (Exception,), {})("cuda")]
)
def test_memory_errors_and_torch_style_oom_errors_are_classified_as_oom(
    tmp_path: Path, error: Exception
) -> None:
    class Raising(FakeRenderer):
        def render(self, request: RenderRequest, tmp_path: Path) -> RenderResult:
            raise error

    with running(make_settings(tmp_path), Raising()) as server:
        status, body = post(server.port, server.path())
    assert status == 500 and body["error"]["code"] == "oom"


# --- the prompt never leaks ---------------------------------------------------------------------


def test_the_prompt_never_appears_in_logs_or_responses(tmp_path: Path) -> None:
    class Leaky(FakeRenderer):
        def render(self, request: RenderRequest, tmp_path: Path) -> RenderResult:
            if self.renders == 0:
                self.renders = 1
                raise ValueError(f"cannot render: {request.prompt}")
            return super().render(request, tmp_path)

    with captured_logs() as stream, running(make_settings(tmp_path), Leaky()) as server:
        seen = [
            post(server.port, server.path("1-1.mp4"), prompt=SENTINEL),  # render error
            post(server.port, server.path("2-1.mp4"), prompt=SENTINEL),  # success
            post(server.port, server.path("3-1.mp4"), prompt=SENTINEL, params={"width": 5}),
            post(server.port, server.path("4-1.mp4"), prompt=SENTINEL, extra=SENTINEL),
            post(server.port, "/etc/x.mp4", prompt=SENTINEL),
        ]
        raw = [
            request(server.port, "POST", "/generate", generate_body(server.path(), prompt=SENTINEL))
        ]
    assert [status for status, _ in seen] == [500, 200, 400, 400, 400]
    assert "ValueError" in seen[0][1]["error"]["message"]
    everything = stream.getvalue() + repr(seen) + repr(raw)
    assert "SENTINEL-PROMPT" not in everything and "purple elephant" not in everything
    assert (
        "event=generate_failed" in stream.getvalue() and "event=generate_done" in stream.getvalue()
    )


def test_the_prompt_is_not_in_the_request_object_repr() -> None:
    request_ = RenderRequest(1, SENTINEL, 64, 64, 9, 24.0, 1)
    assert "SENTINEL" not in repr(request_)


def test_logs_carry_ids_sizes_timings_and_outcomes(tmp_path: Path) -> None:
    with captured_logs() as stream, running(make_settings(tmp_path), FakeRenderer()) as server:
        post(server.port, server.path())
    text = stream.getvalue()
    for expected in (
        "event=listening",
        "event=model_loaded",
        "event=ready",
        "event=generate_start",
        "job_id=412",
        "width=576",
        "frames=361",
        "event=generate_done",
        "render_s=",
    ):
        assert expected in text, expected
