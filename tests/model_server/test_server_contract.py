"""The agent's real HttpModelServer client against the real server shell, FakeRenderer behind it."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from ltx_server.config import ConfigError, Settings, load_settings
from ltx_server.renderer import FakeRenderer
from ltx_server.server import ServerApp
from ms_helpers import generate_body, make_settings, request, running, wait_until
from render_agent.model_server.client import HttpModelServer, ModelServerRejected
from render_agent.model_server.models import GenerateParams, GenerateRequest

FTYP = b"ftyp"


def client_for(port: int, timeout: float = 15.0) -> HttpModelServer:
    return HttpModelServer(
        base_url=f"http://127.0.0.1:{port}", http=httpx.Client(), generate_timeout_seconds=timeout
    )


def agent_request(out: str, **params: Any) -> GenerateRequest:
    return GenerateRequest(
        job_id=412, prompt="a calm lake", params=GenerateParams(**params), output_path=out
    )


# --- readiness -----------------------------------------------------------------------------------


def test_health_is_false_until_load_and_warm_up_finish(tmp_path: Path) -> None:
    load, warm = threading.Event(), threading.Event()
    renderer = FakeRenderer(load_gate=load, warmup_gate=warm)
    with running(make_settings(tmp_path), renderer, wait_ready=False) as server:
        client = client_for(server.port)
        assert request(server.port, "GET", "/health")[1] == {"ok": True, "modelLoaded": False}
        assert not client.is_ready()
        load.set()  # loaded, but still warming up
        assert wait_until(lambda: renderer.warm_ups == 1)
        assert request(server.port, "GET", "/health")[1] == {"ok": True, "modelLoaded": False}
        assert not client.is_ready()
        warm.set()
        assert wait_until(client.is_ready)
        assert request(server.port, "GET", "/health")[1] == {"ok": True, "modelLoaded": True}


def test_generate_while_loading_is_a_503_envelope(tmp_path: Path) -> None:
    gate = threading.Event()
    with running(make_settings(tmp_path), FakeRenderer(load_gate=gate), wait_ready=False) as server:
        status, body, _ = request(server.port, "POST", "/generate", generate_body(server.path()))
        assert status == 503
        assert body == {
            "ok": False,
            "error": {"code": "render_error", "message": "model is still loading"},
        }
        with pytest.raises(ModelServerRejected) as caught:
            client_for(server.port).generate(agent_request(server.path()))
        assert caught.value.code == "render_error" and caught.value.http_status == 503
        gate.set()
        assert wait_until(lambda: server.app.ready)
        assert client_for(server.port).generate(agent_request(server.path())).ok


def test_a_failed_load_exits_non_zero(tmp_path: Path) -> None:
    class Broken(FakeRenderer):
        def load(self) -> None:
            raise RuntimeError("weights missing")

    app = ServerApp(make_settings(tmp_path), Broken(), probe=lambda path: None)
    app.start()
    assert app.wait() == 3
    assert not app.ready


# --- success through the agent's client ----------------------------------------------------------


def test_the_agents_client_gets_a_contract_response(tmp_path: Path) -> None:
    renderer = FakeRenderer()
    with running(make_settings(tmp_path), renderer) as server:
        response = client_for(server.port).generate(
            agent_request(server.path(), width=576, height=1024, num_frames=361, fps=24, seed=7)
        )
    assert response.ok and response.output_path == server.path()
    render = response.render
    assert (render.width, render.height, render.num_frames, render.fps) == (576, 1024, 361, 24)
    assert render.has_audio is True and render.render_seconds >= 0
    assert render.duration_seconds == pytest.approx(361 / 24, abs=0.01)  # fallback: frames / fps
    written = Path(server.path()).read_bytes()
    assert written[4:8] == FTYP
    assert renderer.requests[0].seed == 7 and renderer.requests[0].prompt == "a calm lake"


def test_missing_params_get_the_settings_defaults(tmp_path: Path) -> None:
    renderer = FakeRenderer()
    with running(make_settings(tmp_path), renderer) as server:
        client_for(server.port).generate(agent_request(server.path()))
    only = renderer.requests[0]
    assert (only.width, only.height, only.num_frames, only.fps, only.seed) == (
        576,
        1024,
        361,
        24,
        42,
    )


def test_defaults_come_from_the_settings(tmp_path: Path) -> None:
    renderer = FakeRenderer()
    settings = make_settings(tmp_path, default_width=320, default_num_frames=17, default_seed=1)
    with running(settings, renderer) as server:
        client_for(server.port).generate(agent_request(server.path()))
    only = renderer.requests[0]
    assert (only.width, only.num_frames, only.seed) == (320, 17, 1)


def test_ffprobe_duration_is_used_when_available(tmp_path: Path) -> None:
    app = ServerApp(make_settings(tmp_path), FakeRenderer(), probe=lambda path: 15.04)
    port = app.start()
    try:
        assert wait_until(lambda: app.ready)
        out = str(tmp_path / "scratch" / "1-1.mp4")
        response = client_for(port).generate(agent_request(out))
        assert response.render.duration_seconds == 15.04
    finally:
        app.request_stop(0)
        app.shutdown()


# --- every error code ----------------------------------------------------------------------------


def test_oom_and_render_error_use_the_contract_codes(tmp_path: Path) -> None:
    renderer = FakeRenderer(script={1: "oom", 2: "render_error"})
    with running(make_settings(tmp_path, max_consecutive_ooms=5), renderer) as server:
        client = client_for(server.port)
        with pytest.raises(ModelServerRejected) as oom:
            client.generate(agent_request(server.path()))
        with pytest.raises(ModelServerRejected) as failed:
            client.generate(agent_request(server.path()))
    assert (oom.value.code, oom.value.http_status) == ("oom", 500)
    assert (failed.value.code, failed.value.http_status) == ("render_error", 500)


def test_invalid_params_through_the_client_are_invalid_request(tmp_path: Path) -> None:
    with (
        running(make_settings(tmp_path), FakeRenderer()) as server,
        pytest.raises(ModelServerRejected) as caught,
    ):
        client_for(server.port).generate(agent_request(server.path(), width=100))
    assert (caught.value.code, caught.value.http_status) == ("invalid_request", 400)


def test_a_second_concurrent_generate_gets_busy_without_disturbing_the_first(
    tmp_path: Path,
) -> None:
    gate, started = threading.Event(), threading.Event()
    renderer = FakeRenderer(render_gate=gate, render_started=started)
    with running(make_settings(tmp_path), renderer) as server:
        results: list[object] = []
        first = threading.Thread(
            target=lambda: results.append(
                client_for(server.port).generate(agent_request(server.path("1-1.mp4")))
            )
        )
        first.start()
        assert started.wait(10)
        with pytest.raises(ModelServerRejected) as busy:
            client_for(server.port).generate(agent_request(server.path("2-1.mp4")))
        assert (busy.value.code, busy.value.http_status) == ("busy", 409)
        gate.set()
        first.join(10)
        assert len(results) == 1 and Path(server.path("1-1.mp4")).is_file()
        assert not Path(server.path("2-1.mp4")).exists()
    assert renderer.renders == 1


# --- invalid requests ----------------------------------------------------------------------------

BAD_BODIES: dict[str, str | dict[str, Any]] = {
    "not json": "{nope",
    "array": "[]",
    "empty object": "{}",
    "unknown field": {"extra": 1},
    "job id string": {"jobId": "412"},
    "job id bool": {"jobId": True},
    "prompt empty": {"prompt": ""},
    "prompt blank": {"prompt": "   "},
    "prompt number": {"prompt": 5},
    "params list": {"params": []},
    "unknown param": {"params": {"steps": 8}},
    "width not 64": {"params": {"width": 100}},
    "width zero": {"params": {"width": 0}},
    "width negative": {"params": {"width": -64}},
    "width string": {"params": {"width": "576"}},
    "width bool": {"params": {"width": True}},
    "width float": {"params": {"width": 576.0}},
    "width huge": {"params": {"width": 64 * 100}},
    "height not 64": {"params": {"height": 1000}},
    "frames not 8k+1": {"params": {"numFrames": 360}},
    "frames too few": {"params": {"numFrames": 1}},
    "frames over max": {"params": {"numFrames": 489}},
    "frames string": {"params": {"numFrames": "361"}},
    "fps zero": {"params": {"fps": 0}},
    "fps over 60": {"params": {"fps": 61}},
    "fps string": {"params": {"fps": "24"}},
    "fps bool": {"params": {"fps": True}},
    "seed negative": {"params": {"seed": -1}},
    "seed float": {"params": {"seed": 1.5}},
    "seed bool": {"params": {"seed": True}},
    "output relative": {"outputPath": "412-1.mp4"},
    "output not mp4": {"outputPath": "@dir/412-1.mov"},
    "output only extension": {"outputPath": "@dir/.mp4"},
    "output number": {"outputPath": 5},
    "output elsewhere": {"outputPath": "/etc/passwd.mp4"},
    "output dotdot": {"outputPath": "@dir/../escaped.mp4"},
    "output deep dotdot": {"outputPath": "@dir/sub/../../escaped.mp4"},
    "output missing dir": {"outputPath": "@dir/nope/412-1.mp4"},
    "output nul": {"outputPath": "@dir/a\u0000.mp4"},
}


@pytest.mark.parametrize("name", sorted(BAD_BODIES))
def test_invalid_requests_are_400_invalid_request(tmp_path: Path, name: str) -> None:
    renderer = FakeRenderer()
    with running(make_settings(tmp_path), renderer) as server:
        sample = BAD_BODIES[name]
        payload: object
        if isinstance(sample, str):
            payload = sample
        else:
            fixed = {
                key: (value.replace("@dir", str(server.out)) if isinstance(value, str) else value)
                for key, value in sample.items()
            }
            payload = generate_body(server.path(), **fixed)
            if name == "empty object":
                payload = {}
        status, body, _ = request(server.port, "POST", "/generate", payload)
    assert status == 400, body
    assert body["ok"] is False and body["error"]["code"] == "invalid_request"
    assert renderer.renders == 0 and list(server.out.iterdir()) == []


def test_edge_values_that_are_valid(tmp_path: Path) -> None:
    renderer = FakeRenderer()
    with running(make_settings(tmp_path), renderer) as server:
        for params in (
            {"width": 64, "height": 64, "numFrames": 9, "fps": 60, "seed": 0},
            {"width": 576, "height": 1024, "numFrames": 481, "fps": 0.5},
            {"fps": 23.976},
        ):
            status, _, _ = request(
                server.port, "POST", "/generate", generate_body(server.path(), params=params)
            )
            assert status == 200, params


def test_the_body_size_limit(tmp_path: Path) -> None:
    with running(make_settings(tmp_path, max_body_bytes=2048), FakeRenderer()) as server:
        big = generate_body(server.path(), prompt="x" * 5000)
        status, body, _ = request(server.port, "POST", "/generate", big)
        assert status == 413 and body["error"]["code"] == "invalid_request"
        status, body, _ = request(server.port, "POST", "/generate", generate_body(server.path()))
        assert status == 200


def test_a_content_length_is_required_and_chunking_is_refused(tmp_path: Path) -> None:
    with running(make_settings(tmp_path), FakeRenderer()) as server:
        status, body, _ = request(server.port, "POST", "/generate", None)
        assert status == 400 and body["error"]["code"] == "invalid_request"
        status, body, _ = request(
            server.port, "POST", "/generate", "{}", headers={"Transfer-Encoding": "chunked"}
        )
        assert status == 400 and body["error"]["code"] == "invalid_request"


# --- routes and bind address -----------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        ("GET", "/nope", 404),
        ("POST", "/nope", 404),
        ("GET", "/", 404),
        ("GET", "/generate", 405),
        ("POST", "/health", 405),
        ("PUT", "/generate", 405),
        ("DELETE", "/health", 405),
    ],
)
def test_unknown_routes_and_methods_use_the_envelope(
    tmp_path: Path, method: str, path: str, status: int
) -> None:
    with running(make_settings(tmp_path), FakeRenderer()) as server:
        got, body, _ = request(server.port, method, path, "{}" if method != "GET" else None)
    assert got == status and body["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "192.168.1.5", "example.invalid"])
def test_non_loopback_binds_are_refused(host: str) -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings({"HOST": host})
    assert "HOST must be a loopback address" in str(caught.value) and host not in str(caught.value)
    with pytest.raises(ValueError, match="loopback"):
        ServerApp(Settings(host=host, port=0), FakeRenderer()).start()


def test_the_server_listens_on_loopback_only(tmp_path: Path) -> None:
    with running(make_settings(tmp_path), FakeRenderer()) as server:
        assert server.app._httpd is not None
        assert server.app._httpd.server_address[0] == "127.0.0.1"
