from __future__ import annotations

import json

import httpx
import pytest

from render_agent.model_server.client import (
    HttpModelServer,
    ModelServerProtocolError,
    ModelServerRejected,
    ModelServerTimeout,
    ModelServerUnavailable,
)
from render_agent.model_server.models import GenerateParams, GenerateRequest

PROMPT = "a secret prompt about a cat"
OK_BODY = {
    "ok": True,
    "outputPath": "/scratch/412-1.mp4",
    "render": {
        "renderSeconds": 142.6,
        "width": 576,
        "height": 1024,
        "numFrames": 361,
        "fps": 24,
        "durationSeconds": 15.04,
        "hasAudio": True,
    },
}


def server(handler: httpx.MockTransport | None = None, *, fn=None) -> HttpModelServer:  # type: ignore[no-untyped-def]
    transport = handler or httpx.MockTransport(fn)
    return HttpModelServer(
        base_url="http://127.0.0.1:8000",
        http=httpx.Client(transport=transport),
        generate_timeout_seconds=1200,
    )


def request(params: GenerateParams | None = None) -> GenerateRequest:
    return GenerateRequest(
        job_id=412,
        prompt=PROMPT,
        params=params or GenerateParams(),
        output_path="/scratch/412-1.mp4",
    )


def respond(response: httpx.Response) -> HttpModelServer:
    return server(fn=lambda req: response)


def raise_(exc: Exception) -> HttpModelServer:
    def handler(req: httpx.Request) -> httpx.Response:
        raise exc

    return server(fn=handler)


# --- health ------------------------------------------------------------------------------------


def test_ready_only_when_model_is_loaded() -> None:
    assert respond(httpx.Response(200, json={"ok": True, "modelLoaded": True})).is_ready()
    assert not respond(httpx.Response(200, json={"ok": True, "modelLoaded": False})).is_ready()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, json={"ok": True, "modelLoaded": True}),
        httpx.Response(200, content=b"<html>loading</html>"),
        httpx.Response(200, json={"modelLoaded": True}),
        httpx.Response(200, json={"ok": True}),
    ],
)
def test_anything_unexpected_means_not_ready(response: httpx.Response) -> None:
    assert not respond(response).is_ready()


def test_connection_refused_and_timeouts_mean_not_ready() -> None:
    assert not raise_(httpx.ConnectError("refused")).is_ready()
    assert not raise_(httpx.ReadTimeout("slow")).is_ready()


def test_health_uses_a_short_timeout_and_generate_the_long_one() -> None:
    seen: dict[str, float | None] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen[req.url.path] = req.extensions["timeout"]["read"]
        if req.url.path == "/health":
            return httpx.Response(200, json={"ok": True, "modelLoaded": True})
        return httpx.Response(200, json=OK_BODY)

    model = server(fn=handler)
    model.is_ready()
    model.generate(request())
    assert seen == {"/health": 5.0, "/generate": 1200.0}


# --- generate ----------------------------------------------------------------------------------


def test_generate_sends_only_the_params_the_job_has_and_parses_the_response() -> None:
    sent: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(req)
        return httpx.Response(200, json=OK_BODY)

    result = server(fn=handler).generate(request(GenerateParams(width=320, seed=7)))
    body = json.loads(sent[0].content)
    assert body == {
        "jobId": 412,
        "prompt": PROMPT,
        "params": {"width": 320, "seed": 7},
        "outputPath": "/scratch/412-1.mp4",
    }
    assert sent[0].method == "POST" and sent[0].url.path == "/generate"
    assert result.output_path == "/scratch/412-1.mp4"
    assert result.render.render_seconds == 142.6 and result.render.has_audio is True


def test_empty_params_are_an_empty_object() -> None:
    sent: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(req)
        return httpx.Response(200, json=OK_BODY)

    server(fn=handler).generate(request())
    assert json.loads(sent[0].content)["params"] == {}


@pytest.mark.parametrize(
    ("status", "code"),
    [(500, "oom"), (500, "render_error"), (400, "invalid_request"), (409, "busy")],
)
def test_each_error_code_becomes_a_rejection(status: int, code: str) -> None:
    body = {"ok": False, "error": {"code": code, "message": "something went wrong"}}
    with pytest.raises(ModelServerRejected) as caught:
        respond(httpx.Response(status, json=body)).generate(request())
    assert caught.value.code == code and caught.value.http_status == status
    assert "something went wrong" in str(caught.value)


def test_timeout_and_connection_errors_have_their_own_types() -> None:
    with pytest.raises(ModelServerTimeout):
        raise_(httpx.ReadTimeout("slow")).generate(request())
    with pytest.raises(ModelServerUnavailable) as caught:
        raise_(httpx.ConnectError("refused at 127.0.0.1")).generate(request())
    assert "127.0.0.1" not in str(caught.value) and "ConnectError" in str(caught.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"ok": True, "outputPath": "/x"}),
        httpx.Response(500, content=b"Internal Server Error"),
        httpx.Response(502, json={"detail": "bad gateway"}),
        httpx.Response(404, json={"ok": False}),
    ],
)
def test_non_contract_responses_are_protocol_errors(response: httpx.Response) -> None:
    with pytest.raises(ModelServerProtocolError):
        respond(response).generate(request())


def test_error_messages_are_truncated_and_never_contain_the_prompt_in_the_repr() -> None:
    long = "x" * 5000
    body = {"ok": False, "error": {"code": "render_error", "message": long}}
    with pytest.raises(ModelServerRejected) as caught:
        respond(httpx.Response(500, json=body)).generate(request())
    assert len(str(caught.value)) < 600
    assert PROMPT not in repr(request())
    assert repr(server(fn=lambda r: httpx.Response(200))) == "HttpModelServer()"
