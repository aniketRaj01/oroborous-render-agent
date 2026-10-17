from __future__ import annotations

import json
import logging
from typing import Protocol

import httpx
from pydantic import ValidationError

from render_agent.config import Settings
from render_agent.model_server.models import (
    ErrorEnvelope,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from render_agent.redaction import safe_text

logger = logging.getLogger(__name__)

HEALTH_TIMEOUT_SECONDS = 5.0


class ModelServerError(Exception):
    """Base class. Messages never contain the prompt or the URL."""


class ModelServerUnavailable(ModelServerError):
    """Connection refused or dropped: the server is down or restarting."""


class ModelServerTimeout(ModelServerError):
    """No answer within the generate timeout."""


class ModelServerProtocolError(ModelServerError):
    """The response does not match the model server contract."""


class ModelServerRejected(ModelServerError):
    """The server answered with its error envelope."""

    def __init__(self, code: str, message: str, http_status: int) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.http_status = http_status


class ModelServer(Protocol):
    """What the run loop needs from the local model server. Tests substitute a fake."""

    def is_ready(self) -> bool: ...

    def generate(self, request: GenerateRequest) -> GenerateResponse: ...


class HttpModelServer:
    def __init__(
        self,
        *,
        base_url: str,
        http: httpx.Client,
        generate_timeout_seconds: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http
        self._generate_timeout = generate_timeout_seconds

    def __repr__(self) -> str:
        return "HttpModelServer()"

    def is_ready(self) -> bool:
        """Not ready covers refused connections, non-200 answers and modelLoaded false."""
        try:
            response = self._http.get(f"{self._base_url}/health", timeout=HEALTH_TIMEOUT_SECONDS)
        except httpx.HTTPError:
            return False
        if response.status_code != 200:
            return False
        try:
            return HealthResponse.model_validate_json(response.content).model_loaded
        except ValidationError:
            return False

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        try:
            response = self._http.post(
                f"{self._base_url}/generate",
                content=request.to_json_bytes(),
                headers={"content-type": "application/json"},
                timeout=self._generate_timeout,
            )
        except httpx.TimeoutException:
            raise ModelServerTimeout("generate timed out") from None
        except httpx.TransportError as exc:
            raise ModelServerUnavailable(
                f"model server unreachable ({type(exc).__name__})"
            ) from None

        try:
            payload: object = json.loads(response.content)
        except ValueError:
            payload = None

        if response.status_code == 200:
            try:
                return GenerateResponse.model_validate(payload)
            except ValidationError:
                raise ModelServerProtocolError(
                    "HTTP 200 body does not match the contract"
                ) from None
        try:
            envelope = ErrorEnvelope.model_validate(payload)
        except ValidationError:
            raise ModelServerProtocolError(
                f"unexpected HTTP {response.status_code} without the error envelope"
            ) from None
        raise ModelServerRejected(
            safe_text(envelope.error.code, 50),
            safe_text(envelope.error.message, 500),
            response.status_code,
        )


def create_model_server(
    settings: Settings, *, transport: httpx.BaseTransport | None = None
) -> HttpModelServer:
    return HttpModelServer(
        base_url=settings.model_server_url,
        http=httpx.Client(follow_redirects=False, transport=transport, trust_env=False),
        generate_timeout_seconds=settings.model_server_generate_timeout_seconds,
    )
