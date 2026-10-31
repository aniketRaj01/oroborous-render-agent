from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable
from datetime import datetime
from typing import Protocol, TypeVar

import httpx
from botocore.exceptions import BotoCoreError
from pydantic import BaseModel, ValidationError

from render_agent.clock import Clock
from render_agent.config import Settings
from render_agent.job_api.errors import (
    AUTH_HINT,
    CONTRACT_ERRORS,
    AuthError,
    JobApiError,
    ProtocolError,
    ServerError,
    ThrottledError,
    TransientError,
    TransportError,
    UnknownContractError,
    safe_text,
)
from render_agent.job_api.models import (
    ClaimRequest,
    ClaimResponse,
    CompleteRequest,
    CompleteResponse,
    CompleteUpload,
    ErrorCode,
    ErrorEnvelope,
    FailError,
    FailErrorCode,
    FailRequest,
    FailResponse,
    RenderInfo,
    StartRequest,
    StartResponse,
)
from render_agent.job_api.signing import (
    NoCredentialsError,
    RequestSigner,
    SigV4Signer,
    UnsignedSigner,
)

logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)

CLAIM_ROUTE = "/v1/claim"
START_ROUTE = "/v1/start"
COMPLETE_ROUTE = "/v1/complete"
FAIL_ROUTE = "/v1/fail"


class JobApi(Protocol):
    """What the run loop needs from the Job API. Tests substitute a fake."""

    def claim(
        self, *, deadline_at: datetime | None = None, max_plans: int | None = None
    ) -> ClaimResponse: ...

    def start(self, job_id: int) -> StartResponse: ...

    def complete(
        self,
        job_id: int,
        *,
        upload: CompleteUpload,
        render: RenderInfo,
        cost_inr: float | None = None,
        last_frame: CompleteUpload | None = None,
    ) -> CompleteResponse: ...

    def fail(self, job_id: int, *, code: FailErrorCode, message: str) -> FailResponse: ...


class RetryPolicy:
    def __init__(
        self,
        *,
        max_retries: int,
        base_delay: float,
        max_delay: float,
        jitter: float,
        random_fraction: Callable[[], float] = random.random,
    ) -> None:
        self.max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter
        self._random_fraction = random_fraction

    def delay_before_retry(self, retry_number: int) -> float:
        """Exponential backoff, capped; jitter shortens it by up to `jitter` of its length."""
        ceiling = min(self._max_delay, self._base_delay * (2.0 ** (retry_number - 1)))
        return ceiling * (1 - self._jitter * self._random_fraction())


class JobApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        agent_id: str,
        agent_version: str,
        http: httpx.Client,
        signer: RequestSigner,
        clock: Clock,
        retry: RetryPolicy,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._agent_id = agent_id
        self._agent_version = agent_version
        self._http = http
        self._signer = signer
        self._clock = clock
        self._retry = retry

    def __repr__(self) -> str:
        return f"JobApiClient(agent_id={self._agent_id!r})"

    def claim(
        self, *, deadline_at: datetime | None = None, max_plans: int | None = None
    ) -> ClaimResponse:
        request = ClaimRequest(
            agent_id=self._agent_id,
            agent_version=self._agent_version,
            deadline_at=deadline_at,
            max_plans=max_plans,
        )
        return self._call(CLAIM_ROUTE, request.to_json_bytes(), ClaimResponse)

    def start(self, job_id: int) -> StartResponse:
        request = StartRequest(agent_id=self._agent_id, job_id=job_id)
        return self._call(START_ROUTE, request.to_json_bytes(), StartResponse)

    def complete(
        self,
        job_id: int,
        *,
        upload: CompleteUpload,
        render: RenderInfo,
        cost_inr: float | None = None,
        last_frame: CompleteUpload | None = None,
    ) -> CompleteResponse:
        request = CompleteRequest(
            agent_id=self._agent_id,
            job_id=job_id,
            upload=upload,
            render=render,
            cost_inr=cost_inr,
            last_frame=last_frame,
        )
        return self._call(COMPLETE_ROUTE, request.to_json_bytes(), CompleteResponse)

    def fail(self, job_id: int, *, code: FailErrorCode, message: str) -> FailResponse:
        request = FailRequest(
            agent_id=self._agent_id,
            job_id=job_id,
            error=FailError(code=code, message=message),
        )
        return self._call(FAIL_ROUTE, request.to_json_bytes(), FailResponse)

    def _call(self, route: str, body: bytes, response_model: type[ResponseT]) -> ResponseT:
        attempt = 0
        while True:
            attempt += 1
            started = self._clock.monotonic()
            try:
                result = self._send_once(route, body, response_model)
            except JobApiError as exc:
                self._log(route, attempt, started, exc.http_status, exc.code, type(exc).__name__)
                if not isinstance(exc, TransientError) or attempt > self._retry.max_retries:
                    raise
                self._clock.sleep(self._retry.delay_before_retry(attempt))
                continue
            self._log(route, attempt, started, 200, None, "ok")
            return result

    def _log(
        self,
        route: str,
        attempt: int,
        started: float,
        status: int | None,
        error_code: str | None,
        outcome: str,
    ) -> None:
        level = logging.INFO if outcome == "ok" else logging.WARNING
        logger.log(
            level,
            "job_api_call",
            extra={
                "route": route,
                "status": status if status is not None else "none",
                "error_code": error_code or "none",
                "outcome": outcome,
                "attempt": attempt,
                "duration_ms": round((self._clock.monotonic() - started) * 1000),
            },
        )

    def _send_once(self, route: str, body: bytes, response_model: type[ResponseT]) -> ResponseT:
        url = self._base_url + route
        try:
            headers = self._signer.headers_for(method="POST", url=url, body=body)
        except (NoCredentialsError, BotoCoreError) as exc:
            raise AuthError(
                f"Could not obtain AWS credentials to sign the request ({type(exc).__name__}).",
                route=route,
            ) from None
        try:
            response = self._http.post(url, content=body, headers=headers)
        except httpx.TimeoutException:
            raise TransportError("The request timed out.", route=route) from None
        except httpx.TransportError as exc:
            raise TransportError(
                f"The request failed to reach the Job API ({type(exc).__name__}).", route=route
            ) from None
        return self._interpret(route, response, response_model)

    def _interpret(
        self, route: str, response: httpx.Response, response_model: type[ResponseT]
    ) -> ResponseT:
        status = response.status_code
        payload = self._parse_json(response)

        if status == 200:
            if payload is None:
                raise ProtocolError(
                    f"HTTP 200 with a body that is not JSON: {safe_text(response.text)!r}",
                    route=route,
                    http_status=status,
                )
            try:
                return response_model.model_validate(payload)
            except ValidationError as exc:
                fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
                raise ProtocolError(
                    f"HTTP 200 body does not match the contract (fields: {', '.join(fields)}).",
                    route=route,
                    http_status=status,
                ) from None

        if status == 403:
            raise AuthError(AUTH_HINT, route=route, http_status=status)
        if status == 429:
            raise ThrottledError(
                "Throttled by the function's reserved concurrency (HTTP 429).",
                route=route,
                http_status=status,
            )
        envelope = self._parse_envelope(payload)
        if envelope is not None:
            raise self._error_from_envelope(route, status, envelope)
        if status >= 500:
            raise ServerError(
                f"Server error without the contract's error body (HTTP {status}).",
                route=route,
                http_status=status,
            )
        if payload is None:
            detail = f"body is not JSON: {safe_text(response.text)!r}"
        else:
            detail = "body is JSON but not the error envelope"
        raise ProtocolError(f"Unexpected HTTP {status}; {detail}", route=route, http_status=status)

    @staticmethod
    def _parse_json(response: httpx.Response) -> object | None:
        try:
            parsed: object = json.loads(response.content)
        except ValueError:
            return None
        return parsed

    @staticmethod
    def _parse_envelope(payload: object | None) -> ErrorEnvelope | None:
        if not isinstance(payload, dict) or payload.get("ok") is not False:
            return None
        try:
            return ErrorEnvelope.model_validate(payload)
        except ValidationError:
            return None

    @staticmethod
    def _error_from_envelope(route: str, status: int, envelope: ErrorEnvelope) -> JobApiError:
        code = safe_text(envelope.error.code)
        message = f"{code}: {safe_text(envelope.error.message)}"
        if status >= 500:
            return ServerError(message, route=route, http_status=status, code=code)
        try:
            error_class = CONTRACT_ERRORS[ErrorCode(envelope.error.code)]
        except (ValueError, KeyError):
            error_class = UnknownContractError
        return error_class(message, route=route, http_status=status, code=code)


def create_job_api_client(
    settings: Settings,
    *,
    clock: Clock,
    transport: httpx.BaseTransport | None = None,
    signer: RequestSigner | None = None,
) -> JobApiClient:
    if signer is None:
        if settings.job_api_signing == "none":
            signer = UnsignedSigner()
        else:
            region = settings.region
            if region is None:
                raise ValueError("region is required for sigv4 signing")
            signer = SigV4Signer(region=region)
    http = httpx.Client(
        timeout=httpx.Timeout(settings.job_api_timeout_seconds, connect=10.0),
        follow_redirects=False,
        transport=transport,
    )
    return JobApiClient(
        base_url=settings.job_api_url.get_secret_value(),
        agent_id=settings.agent_id,
        agent_version=settings.agent_version,
        http=http,
        signer=signer,
        clock=clock,
        retry=RetryPolicy(
            max_retries=settings.job_api_max_retries,
            base_delay=settings.job_api_retry_base_delay_seconds,
            max_delay=settings.job_api_retry_max_delay_seconds,
            jitter=settings.job_api_retry_jitter,
        ),
    )
