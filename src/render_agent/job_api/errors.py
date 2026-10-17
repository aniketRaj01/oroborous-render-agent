from __future__ import annotations

from render_agent.job_api.models import ErrorCode
from render_agent.redaction import redact

_MAX_MESSAGE_CHARS = 200

AUTH_HINT = (
    "The request was rejected before reaching the Job API (HTTP 403). Check that AWS credentials "
    "are available, the role may invoke the function URL (lambda:InvokeFunctionUrl and "
    "lambda:InvokeFunction), JOB_API_REGION matches the function's region, and the system clock "
    "is correct."
)


def safe_text(text: str) -> str:
    """Single line, redacted and truncated: safe to put in an exception message or a log line."""
    one_line = " ".join(text.split())
    return redact(one_line)[:_MAX_MESSAGE_CHARS]


class JobApiError(Exception):
    def __init__(
        self,
        message: str,
        *,
        route: str,
        http_status: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.route = route
        self.http_status = http_status
        self.code = code


class TransientError(JobApiError):
    """A failure worth retrying later. Raised after the client's own retries are used up."""


class TransportError(TransientError):
    """Connection error or timeout."""


class ThrottledError(TransientError):
    """HTTP 429 from the function's reserved-concurrency limit."""


class ServerError(TransientError):
    """HTTP 5xx, with or without the error envelope (`internal`)."""


class AuthError(JobApiError):
    """HTTP 403 from AWS: signing or IAM problem. Never retried."""


class ProtocolError(JobApiError):
    """The response does not match the contract (not JSON, wrong shape)."""


class ContractError(JobApiError):
    """An error envelope with a 4xx status."""


class InvalidRequestError(ContractError):
    pass


class JobNotFoundError(ContractError):
    pass


class RouteNotFoundError(ContractError):
    pass


class MethodNotAllowedError(ContractError):
    pass


class NotYourLeaseError(ContractError):
    pass


class LeaseExpiredError(ContractError):
    pass


class WrongStateError(ContractError):
    pass


class MaxAttemptsReachedError(ContractError):
    pass


class KeyMismatchError(ContractError):
    pass


class ObjectMissingError(ContractError):
    pass


class SizeMismatchError(ContractError):
    pass


class UnknownContractError(ContractError):
    """A 4xx envelope whose code the contract does not list."""


CONTRACT_ERRORS: dict[ErrorCode, type[ContractError]] = {
    ErrorCode.INVALID_REQUEST: InvalidRequestError,
    ErrorCode.JOB_NOT_FOUND: JobNotFoundError,
    ErrorCode.NOT_FOUND: RouteNotFoundError,
    ErrorCode.METHOD_NOT_ALLOWED: MethodNotAllowedError,
    ErrorCode.NOT_YOUR_LEASE: NotYourLeaseError,
    ErrorCode.LEASE_EXPIRED: LeaseExpiredError,
    ErrorCode.WRONG_STATE: WrongStateError,
    ErrorCode.MAX_ATTEMPTS_REACHED: MaxAttemptsReachedError,
    ErrorCode.KEY_MISMATCH: KeyMismatchError,
    ErrorCode.OBJECT_MISSING: ObjectMissingError,
    ErrorCode.SIZE_MISMATCH: SizeMismatchError,
}
