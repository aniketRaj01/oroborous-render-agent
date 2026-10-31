from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    field_validator,
)
from pydantic.alias_generators import to_camel
from pydantic_core import PydanticCustomError

MAX_ERROR_MESSAGE_CHARS = 2000


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise PydanticCustomError("timezone_required", "timestamp must carry a timezone")
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


UtcDatetime = Annotated[
    datetime, AfterValidator(_require_utc), PlainSerializer(_format_utc, return_type=str)
]


class JobStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"


class FailErrorCode(str, Enum):
    RENDER_ERROR = "render_error"
    OOM = "oom"
    VERIFY_FAILED = "verify_failed"
    UPLOAD_ERROR = "upload_error"
    UNKNOWN = "unknown"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    JOB_NOT_FOUND = "job_not_found"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    NOT_YOUR_LEASE = "not_your_lease"
    LEASE_EXPIRED = "lease_expired"
    WRONG_STATE = "wrong_state"
    MAX_ATTEMPTS_REACHED = "max_attempts_reached"
    KEY_MISMATCH = "key_mismatch"
    OBJECT_MISSING = "object_missing"
    SIZE_MISMATCH = "size_mismatch"
    INTERNAL = "internal"


class _Request(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, extra="forbid", frozen=True
    )

    def to_json_bytes(self) -> bytes:
        return self.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")


class _Response(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


# --- claim -------------------------------------------------------------------------------------


class ClaimRequest(_Request):
    agent_id: str = Field(min_length=1, max_length=64)
    agent_version: str = Field(min_length=1, max_length=64)
    deadline_at: UtcDatetime | None = None
    max_plans: int | None = Field(default=None, ge=1, le=50)


class RenderParams(_Response):
    """Stored render parameters. The enqueue step decides which are present."""

    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    fps: float | None = None
    seed: int | None = None


class ClipUpload(_Response):
    bucket: str
    region: str
    key: str


class Clip(_Response):
    job_id: int
    position: int
    prompt: str = Field(repr=False)
    params: RenderParams | None
    upload: ClipUpload
    attempts: int
    resumed: bool
    # Chained clips only (absent or null otherwise): where to fetch this clip's start frame and
    # where to put its last frame. Same shape as `upload`.
    start_image: ClipUpload | None = None
    last_frame_upload: ClipUpload | None = None

    @property
    def is_chained(self) -> bool:
        return self.start_image is not None or self.last_frame_upload is not None


class Plan(_Response):
    plan_id: int
    account_id: int
    clips_total: int
    clips_already_done: int
    clips: list[Clip]


class ClaimTotals(_Response):
    plans: int
    clips: int


class ClaimResponse(_Response):
    ok: Literal[True]
    server_time: UtcDatetime
    lease_expires_at: UtcDatetime
    totals: ClaimTotals
    plans: list[Plan]


# --- start -------------------------------------------------------------------------------------


class StartRequest(_Request):
    agent_id: str = Field(min_length=1, max_length=64)
    job_id: int


class StartResponse(_Response):
    ok: Literal[True]
    job_id: int
    attempt: int
    max_attempts: int
    already_started: bool


# --- complete ----------------------------------------------------------------------------------


class CompleteUpload(_Request):
    bucket: str = Field(min_length=1)
    key: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)


class RenderInfo(_Request):
    render_seconds: float = Field(ge=0)
    width: int
    height: int
    num_frames: int
    fps: float
    duration_seconds: float = Field(ge=0)
    has_audio: bool


class CompleteRequest(_Request):
    agent_id: str = Field(min_length=1, max_length=64)
    job_id: int
    upload: CompleteUpload
    render: RenderInfo
    cost_inr: float | None = Field(default=None, ge=0)
    # Sent only when the claim gave this clip a lastFrameUpload.
    last_frame: CompleteUpload | None = None


class PlanProgress(_Response):
    id: int
    clips_done: int
    clips_total: int
    ready: bool


class CompleteResponse(_Response):
    ok: Literal[True]
    job_id: int
    status: JobStatus
    storage_key: str
    plan: PlanProgress


# --- fail --------------------------------------------------------------------------------------


class FailError(_Request):
    code: FailErrorCode
    message: str

    @field_validator("message")
    @classmethod
    def _truncate(cls, value: str) -> str:
        return value[:MAX_ERROR_MESSAGE_CHARS]


class FailRequest(_Request):
    agent_id: str = Field(min_length=1, max_length=64)
    job_id: int
    error: FailError


class FailResponse(_Response):
    ok: Literal[True]
    job_id: int
    status: JobStatus
    attempts_left: int
    will_retry: bool


# --- error envelope ----------------------------------------------------------------------------


class ErrorBody(_Response):
    code: str
    message: str
    details: JsonValue = None


class ErrorEnvelope(_Response):
    ok: Literal[False]
    error: ErrorBody
