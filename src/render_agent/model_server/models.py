from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class ModelServerErrorCode(str, Enum):
    OOM = "oom"
    RENDER_ERROR = "render_error"
    INVALID_REQUEST = "invalid_request"
    BUSY = "busy"


class _Request(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, extra="forbid", frozen=True
    )


class _Response(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class GenerateParams(_Request):
    """Only the keys the job has; the server applies its own defaults for the rest."""

    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    fps: float | None = None
    seed: int | None = None


class GenerateRequest(_Request):
    job_id: int
    prompt: str = Field(repr=False)
    params: GenerateParams
    output_path: str
    # First-frame image for chained clips; omitted from the body when None.
    condition_image_path: str | None = None

    def to_json_bytes(self) -> bytes:
        return self.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")


class RenderReport(_Response):
    render_seconds: float = Field(ge=0)
    width: int
    height: int
    num_frames: int
    fps: float
    duration_seconds: float = Field(ge=0)
    has_audio: bool


class GenerateResponse(_Response):
    ok: Literal[True]
    output_path: str
    render: RenderReport


class HealthResponse(_Response):
    ok: Literal[True]
    model_loaded: bool


class ErrorBody(_Response):
    code: str
    message: str


class ErrorEnvelope(_Response):
    ok: Literal[False]
    error: ErrorBody
