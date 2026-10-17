from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from render_agent import package_version

_FUNCTION_URL_HOST = re.compile(
    r"^(?P<id>[a-z0-9-]+)\.lambda-url\.(?P<region>[a-z0-9-]+)\.on\.aws$"
)
_LOOPBACK_NAMES = frozenset({"localhost"})


class ConfigError(Exception):
    """Invalid configuration. Carries setting names and error types only, never values."""

    def __init__(self, problems: list[tuple[str, str]]) -> None:
        self.problems = tuple(problems)
        lines = [f"  - {name}: {kind}" for name, kind in self.problems]
        super().__init__("Invalid configuration:\n" + "\n".join(lines))


def is_loopback_host(host: str) -> bool:
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def derive_region(host: str) -> str | None:
    match = _FUNCTION_URL_HOST.match(host.lower())
    return match["region"] if match else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
        populate_by_name=True,
    )

    job_api_url: SecretStr = Field(validation_alias="JOB_API_URL")
    job_api_signing: Literal["sigv4", "none"] = Field(
        default="sigv4", validation_alias="JOB_API_SIGNING"
    )
    job_api_region: str | None = Field(default=None, validation_alias="JOB_API_REGION")
    agent_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$", validation_alias="AGENT_ID")
    agent_version: str = Field(
        default_factory=package_version,
        min_length=1,
        max_length=64,
        validation_alias="AGENT_VERSION",
    )
    job_api_timeout_seconds: float = Field(
        default=35.0, gt=0, le=300, validation_alias="JOB_API_TIMEOUT_SECONDS"
    )
    job_api_max_retries: int = Field(default=4, ge=0, le=20, validation_alias="JOB_API_MAX_RETRIES")
    job_api_retry_base_delay_seconds: float = Field(
        default=0.5, ge=0, validation_alias="JOB_API_RETRY_BASE_DELAY_SECONDS"
    )
    job_api_retry_max_delay_seconds: float = Field(
        default=10.0, ge=0, validation_alias="JOB_API_RETRY_MAX_DELAY_SECONDS"
    )
    job_api_retry_jitter: float = Field(
        default=0.5, ge=0, le=1, validation_alias="JOB_API_RETRY_JITTER"
    )

    model_server_url: str = Field(
        default="http://127.0.0.1:8000", validation_alias="MODEL_SERVER_URL"
    )
    model_server_health_poll_seconds: float = Field(
        default=5.0, gt=0, validation_alias="MODEL_SERVER_HEALTH_POLL_SECONDS"
    )
    model_server_ready_timeout_minutes: float = Field(
        default=20.0, gt=0, validation_alias="MODEL_SERVER_READY_TIMEOUT_MINUTES"
    )
    model_server_generate_timeout_seconds: float = Field(
        default=1200.0, gt=0, validation_alias="MODEL_SERVER_GENERATE_TIMEOUT_SECONDS"
    )
    scratch_dir: Path = Field(
        default=Path("/tmp/render-agent"),  # noqa: S108 - default location, created 0700
        validation_alias="SCRATCH_DIR",
    )
    max_plans_per_claim: int = Field(
        default=10, ge=1, le=50, validation_alias="MAX_PLANS_PER_CLAIM"
    )
    empty_claim_wait_seconds: float = Field(
        default=60.0, gt=0, validation_alias="EMPTY_CLAIM_WAIT_SECONDS"
    )
    idle_exit_minutes: float = Field(default=5.0, gt=0, validation_alias="IDLE_EXIT_MINUTES")
    # The Job API ignores a deadline more than 180 minutes away, so stay well inside that.
    run_deadline_minutes: float = Field(
        default=55.0, gt=0, le=170, validation_alias="RUN_DEADLINE_MINUTES"
    )
    clip_time_estimate_seconds: float = Field(
        default=300.0, gt=0, validation_alias="CLIP_TIME_ESTIMATE_SECONDS"
    )
    max_consecutive_failures: int = Field(
        default=3, ge=1, validation_alias="MAX_CONSECUTIVE_FAILURES"
    )
    complete_retry_budget_minutes: float = Field(
        default=10.0, gt=0, validation_alias="COMPLETE_RETRY_BUDGET_MINUTES"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Sources and precedence are resolved explicitly in config_sources.load_settings.
        return (init_settings,)

    @field_validator("job_api_url")
    @classmethod
    def _check_url(cls, value: SecretStr) -> SecretStr:
        try:
            parts = urlsplit(value.get_secret_value())
            host = parts.hostname
            parts.port  # noqa: B018 - raises ValueError for a malformed port
        except ValueError:
            raise PydanticCustomError("url_invalid", "not a valid URL") from None
        if not host:
            raise PydanticCustomError("url_invalid", "not a valid URL")
        if parts.username or parts.password:
            raise PydanticCustomError("url_has_credentials", "URL must not contain credentials")
        if parts.query or parts.fragment:
            raise PydanticCustomError("url_has_query", "URL must not have a query or fragment")
        if parts.scheme == "https" or (parts.scheme == "http" and is_loopback_host(host)):
            pass
        else:
            raise PydanticCustomError("url_scheme_not_allowed", "https required")
        return SecretStr(value.get_secret_value().rstrip("/"))

    @field_validator("model_server_url")
    @classmethod
    def _check_model_server_url(cls, value: str) -> str:
        try:
            parts = urlsplit(value)
            host = parts.hostname
            parts.port  # noqa: B018 - raises ValueError for a malformed port
        except ValueError:
            raise PydanticCustomError("url_invalid", "not a valid URL") from None
        if parts.scheme not in ("http", "https") or not host:
            raise PydanticCustomError("url_invalid", "not a valid URL")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise PydanticCustomError("url_invalid", "URL must be plain host and port")
        if not is_loopback_host(host):
            raise PydanticCustomError(
                "model_server_not_loopback", "the model server must be on a loopback host"
            )
        return value.rstrip("/")

    @field_validator("scratch_dir")
    @classmethod
    def _check_scratch_dir(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise PydanticCustomError("path_not_absolute", "SCRATCH_DIR must be absolute")
        return value

    @field_validator("job_api_region")
    @classmethod
    def _check_region(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[a-z]{2}(-[a-z]+)+-\d", value):
            raise PydanticCustomError("region_invalid", "not an AWS region name")
        return value

    @model_validator(mode="after")
    def _check_signing_and_region(self) -> Settings:
        host = self.job_api_host
        if self.job_api_signing == "none":
            if not is_loopback_host(host):
                raise PydanticCustomError(
                    "signing_none_requires_loopback",
                    "JOB_API_SIGNING=none is only allowed for a loopback JOB_API_URL",
                )
        elif self.region is None:
            raise PydanticCustomError(
                "region_not_derivable",
                "JOB_API_REGION is unset and cannot be derived from JOB_API_URL",
            )
        if self.job_api_retry_max_delay_seconds < self.job_api_retry_base_delay_seconds:
            raise PydanticCustomError(
                "retry_delay_order",
                "JOB_API_RETRY_MAX_DELAY_SECONDS must be >= JOB_API_RETRY_BASE_DELAY_SECONDS",
            )
        return self

    @property
    def job_api_host(self) -> str:
        return urlsplit(self.job_api_url.get_secret_value()).hostname or ""

    @property
    def region(self) -> str | None:
        return self.job_api_region or derive_region(self.job_api_host)

    def host_summary(self) -> str:
        host = self.job_api_host
        if is_loopback_host(host):
            return "loopback"
        derived = derive_region(host)
        if derived is not None:
            return f"<id>.lambda-url.{derived}.on.aws"
        return "<custom host>"


def build_settings(values: dict[str, Any]) -> Settings:
    try:
        return Settings(**values)
    except ValidationError as exc:
        problems = [(_setting_name(error["loc"]), error["type"]) for error in exc.errors()]
        raise ConfigError(problems) from None


def _setting_name(loc: tuple[int | str, ...]) -> str:
    return str(loc[0]) if loc else "settings"
