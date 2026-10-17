from __future__ import annotations

import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import dotenv_values

from render_agent.config import ConfigError, Settings, build_settings

if TYPE_CHECKING:
    from mypy_boto3_ssm import SSMClient

CONFIG_FILE_VAR = "RENDER_AGENT_CONFIG_FILE"
SSM_PREFIX_VAR = "RENDER_AGENT_SSM_PREFIX"
SSM_REGION_VAR = "RENDER_AGENT_SSM_REGION"

SETTING_NAMES = frozenset(
    field.validation_alias
    for field in Settings.model_fields.values()
    if isinstance(field.validation_alias, str)
)

SsmFetch = Callable[[str, str], dict[str, str]]


@dataclass(frozen=True)
class LoadedSettings:
    settings: Settings
    sources: tuple[str, ...]


def fetch_ssm_parameters(
    prefix: str, region: str, client: SSMClient | None = None
) -> dict[str, str]:
    """Read parameters directly under `prefix`; the last path segment names the setting."""
    if client is None:
        import boto3

        client = boto3.client("ssm", region_name=region)
    found: dict[str, str] = {}
    try:
        pages = client.get_paginator("get_parameters_by_path").paginate(
            Path=prefix, Recursive=False, WithDecryption=True
        )
        for page in pages:
            for parameter in page.get("Parameters", []):
                name = parameter.get("Name", "").rsplit("/", 1)[-1].upper().replace("-", "_")
                value = parameter.get("Value")
                if name in SETTING_NAMES and value is not None:
                    found[name] = value
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "unknown")
        raise ConfigError([("RENDER_AGENT_SSM_PREFIX", f"ssm_read_failed_{code}")]) from None
    except BotoCoreError as exc:
        raise ConfigError(
            [("RENDER_AGENT_SSM_PREFIX", f"ssm_read_failed_{type(exc).__name__}")]
        ) from None
    return found


def _read_config_file(path: Path) -> dict[str, str]:
    try:
        mode = path.stat().st_mode
    except OSError:
        raise ConfigError([(CONFIG_FILE_VAR, "file_unreadable")]) from None
    if not stat.S_ISREG(mode):
        raise ConfigError([(CONFIG_FILE_VAR, "file_not_regular")])
    if mode & 0o077:
        raise ConfigError([(CONFIG_FILE_VAR, "file_permissions_too_open")])
    return _clean(dotenv_values(path))


def _clean(values: Mapping[str, str | None]) -> dict[str, str]:
    return {key: value for key, value in values.items() if value is not None}


def _only_settings(values: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in values.items() if key in SETTING_NAMES}


def load_settings(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    ssm_fetch: SsmFetch = fetch_ssm_parameters,
) -> LoadedSettings:
    """Precedence, highest first: process env, config file, local .env, SSM Parameter Store."""
    env = dict(os.environ if environ is None else environ)
    dotenv_path = (cwd or Path.cwd()) / ".env"
    dotenv = _clean(dotenv_values(dotenv_path)) if dotenv_path.is_file() else {}

    config_file = env.get(CONFIG_FILE_VAR) or dotenv.get(CONFIG_FILE_VAR)
    file_values = _read_config_file(Path(config_file)) if config_file else {}

    def control(name: str) -> str | None:
        return env.get(name) or file_values.get(name) or dotenv.get(name)

    ssm_values: dict[str, str] = {}
    prefix = control(SSM_PREFIX_VAR)
    if prefix:
        region = control(SSM_REGION_VAR)
        if not region:
            raise ConfigError([(SSM_REGION_VAR, "missing")])
        ssm_values = _only_settings(ssm_fetch(prefix, region))

    layers = (
        ("ssm", ssm_values),
        ("dotenv", _only_settings(dotenv)),
        ("config_file", _only_settings(file_values)),
        ("env", _only_settings(env)),
    )
    merged: dict[str, str] = {}
    used: list[str] = []
    for name, values in layers:
        if values:
            used.append(name)
            merged.update(values)
    used.reverse()
    return LoadedSettings(settings=build_settings(dict(merged)), sources=tuple(used))
