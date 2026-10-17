from __future__ import annotations

import os
from pathlib import Path

import boto3
import pytest
from botocore.stub import Stubber
from mypy_boto3_ssm import SSMClient

from conftest import FAKE_AGENT, FAKE_URL
from render_agent.config import ConfigError
from render_agent.config_sources import fetch_ssm_parameters, load_settings

PREFIX = "/fake/prefix"


def write_file(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text)
    os.chmod(path, mode)
    return path


def no_ssm(prefix: str, region: str) -> dict[str, str]:
    raise AssertionError("SSM must not be called")


def test_environment_only(tmp_path: Path) -> None:
    loaded = load_settings(
        environ={"JOB_API_URL": FAKE_URL, "AGENT_ID": FAKE_AGENT}, cwd=tmp_path, ssm_fetch=no_ssm
    )
    assert loaded.settings.agent_id == FAKE_AGENT
    assert loaded.sources == ("env",)


def test_local_dotenv_is_read(tmp_path: Path) -> None:
    write_file(tmp_path / ".env", f"JOB_API_URL={FAKE_URL}\nAGENT_ID=from-dotenv\n")
    loaded = load_settings(environ={}, cwd=tmp_path, ssm_fetch=no_ssm)
    assert loaded.settings.agent_id == "from-dotenv"
    assert loaded.sources == ("dotenv",)


def test_config_file_is_read_and_beats_dotenv(tmp_path: Path) -> None:
    write_file(tmp_path / ".env", f"JOB_API_URL={FAKE_URL}\nAGENT_ID=from-dotenv\n", 0o644)
    config = write_file(tmp_path / "agent.env", "AGENT_ID=from-file\n")
    loaded = load_settings(
        environ={"RENDER_AGENT_CONFIG_FILE": str(config)}, cwd=tmp_path, ssm_fetch=no_ssm
    )
    assert loaded.settings.agent_id == "from-file"
    assert loaded.sources == ("config_file", "dotenv")


def test_environment_beats_file_beats_ssm(tmp_path: Path) -> None:
    config = write_file(tmp_path / "agent.env", "AGENT_ID=from-file\nJOB_API_REGION=eu-west-1\n")

    def ssm(prefix: str, region: str) -> dict[str, str]:
        assert (prefix, region) == (PREFIX, "us-east-1")
        return {
            "JOB_API_URL": FAKE_URL,
            "AGENT_ID": "from-ssm",
            "JOB_API_REGION": "ap-south-1",
            "AGENT_VERSION": "ssm-version",
        }

    loaded = load_settings(
        environ={
            "AGENT_ID": "from-env",
            "RENDER_AGENT_CONFIG_FILE": str(config),
            "RENDER_AGENT_SSM_PREFIX": PREFIX,
            "RENDER_AGENT_SSM_REGION": "us-east-1",
        },
        cwd=tmp_path,
        ssm_fetch=ssm,
    )
    settings = loaded.settings
    assert settings.agent_id == "from-env"
    assert settings.region == "eu-west-1"
    assert settings.agent_version == "ssm-version"
    assert settings.job_api_url.get_secret_value() == FAKE_URL
    assert loaded.sources == ("env", "config_file", "ssm")


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644, 0o660, 0o666])
def test_config_file_readable_by_group_or_others_fails(tmp_path: Path, mode: int) -> None:
    config = write_file(tmp_path / "agent.env", f"JOB_API_URL={FAKE_URL}\nAGENT_ID=a\n", mode)
    with pytest.raises(ConfigError) as caught:
        load_settings(environ={"RENDER_AGENT_CONFIG_FILE": str(config)}, cwd=tmp_path)
    assert caught.value.problems == (("RENDER_AGENT_CONFIG_FILE", "file_permissions_too_open"),)


def test_missing_config_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings(environ={"RENDER_AGENT_CONFIG_FILE": str(tmp_path / "nope")}, cwd=tmp_path)
    assert caught.value.problems == (("RENDER_AGENT_CONFIG_FILE", "file_unreadable"),)


def test_ssm_prefix_without_region_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings(environ={"RENDER_AGENT_SSM_PREFIX": PREFIX}, cwd=tmp_path, ssm_fetch=no_ssm)
    assert caught.value.problems == (("RENDER_AGENT_SSM_REGION", "missing"),)


def test_unrelated_environment_variables_are_ignored(tmp_path: Path) -> None:
    loaded = load_settings(
        environ={"JOB_API_URL": FAKE_URL, "AGENT_ID": FAKE_AGENT, "HOME": "/x", "PATH": "/bin"},
        cwd=tmp_path,
    )
    assert loaded.sources == ("env",)


def test_missing_values_are_reported_by_name_only(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings(environ={}, cwd=tmp_path)
    assert dict(caught.value.problems) == {"JOB_API_URL": "missing", "AGENT_ID": "missing"}


# --- SSM loader against botocore's Stubber -----------------------------------------------------


def stubbed_client() -> tuple[SSMClient, Stubber]:
    client = boto3.client(
        "ssm",
        region_name="us-east-1",
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
    )
    return client, Stubber(client)


def test_ssm_loader_maps_last_segment_and_ignores_unknown_names() -> None:
    client, stubber = stubbed_client()
    expected = {"Path": PREFIX, "Recursive": False, "WithDecryption": True}
    stubber.add_response(
        "get_parameters_by_path",
        {
            "Parameters": [
                {"Name": f"{PREFIX}/job-api-url", "Value": FAKE_URL, "Type": "SecureString"},
                {"Name": f"{PREFIX}/unrelated-thing", "Value": "x", "Type": "String"},
            ],
            "NextToken": "page2",
        },
        expected,
    )
    stubber.add_response(
        "get_parameters_by_path",
        {"Parameters": [{"Name": f"{PREFIX}/agent-id", "Value": "ssm-agent", "Type": "String"}]},
        {**expected, "NextToken": "page2"},
    )
    with stubber:
        values = fetch_ssm_parameters(PREFIX, "us-east-1", client=client)
    stubber.assert_no_pending_responses()
    assert values == {"JOB_API_URL": FAKE_URL, "AGENT_ID": "ssm-agent"}


def test_ssm_error_names_only_the_error_code() -> None:
    client, stubber = stubbed_client()
    stubber.add_client_error(
        "get_parameters_by_path",
        service_error_code="AccessDeniedException",
        service_message=f"not allowed to read {FAKE_URL}",
    )
    with stubber, pytest.raises(ConfigError) as caught:
        fetch_ssm_parameters(PREFIX, "us-east-1", client=client)
    assert caught.value.problems == (
        ("RENDER_AGENT_SSM_PREFIX", "ssm_read_failed_AccessDeniedException"),
    )
    assert FAKE_URL not in str(caught.value)
