from __future__ import annotations

import logging

import pytest

from conftest import FAKE_AGENT, FAKE_URL, valid_env
from render_agent.config import ConfigError, Settings, build_settings
from render_agent.logging_setup import configure_logging


def settings_from(**overrides: str) -> Settings:
    return build_settings({**valid_env(), **overrides})


def problems(values: dict[str, str]) -> dict[str, str]:
    with pytest.raises(ConfigError) as caught:
        build_settings(values)
    return dict(caught.value.problems)


def test_valid_minimal_settings_use_defaults() -> None:
    settings = settings_from()
    assert settings.agent_id == FAKE_AGENT
    assert settings.job_api_signing == "sigv4"
    assert settings.region == "us-east-1"
    assert settings.job_api_max_retries == 4
    assert settings.job_api_retry_base_delay_seconds == 0.5
    assert settings.job_api_retry_max_delay_seconds == 10.0
    assert settings.agent_version


def test_explicit_region_wins_over_derived() -> None:
    assert settings_from(JOB_API_REGION="eu-west-1").region == "eu-west-1"


def test_agent_id_is_required() -> None:
    assert problems({"JOB_API_URL": FAKE_URL}) == {"AGENT_ID": "missing"}


@pytest.mark.parametrize("agent_id", ["", "a b", "x" * 65, "ok\n", "semi;colon"])
def test_agent_id_pattern_is_enforced(agent_id: str) -> None:
    assert problems({**valid_env(), "AGENT_ID": agent_id}) == {
        "AGENT_ID": "string_pattern_mismatch"
    }


@pytest.mark.parametrize("agent_id", ["i-0123456789abcdef0", "a.b_c-d", "x" * 64])
def test_agent_id_accepts_allowed_characters(agent_id: str) -> None:
    assert settings_from(AGENT_ID=agent_id).agent_id == agent_id


def test_job_api_url_is_required() -> None:
    assert problems({"AGENT_ID": FAKE_AGENT}) == {"JOB_API_URL": "missing"}


@pytest.mark.parametrize(
    "url",
    [
        "http://fake-id.lambda-url.us-east-1.on.aws",
        "ftp://example.invalid",
        "example.invalid",
        "https://user:pw@example.invalid",
        "https://example.invalid/?q=1",
        "https://",
    ],
)
def test_bad_urls_are_rejected(url: str) -> None:
    assert "JOB_API_URL" in problems(
        {**valid_env(), "JOB_API_URL": url, "JOB_API_REGION": "us-east-1"}
    )


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8080", "http://localhost:9000", "http://[::1]:9000"]
)
def test_http_is_allowed_for_loopback_only(url: str) -> None:
    settings = settings_from(JOB_API_URL=url, JOB_API_SIGNING="none")
    assert settings.host_summary() == "loopback"


def test_http_on_loopback_with_sigv4_needs_an_explicit_region() -> None:
    assert problems({**valid_env(), "JOB_API_URL": "http://127.0.0.1:8080"}) == {
        "settings": "region_not_derivable"
    }
    assert settings_from(JOB_API_URL="http://127.0.0.1:8080", JOB_API_REGION="us-east-1")


def test_signing_none_is_rejected_for_non_loopback_hosts() -> None:
    assert problems({**valid_env(), "JOB_API_SIGNING": "none"}) == {
        "settings": "signing_none_requires_loopback"
    }


def test_unknown_signing_mode_is_rejected() -> None:
    assert problems({**valid_env(), "JOB_API_SIGNING": "basic"}) == {
        "JOB_API_SIGNING": "literal_error"
    }


def test_region_that_cannot_be_derived_fails_for_sigv4() -> None:
    values = {**valid_env(), "JOB_API_URL": "https://example.invalid"}
    assert problems(values) == {"settings": "region_not_derivable"}
    assert build_settings({**values, "JOB_API_REGION": "us-east-1"}).region == "us-east-1"


def test_invalid_region_name_is_rejected() -> None:
    assert problems({**valid_env(), "JOB_API_REGION": "nowhere"}) == {
        "JOB_API_REGION": "region_invalid"
    }


def test_retry_delays_must_be_ordered() -> None:
    values = {
        **valid_env(),
        "JOB_API_RETRY_BASE_DELAY_SECONDS": "5",
        "JOB_API_RETRY_MAX_DELAY_SECONDS": "1",
    }
    assert problems(values) == {"settings": "retry_delay_order"}


def test_trailing_slash_is_stripped() -> None:
    assert settings_from(JOB_API_URL=FAKE_URL + "/").job_api_url.get_secret_value() == FAKE_URL


def test_host_summary_masks_the_function_id() -> None:
    summary = settings_from().host_summary()
    assert summary == "<id>.lambda-url.us-east-1.on.aws"
    assert "fake-id" not in summary


SECRET_URL = "https://topsecret1.lambda-url.us-east-1.on.aws"


def test_url_never_appears_in_repr_str_or_dumps() -> None:
    settings = settings_from(JOB_API_URL=SECRET_URL)
    rendered = [
        repr(settings),
        str(settings),
        str(settings.model_dump()),
        settings.model_dump_json(),
        str(settings.model_dump(mode="json")),
    ]
    for text in rendered:
        assert "topsecret1" not in text
        assert SECRET_URL not in text


@pytest.mark.parametrize(
    "overrides",
    [
        {"JOB_API_URL": "http://topsecret1.lambda-url.us-east-1.on.aws"},
        {"JOB_API_URL": "https://topsecret1:pw@example.invalid"},
        {"JOB_API_URL": "ftp://topsecret1.example.invalid"},
        {"JOB_API_SIGNING": "topsecret1"},
        {"JOB_API_REGION": "topsecret1"},
        {"AGENT_ID": "topsecret1 !"},
        {"JOB_API_MAX_RETRIES": "topsecret1"},
        {"JOB_API_TIMEOUT_SECONDS": "topsecret1"},
    ],
)
def test_validation_errors_never_echo_values(overrides: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as caught:
        build_settings({**valid_env(), **overrides})
    assert "topsecret1" not in str(caught.value)
    assert "topsecret1" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_url_never_appears_in_a_log_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(logging.INFO)
    settings = settings_from(JOB_API_URL=SECRET_URL)
    logging.getLogger("render_agent.test").info("settings loaded: %s", settings)
    logging.getLogger("render_agent.test").info("url leak attempt %s", SECRET_URL)
    err = capsys.readouterr().err
    assert "topsecret1" not in err
    assert "msg=" in err
