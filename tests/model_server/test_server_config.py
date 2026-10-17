from __future__ import annotations

from pathlib import Path

import pytest

from ltx_server.config import ConfigError, Settings, load_settings


def test_defaults_match_the_documented_ones() -> None:
    settings = load_settings({})
    assert settings == Settings()
    assert (settings.host, settings.port, settings.renderer) == ("127.0.0.1", 8000, "ltx")
    assert settings.offload_mode == "cpu" and settings.quantization == "fp8-cast"
    assert settings.models_subdir == "models/ltx-2.5" and settings.ltx_root == Path("~/LTX-2")
    assert str(settings.allowed_output_dir) == "/tmp/render-agent"
    assert (settings.default_width, settings.default_height, settings.default_num_frames) == (
        576,
        1024,
        361,
    )
    assert (settings.default_fps, settings.default_seed, settings.max_frames) == (24.0, 42, 481)
    assert (
        settings.warmup,
        settings.warmup_width,
        settings.warmup_height,
        settings.warmup_frames,
    ) == (
        True,
        256,
        256,
        9,
    )
    assert settings.max_consecutive_ooms == 2 and settings.shutdown_grace_seconds == 30.0


def test_every_variable_is_read() -> None:
    settings = load_settings(
        {
            "HOST": "::1",
            "PORT": "9000",
            "RENDERER": "fake",
            "LTX_ROOT": "/opt/ltx",
            "MODELS_SUBDIR": "models/other",
            "OFFLOAD_MODE": "disk",
            "QUANTIZATION": "fp8-scaled-mm",
            "USE_REGISTRY": "0",
            "ALLOWED_OUTPUT_DIR": "/var/scratch",
            "DEFAULT_WIDTH": "640",
            "DEFAULT_HEIGHT": "384",
            "DEFAULT_NUM_FRAMES": "121",
            "DEFAULT_FPS": "30",
            "DEFAULT_SEED": "7",
            "MAX_FRAMES": "241",
            "WARMUP": "0",
            "WARMUP_WIDTH": "128",
            "WARMUP_HEIGHT": "192",
            "WARMUP_FRAMES": "17",
            "MAX_CONSECUTIVE_OOMS": "3",
            "SHUTDOWN_GRACE_SECONDS": "12.5",
            "MAX_BODY_BYTES": "4096",
        }
    )
    assert settings.host == "::1" and settings.port == 9000 and settings.renderer == "fake"
    assert settings.models_dir == Path("/opt/ltx/models/other")
    assert settings.offload_mode == "disk" and settings.quantization == "fp8-scaled-mm"
    assert settings.use_registry is False and settings.warmup is False
    assert (settings.default_width, settings.default_height, settings.default_num_frames) == (
        640,
        384,
        121,
    )
    assert settings.default_fps == 30 and settings.default_seed == 7 and settings.max_frames == 241
    assert (settings.warmup_width, settings.warmup_height, settings.warmup_frames) == (128, 192, 17)
    assert settings.max_consecutive_ooms == 3 and settings.shutdown_grace_seconds == 12.5
    assert settings.max_body_bytes == 4096


def test_the_home_directory_in_ltx_root_is_expanded_late() -> None:
    assert not str(load_settings({}).models_dir).startswith("~")


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("HOST", "0.0.0.0"),
        ("HOST", "localhost"),
        ("PORT", "-1"),
        ("PORT", "70000"),
        ("PORT", "eighty"),
        ("RENDERER", "cuda"),
        ("OFFLOAD_MODE", "ram"),
        ("QUANTIZATION", "fp8 cast; rm"),
        ("ALLOWED_OUTPUT_DIR", "relative/dir"),
        ("MODELS_SUBDIR", "/abs"),
        ("MODELS_SUBDIR", "../up"),
        ("DEFAULT_WIDTH", "100"),
        ("DEFAULT_HEIGHT", "0"),
        ("DEFAULT_NUM_FRAMES", "360"),
        ("DEFAULT_NUM_FRAMES", "489"),
        ("DEFAULT_FPS", "0"),
        ("DEFAULT_FPS", "75"),
        ("DEFAULT_FPS", "fast"),
        ("DEFAULT_SEED", "-1"),
        ("MAX_FRAMES", "4"),
        ("WARMUP", "maybe"),
        ("WARMUP_WIDTH", "100"),
        ("WARMUP_FRAMES", "10"),
        ("MAX_CONSECUTIVE_OOMS", "0"),
        ("SHUTDOWN_GRACE_SECONDS", "-1"),
        ("SHUTDOWN_GRACE_SECONDS", "nan"),
        ("MAX_BODY_BYTES", "10"),
    ],
)
def test_invalid_values_are_rejected_naming_the_variable_not_the_value(
    variable: str, value: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings({variable: value})
    message = str(caught.value)
    assert variable in message
    assert value not in message.replace(variable, "") or value in {"0", "-1", "10", "4", "100"}


def test_secret_looking_values_are_never_echoed() -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings(
            {"PORT": "hunter2-secret-token", "OFFLOAD_MODE": "/home/user/.aws/credentials"}
        )
    assert "hunter2" not in str(caught.value) and ".aws" not in str(caught.value)
    assert "PORT" in str(caught.value) and "OFFLOAD_MODE" in str(caught.value)


def test_all_problems_are_reported_together() -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings({"PORT": "x", "HOST": "0.0.0.0", "RENDERER": "y"})
    assert len(caught.value.problems) == 3


def test_defaults_that_break_the_request_rules_are_refused() -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings({"DEFAULT_NUM_FRAMES": "481", "MAX_FRAMES": "241"})
    assert "DEFAULT_NUM_FRAMES" in str(caught.value)


@pytest.mark.parametrize("raw", ["1", "true", "YES", "on"])
def test_true_flags(raw: str) -> None:
    assert load_settings({"WARMUP": raw}).warmup is True


@pytest.mark.parametrize("raw", ["0", "false", "No", "off"])
def test_false_flags(raw: str) -> None:
    assert load_settings({"WARMUP": raw}).warmup is False
