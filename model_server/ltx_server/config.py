from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

LOOPBACK_HOSTS = ("127.0.0.1", "::1")
OFFLOAD_MODES = ("cpu", "disk", "none")
RENDERERS = ("ltx", "fake")
MAX_DIMENSION = 4096


class ConfigError(Exception):
    """Invalid configuration. Names the variables, never their values."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("Invalid configuration:\n" + "\n".join(f"  - {p}" for p in problems))


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    renderer: str = "ltx"
    ltx_root: Path = Path("~/LTX-2")
    models_subdir: str = "models/ltx-2.5"
    offload_mode: str = "cpu"
    quantization: str = "fp8-cast"
    use_registry: bool = True
    allowed_output_dir: Path = Path("/tmp/render-agent")
    default_width: int = 576
    default_height: int = 1024
    default_num_frames: int = 361
    default_fps: float = 24.0
    default_seed: int = 42
    max_frames: int = 481
    warmup: bool = True
    warmup_width: int = 256
    warmup_height: int = 256
    warmup_frames: int = 9
    max_consecutive_ooms: int = 2
    shutdown_grace_seconds: float = 30.0
    max_body_bytes: int = 262144

    @property
    def models_dir(self) -> Path:
        return self.ltx_root.expanduser() / self.models_subdir


def valid_dimension(value: int) -> bool:
    return 0 < value <= MAX_DIMENSION and value % 64 == 0


def valid_frame_count(value: int, maximum: int) -> bool:
    return 9 <= value <= maximum and (value - 1) % 8 == 0


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Read settings from the environment. Every problem is reported, none echoes a value."""
    env = os.environ if environ is None else environ
    problems: list[str] = []
    defaults = Settings()

    def text(name: str, default: str, allowed: tuple[str, ...] | None = None) -> str:
        value = env.get(name, "").strip() or default
        if allowed is not None and value not in allowed:
            problems.append(f"{name} must be one of: {', '.join(allowed)}")
            return default
        return value

    def number(name: str, default: float, low: float, high: float, *, integer: bool) -> float:
        raw = env.get(name, "").strip()
        if not raw:
            return default
        try:
            parsed = float(int(raw)) if integer else float(raw)
        except ValueError:
            problems.append(f"{name} must be {'an integer' if integer else 'a number'}")
            return default
        if not low <= parsed <= high or parsed != parsed:
            problems.append(f"{name} must be between {low:g} and {high:g}")
            return default
        return parsed

    def flag(name: str, default: bool) -> bool:
        raw = env.get(name, "").strip().lower()
        if not raw:
            return default
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        problems.append(f"{name} must be 1 or 0")
        return default

    def whole(name: str, default: int, low: int, high: int) -> int:
        return int(number(name, float(default), float(low), float(high), integer=True))

    host = text("HOST", defaults.host)
    if host not in LOOPBACK_HOSTS:
        problems.append(f"HOST must be a loopback address ({' or '.join(LOOPBACK_HOSTS)})")
        host = defaults.host
    port = whole("PORT", defaults.port, 0, 65535)
    renderer = text("RENDERER", defaults.renderer, RENDERERS)
    ltx_root = Path(text("LTX_ROOT", str(defaults.ltx_root)))
    models_subdir = text("MODELS_SUBDIR", defaults.models_subdir)
    if Path(models_subdir).is_absolute() or ".." in Path(models_subdir).parts:
        problems.append("MODELS_SUBDIR must be a relative path without '..'")
        models_subdir = defaults.models_subdir
    offload = text("OFFLOAD_MODE", defaults.offload_mode, OFFLOAD_MODES)
    quantization = text("QUANTIZATION", defaults.quantization)
    if not quantization.replace("-", "").isalnum():
        problems.append("QUANTIZATION must be a name such as fp8-cast, or 'none'")
        quantization = defaults.quantization
    allowed_dir = Path(text("ALLOWED_OUTPUT_DIR", str(defaults.allowed_output_dir)))
    if not allowed_dir.is_absolute():
        problems.append("ALLOWED_OUTPUT_DIR must be an absolute path")
        allowed_dir = defaults.allowed_output_dir

    max_frames = whole("MAX_FRAMES", defaults.max_frames, 9, 2001)
    width = whole("DEFAULT_WIDTH", defaults.default_width, 64, MAX_DIMENSION)
    height = whole("DEFAULT_HEIGHT", defaults.default_height, 64, MAX_DIMENSION)
    frames = whole("DEFAULT_NUM_FRAMES", defaults.default_num_frames, 9, 2001)
    fps = number("DEFAULT_FPS", defaults.default_fps, 0.001, 60, integer=False)
    seed = whole("DEFAULT_SEED", defaults.default_seed, 0, 2**63 - 1)
    warmup_width = whole("WARMUP_WIDTH", defaults.warmup_width, 64, MAX_DIMENSION)
    warmup_height = whole("WARMUP_HEIGHT", defaults.warmup_height, 64, MAX_DIMENSION)
    warmup_frames = whole("WARMUP_FRAMES", defaults.warmup_frames, 9, 2001)

    checks: list[tuple[str, bool]] = [
        ("DEFAULT_WIDTH must be a multiple of 64", valid_dimension(width)),
        ("DEFAULT_HEIGHT must be a multiple of 64", valid_dimension(height)),
        (
            "DEFAULT_NUM_FRAMES must be 8k+1 and at most MAX_FRAMES",
            valid_frame_count(frames, max_frames),
        ),
        ("WARMUP_WIDTH must be a multiple of 64", valid_dimension(warmup_width)),
        ("WARMUP_HEIGHT must be a multiple of 64", valid_dimension(warmup_height)),
        (
            "WARMUP_FRAMES must be 8k+1 and at most MAX_FRAMES",
            valid_frame_count(warmup_frames, max_frames),
        ),
    ]
    problems.extend(message for message, ok in checks if not ok and not _already(message, problems))

    settings = Settings(
        host=host,
        port=port,
        renderer=renderer,
        ltx_root=ltx_root,
        models_subdir=models_subdir,
        offload_mode=offload,
        quantization=quantization,
        use_registry=flag("USE_REGISTRY", defaults.use_registry),
        allowed_output_dir=allowed_dir,
        default_width=width,
        default_height=height,
        default_num_frames=frames,
        default_fps=fps,
        default_seed=seed,
        max_frames=max_frames,
        warmup=flag("WARMUP", defaults.warmup),
        warmup_width=warmup_width,
        warmup_height=warmup_height,
        warmup_frames=warmup_frames,
        max_consecutive_ooms=whole("MAX_CONSECUTIVE_OOMS", defaults.max_consecutive_ooms, 1, 100),
        shutdown_grace_seconds=number(
            "SHUTDOWN_GRACE_SECONDS", defaults.shutdown_grace_seconds, 0, 3600, integer=False
        ),
        max_body_bytes=whole("MAX_BODY_BYTES", defaults.max_body_bytes, 1024, 16 * 1024 * 1024),
    )
    if problems:
        raise ConfigError(problems)
    return settings


def _already(message: str, problems: list[str]) -> bool:
    """A default that failed its own range check was already reported by name."""
    name = message.split(" ", 1)[0]
    return any(p.startswith(name) for p in problems)
