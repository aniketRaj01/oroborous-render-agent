"""RENDER_MODE=resident against stub torch and LTX objects: what is built when, and in what order
the models are moved. Correctness of the pictures is proven on the instance, not here."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from ltx_server.config import ConfigError, load_settings
from ltx_server.ltx_binding import LtxBinding
from ltx_server.renderer import RenderRequest
from ltx_server.resident_binding import ResidentModeError
from ltx_server.resident_renderer import ResidentLtxRenderer
from ltx_stubs import install
from ms_helpers import make_settings

REQUEST = RenderRequest(7, "a prompt", 576, 1024, 361, 24.0, 42)


class Data:
    def __init__(self, owner: str, device: str, log: list[str]) -> None:
        self.owner, self.device, self.log = owner, device, log

    def to(self, device: Any, non_blocking: bool = False) -> Data:
        self.log.append(f"copy {self.owner} -> {device}")
        return Data(self.owner, str(device), self.log)

    def pin_memory(self) -> Data:
        return self


class Param:
    def __init__(self, owner: str, device: str, log: list[str]) -> None:
        self._data = Data(owner, device, log)
        self._log = log

    @property
    def data(self) -> Data:
        return self._data

    @data.setter
    def data(self, value: Data) -> None:
        self._data = value
        self._log.append(f"{value.owner} now on {value.device}")


class Module:
    def __init__(self, name: str, device: str, log: list[str]) -> None:
        self.name, self.log = name, log
        self._param = Param(name, device, log)
        log.append(f"build {name}")

    def parameters(self) -> list[Param]:
        return [self._param]

    def buffers(self) -> list[Param]:
        return []

    def eval(self) -> Module:
        return self

    def encode(self, prompts: list[str]) -> list[tuple[str, str]]:
        self.log.append("encode text")
        return [("hidden", "mask")]

    def process_hidden_states(self, hidden: str, mask: str) -> str:
        self.log.append("embeddings")
        return "context"

    def decode_video(self, *args: Any, **kwargs: Any) -> Iterator[str]:
        self.log.append("decode video")
        yield "frames"


class Builder:
    def __init__(self, name: str, device: str, log: list[str]) -> None:
        self.name, self.device, self.log = name, device, log

    def build(self, device: Any = None, dtype: Any = None) -> Module:
        return Module(self.name, str(device), self.log)

    def with_module_ops(self, ops: object) -> Builder:
        return self


class Latent:
    def to(self, dtype: Any = None) -> Latent:
        return self


class StubPipeline:
    """Calls its blocks in the order of LTX's DistilledPipeline.__call__ (read from its source)."""

    def __init__(self, log: list[str], stage_calls: int = 2) -> None:
        self.device = "cuda"
        self.stage_calls = stage_calls
        encoder = types.SimpleNamespace(
            _text_encoder_builder=Builder("gemma", "cpu", log),
            _build_embeddings_processor=lambda: Module("embeddings", "cuda", log),
        )
        self.prompt_encoder: Any = encoder
        self.stage: Any = types.SimpleNamespace(
            _build_transformer=lambda: Module("transformer", "cuda", log)
        )
        self.upsampler: Any = types.SimpleNamespace(
            _encoder_builder=Builder("video_encoder", "cuda", log),
            _upsampler_builder=Builder("upsampler", "cuda", log),
        )
        self.video_decoder: Any = types.SimpleNamespace(
            _prepared_builder=lambda: Builder("video_decoder", "cuda", log), checkpoint_path="vae"
        )
        self.audio_decoder: Any = types.SimpleNamespace(
            _decoder_builder=Builder("audio_decoder", "cuda", log),
            _vocoder_builder=Builder("vocoder", "cuda", log),
        )
        self.log = log

    def __call__(self, **kwargs: Any) -> Any:
        self.prompt_encoder([kwargs["prompt"]], enhance_first_prompt=False)
        for index in range(self.stage_calls):
            with self.stage._transformer_ctx(video_tools=None):
                self.log.append(f"denoise {index + 1}")
            if index == 0:
                self.upsampler(Latent())
        list(self.video_decoder(Latent(), None, None, dtype=None))
        self.audio_decoder(Latent())
        return types.SimpleNamespace(
            video=iter(["frames"]), audio=object(), num_frames=9, tiling_config=None
        )


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    log: list[str] = []
    install(monkeypatch)
    cuda = sys.modules["torch"].cuda
    cuda.synchronize = lambda: None
    cuda.memory_allocated = cuda.memory_reserved = cuda.max_memory_allocated = lambda: 0
    cuda.reset_peak_memory_stats = lambda: None

    @contextmanager
    def inference_mode(mode: bool = True) -> Iterator[None]:
        yield

    torch: Any = sys.modules["torch"]
    torch.inference_mode, torch.bfloat16, torch.device = inference_mode, "bf16", str

    def module(name: str, **attributes: Any) -> None:
        created = types.ModuleType(name)
        created.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, created)

    def upsample_video(latent: Latent, video_encoder: Any, upsampler: Any) -> Latent:
        log.append("upsample")
        return latent

    def decode_audio(latent: Any, decoder: Any, vocoder: Any) -> str:
        log.append("decode audio")
        return "audio"

    module("ltx_core.model.upsampler", upsample_video=upsample_video)
    module("ltx_core.model.audio_vae", decode_audio=decode_audio)
    pipelines: list[StubPipeline] = []

    def build(self: LtxBinding) -> StubPipeline:
        pipelines.append(StubPipeline(log, stage_calls=3 if "three" in log else 2))
        return pipelines[-1]

    monkeypatch.setattr(LtxBinding, "build_pipeline", build)
    return log


def renderer(tmp_path: Path, **overrides: Any) -> ResidentLtxRenderer:
    return ResidentLtxRenderer(make_settings(tmp_path, render_mode="resident", **overrides))


def test_settings_for_the_render_mode() -> None:
    defaults = load_settings({})
    assert (defaults.render_mode, defaults.offload_mode, defaults.use_registry) == (
        "pipeline",
        "none",
        False,
    )
    assert defaults.park_transformer_for_decode is True
    chosen = load_settings({"RENDER_MODE": "resident", "PARK_TRANSFORMER_FOR_DECODE": "0"})
    assert chosen.render_mode == "resident" and chosen.park_transformer_for_decode is False
    with pytest.raises(ConfigError, match="RENDER_MODE") as error:
        load_settings({"RENDER_MODE": "sneaky-value"})
    assert "sneaky-value" not in str(error.value)


def test_each_model_is_built_once_across_load_and_two_renders(
    stubs: list[str], tmp_path: Path
) -> None:
    r = renderer(tmp_path)
    r.load()
    for _ in range(2):
        r.render(REQUEST, tmp_path / "out.mp4")
    built = [line for line in stubs if line.startswith("build ")]
    assert sorted(built) == sorted(
        f"build {name}"
        for name in (
            "gemma",
            "transformer",
            "embeddings",
            "video_encoder",
            "upsampler",
            "video_decoder",
            "audio_decoder",
            "vocoder",
        )
    )
    assert stubs.count("denoise 1") == 2 and stubs.count("denoise 2") == 2


@pytest.mark.parametrize("park", [True, False])
def test_phase_order_with_the_gemma_swap_and_the_decode_park(
    stubs: list[str], tmp_path: Path, park: bool
) -> None:
    r = renderer(tmp_path, park_transformer_for_decode=park)
    r.load()
    stubs.clear()
    result = r.render(REQUEST, tmp_path / "out.mp4")
    transformer_out = "transformer now on cpu"
    transformer_back = "transformer now on cuda"
    expected = [
        transformer_out,
        "gemma now on cuda",
        "encode text",
        "gemma now on cpu",
        "embeddings",
        transformer_back,
        "denoise 1",
        "upsample",
        "denoise 2",
        *([transformer_out] if park else []),
        "decode video",
        "decode audio",
    ]
    assert [line for line in stubs if not line.startswith("copy ")] == expected
    assert list(result.phase_seconds)[-3:] == ["decode_video", "decode_audio", "write_mp4"]
    assert {"encode_text", "stage_1", "upsample", "stage_2"} <= set(result.phase_seconds)


def test_a_multi_chunk_plan_names_the_fallback(stubs: list[str], tmp_path: Path) -> None:
    stubs.append("three")
    r = renderer(tmp_path)
    r.load()
    with pytest.raises(ResidentModeError, match="RENDER_MODE=pipeline"):
        r.render(REQUEST, tmp_path / "out.mp4")
