"""LtxRenderer with an injected fake binding, and LtxBinding against stub torch and ltx modules."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ltx_server.config import Settings
from ltx_server.ltx_binding import LtxBinding, encode_fps
from ltx_server.ltx_renderer import WARMUP_PROMPT, LtxRenderer
from ltx_server.renderer import RendererOutOfMemory, RenderRequest
from ltx_server.server import ServerApp
from ltx_stubs import (
    OffloadMode,
    StubOutOfMemoryError,
    StubRegistry,
    install,
    make_model_files,
)
from ms_helpers import generate_body, make_settings, request, wait_until

REQUEST = RenderRequest(7, "a prompt", 576, 1024, 361, 24.0, 42)


@dataclass
class FakeBinding:
    oom_types: tuple[type[BaseException], ...] = (StubOutOfMemoryError,)
    events: list[str] = field(default_factory=list)
    pipelines: list[object] = field(default_factory=list)
    generated: list[tuple[object, RenderRequest, Path]] = field(default_factory=list)
    warmup_path_existed: list[bool] = field(default_factory=list)
    fail: BaseException | None = None
    audio: bool = True

    def prepare_environment(self) -> None:
        self.events.append("prepare")

    def build_pipeline(self) -> Any:
        self.events.append("build")
        pipeline = object()
        self.pipelines.append(pipeline)
        return pipeline

    def generate(self, pipeline: Any, request: RenderRequest, output_path: Path) -> bool:
        self.events.append("generate")
        self.generated.append((pipeline, request, output_path))
        self.warmup_path_existed.append(output_path.parent.exists())
        if self.fail is not None:
            raise self.fail
        return self.audio

    def free_gpu_cache(self) -> None:
        self.events.append("free")

    def is_out_of_memory(self, exc: BaseException) -> bool:
        return isinstance(exc, self.oom_types)


def renderer_with(binding: FakeBinding, **overrides: Any) -> LtxRenderer:
    return LtxRenderer(Settings(**overrides), binding=binding)


# --- LtxRenderer with a fake binding --------------------------------------------------------------


def test_the_pipeline_is_built_once_and_after_the_environment_is_prepared() -> None:
    binding = FakeBinding()
    renderer = renderer_with(binding)
    renderer.load()
    assert binding.events == ["prepare", "build"]
    with pytest.raises(RuntimeError, match="built once"):
        renderer.load()
    assert len(binding.pipelines) == 1


def test_every_request_reuses_the_same_pipeline_object(tmp_path: Path) -> None:
    binding = FakeBinding()
    renderer = renderer_with(binding, warmup=True)
    renderer.load()
    renderer.warm_up()
    for index in range(3):
        renderer.render(REQUEST, tmp_path / f"{index}.mp4")
    assert len(binding.pipelines) == 1
    assert len(binding.generated) == 4  # warm-up plus three clips
    assert all(pipeline is binding.pipelines[0] for pipeline, _, _ in binding.generated)


def test_warm_up_uses_the_warmup_size_and_a_throwaway_path_that_is_deleted() -> None:
    binding = FakeBinding()
    renderer = renderer_with(binding, warmup_width=192, warmup_height=128, warmup_frames=17)
    renderer.load()
    renderer.warm_up()
    _, warm_request, path = binding.generated[0]
    assert (warm_request.width, warm_request.height, warm_request.num_frames) == (192, 128, 17)
    assert warm_request.prompt == WARMUP_PROMPT and warm_request.job_id == 0
    assert binding.warmup_path_existed == [True]  # the directory existed during the generation
    assert not path.parent.exists() and not path.exists()  # and is gone afterwards
    assert binding.events[-2:] == ["generate", "free"]


def test_warm_up_can_be_disabled() -> None:
    binding = FakeBinding()
    renderer = renderer_with(binding, warmup=False)
    renderer.load()
    renderer.warm_up()
    assert binding.generated == []


def test_render_maps_out_of_memory_and_lets_other_errors_through(tmp_path: Path) -> None:
    binding = FakeBinding(fail=StubOutOfMemoryError("cuda oom"))
    renderer = renderer_with(binding)
    renderer.load()
    with pytest.raises(RendererOutOfMemory):
        renderer.render(REQUEST, tmp_path / "a.mp4")
    binding.fail = ValueError("bad")
    with pytest.raises(ValueError, match="bad"):
        renderer.render(REQUEST, tmp_path / "a.mp4")


def test_render_result_reflects_the_request_and_the_audio(tmp_path: Path) -> None:
    binding = FakeBinding(audio=False)
    renderer = renderer_with(binding)
    renderer.load()
    result = renderer.render(REQUEST, tmp_path / "a.mp4")
    assert (result.width, result.height, result.num_frames, result.fps) == (576, 1024, 361, 24.0)
    assert result.has_audio is False


def test_after_clip_and_recover_free_gpu_memory_only() -> None:
    binding = FakeBinding()
    renderer = renderer_with(binding)
    renderer.load()
    binding.events.clear()
    renderer.after_clip()
    renderer.recover()
    assert binding.events == ["free", "free"]  # nothing that could drop the weights


def test_rendering_before_loading_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        renderer_with(FakeBinding()).render(REQUEST, tmp_path / "a.mp4")


# --- LtxBinding against stub modules ---------------------------------------------------


def binding_for(tmp_path: Path, **overrides: Any) -> LtxBinding:
    make_model_files(tmp_path)
    return LtxBinding(Settings(ltx_root=tmp_path, **overrides))


def test_the_pipeline_gets_paths_quantization_offload_and_registry_from_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install(monkeypatch)
    binding_for(tmp_path, offload_mode="cpu").build_pipeline()
    (built,) = calls.constructed
    paths = built["model_paths"]
    root = tmp_path / "models/ltx-2.5"
    assert paths.transformer_path == str(
        root / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    )
    assert paths.text_encoder_path.endswith("gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    assert paths.video_vae_path.endswith("vae/ltx-2.5-video-vae-bf16.safetensors")
    assert paths.audio_vae_path.endswith("vae/ltx-2.5-audio-vae-bf16.safetensors")
    assert paths.duration_head_path is None
    assert built["spatial_upsampler_path"].endswith(
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    )
    assert built["loras"] == []
    assert built["quantization"] == ("fp8-cast", paths.transformer_path)
    assert built["offload_mode"] is OffloadMode.CPU
    assert isinstance(built["registry"], StubRegistry) and built["registry"].cache_weights is True


@pytest.mark.parametrize("mode", ["cpu", "disk", "none"])
def test_offload_mode_comes_from_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    calls = install(monkeypatch)
    binding_for(tmp_path, offload_mode=mode).build_pipeline()
    assert calls.constructed[0]["offload_mode"] is OffloadMode(mode)


def test_the_registry_can_be_turned_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install(monkeypatch)
    binding_for(tmp_path, use_registry=False).build_pipeline()
    assert calls.constructed[0]["registry"] is None and StubRegistry.instances == []


def test_the_registry_is_skipped_when_the_constructor_does_not_accept_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install(monkeypatch, accepts_registry=False)
    binding_for(tmp_path, use_registry=True).build_pipeline()
    assert "registry" not in calls.constructed[0] and StubRegistry.instances == []


def test_quantization_none_builds_no_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install(monkeypatch)
    binding_for(tmp_path, quantization="none").build_pipeline()
    assert calls.constructed[0]["quantization"] is None


def test_missing_model_files_are_named_by_role_not_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    with pytest.raises(
        FileNotFoundError, match="transformer, text_encoder, video_vae, audio_vae, upsampler"
    ) as caught:
        LtxBinding(Settings(ltx_root=tmp_path)).build_pipeline()
    assert str(tmp_path) not in str(caught.value)


def test_generate_calls_the_pipeline_with_keywords_and_encodes_like_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install(monkeypatch)
    binding = binding_for(tmp_path)
    pipeline = binding.build_pipeline()
    target = tmp_path / "clip.mp4"
    assert binding.generate(pipeline, REQUEST, target) is True
    (call,) = calls.pipeline_calls
    assert call == {
        "prompt": "a prompt",
        "seed": 42,
        "height": 1024,
        "width": 576,
        "frame_rate": 24.0,
        "images": [],
        "num_frames": 361,
    }
    (encode,) = calls.encodes
    assert (
        encode["output_path"] == str(target)
        and encode["fps"] == 24
        and isinstance(encode["fps"], int)
    )
    assert encode["video_chunks_number"] == 3 and encode["audio"] is not None
    assert calls.inference_mode_entries == 1
    assert target.read_bytes()[4:8] == b"ftyp"


def test_a_clip_without_audio_reports_no_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install(monkeypatch)
    calls.audio_present = False
    binding = binding_for(tmp_path)
    assert binding.generate(binding.build_pipeline(), REQUEST, tmp_path / "c.mp4") is False


def test_fractional_frame_rates_are_passed_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install(monkeypatch)
    binding = binding_for(tmp_path)
    binding.generate(
        binding.build_pipeline(), RenderRequest(1, "p", 64, 64, 9, 23.976, 1), tmp_path / "c.mp4"
    )
    assert calls.encodes[0]["fps"] == 23.976 and encode_fps(30.0) == 30 and encode_fps(30.5) == 30.5


def test_gpu_cache_is_emptied_and_oom_is_recognised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install(monkeypatch)
    binding = binding_for(tmp_path)
    binding.free_gpu_cache()
    assert calls.empty_cache == 1
    assert binding.is_out_of_memory(StubOutOfMemoryError("x")) and binding.is_out_of_memory(
        MemoryError()
    )
    assert not binding.is_out_of_memory(ValueError("x"))


def test_without_torch_nothing_is_freed_and_nothing_is_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)
    binding = binding_for(tmp_path)
    binding.free_gpu_cache()  # no torch imported: must not import it just to free memory
    assert not binding.is_out_of_memory(ValueError("x"))
    assert "torch" not in __import__("sys").modules


def test_prepare_environment_sets_the_allocator_offline_mode_and_drops_ld_library_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/somewhere")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    binding_for(tmp_path).prepare_environment()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert os.environ["HF_HUB_OFFLINE"] == "1" and "LD_LIBRARY_PATH" not in os.environ
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    binding_for(tmp_path).prepare_environment()
    assert (
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"
    )  # an explicit value wins


def test_without_ltx_installed_loading_fails_with_an_import_error(tmp_path: Path) -> None:
    with pytest.raises(ImportError):
        binding_for(tmp_path).build_pipeline()


# --- the whole chain: server, LtxRenderer, LtxBinding, stub LTX --------------------------


def test_server_with_the_ltx_renderer_builds_once_warms_up_and_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install(monkeypatch)
    make_model_files(tmp_path / "ltx")
    settings = make_settings(tmp_path, renderer="ltx", ltx_root=tmp_path / "ltx")
    app = ServerApp(settings, LtxRenderer(settings), probe=lambda path: None)
    port = app.start()
    try:
        assert wait_until(lambda: app.ready)
        assert len(calls.constructed) == 1 and len(calls.pipeline_calls) == 1  # built + warmed up
        warm = calls.pipeline_calls[0]
        assert (warm["width"], warm["height"], warm["num_frames"]) == (256, 256, 9)
        assert not Path(calls.encodes[0]["output_path"]).exists()  # the throwaway clip is deleted
        for index in range(3):
            out = str(settings.allowed_output_dir / f"{index}-1.mp4")
            status, body, _ = request(port, "POST", "/generate", generate_body(out))
            assert status == 200 and Path(out).read_bytes()[4:8] == b"ftyp", body
        assert len(calls.constructed) == 1  # still one pipeline
        assert len(calls.pipeline_calls) == 4 and all(
            p is calls.pipelines[0] for p in calls.pipelines
        )
        assert calls.empty_cache >= 4  # after the warm-up and after each clip
        assert [
            f.name for f in settings.allowed_output_dir.iterdir() if f.name.startswith(".")
        ] == []
        calls.fail_with = StubOutOfMemoryError("cuda")
        status, body, _ = request(
            port, "POST", "/generate", generate_body(str(settings.allowed_output_dir / "9-1.mp4"))
        )
        assert status == 500 and body["error"]["code"] == "oom"
    finally:
        app.request_stop(0)
        app.shutdown()
