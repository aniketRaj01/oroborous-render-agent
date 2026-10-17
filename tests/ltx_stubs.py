"""Stand-ins for torch and the LTX-2 modules the binding imports, so its code runs without a GPU.

They record how they are called. They mirror the signatures read from the public LTX-2 source, not
a real checkout: that comparison is the instance checklist in docs/ltx-binding-notes.md.
"""

from __future__ import annotations

import contextlib
import enum
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pytest


class StubOutOfMemoryError(RuntimeError):
    pass


class OffloadMode(enum.Enum):
    NONE = "none"
    CPU = "cpu"
    DISK = "disk"


class QuantizationKind(str, enum.Enum):
    FP8_CAST = "fp8-cast"
    FP8_SCALED_MM = "fp8-scaled-mm"

    def to_policy(self, checkpoint_path: str | None = None) -> tuple[str, str | None]:
        return (self.value, checkpoint_path)


@dataclass
class StubModelPaths:
    mode: str
    transformer_path: str | None
    text_encoder_path: str | None
    video_vae_path: str | None
    audio_vae_path: str | None
    duration_head_path: str | None

    @classmethod
    def from_split(cls, **paths: str | None) -> StubModelPaths:
        return cls(
            "split",
            **{
                k: paths.get(k)
                for k in (
                    "transformer_path",
                    "text_encoder_path",
                    "video_vae_path",
                    "audio_vae_path",
                    "duration_head_path",
                )
            },
        )


class StubRegistry:
    instances: ClassVar[list[StubRegistry]] = []

    def __init__(self, *, cache_weights: bool = True, cache_models: bool = True) -> None:
        self.cache_weights, self.cache_models = cache_weights, cache_models
        StubRegistry.instances.append(self)


@dataclass
class Calls:
    constructed: list[dict[str, Any]] = field(default_factory=list)
    pipeline_calls: list[dict[str, Any]] = field(default_factory=list)
    encodes: list[dict[str, Any]] = field(default_factory=list)
    empty_cache: int = 0
    inference_mode_entries: int = 0
    audio_present: bool = True
    fail_with: BaseException | None = None
    pipelines: list[Any] = field(default_factory=list)


def install(monkeypatch: pytest.MonkeyPatch, *, accepts_registry: bool = True) -> Calls:
    calls = Calls()
    StubRegistry.instances = []

    class PipelineOutput:
        def __init__(self) -> None:
            self.video: Iterator[bytes] = iter([b"frame"])
            self.audio: object | None = object() if calls.audio_present else None
            self.num_frames = 9
            self.tiling_config = "tiling"

    class _Pipeline:
        def __call__(self, **kwargs: Any) -> PipelineOutput:
            calls.pipeline_calls.append(kwargs)
            if calls.fail_with is not None:
                raise calls.fail_with
            return PipelineOutput()

    if accepts_registry:

        class DistilledPipeline(_Pipeline):
            def __init__(
                self,
                model_paths: Any,
                spatial_upsampler_path: str,
                loras: list[Any],
                device: Any = None,
                quantization: Any = None,
                registry: Any = None,
                offload_mode: Any = OffloadMode.NONE,
            ) -> None:
                calls.constructed.append(
                    {
                        "model_paths": model_paths,
                        "spatial_upsampler_path": spatial_upsampler_path,
                        "loras": loras,
                        "quantization": quantization,
                        "registry": registry,
                        "offload_mode": offload_mode,
                    }
                )
                calls.pipelines.append(self)

    else:

        class DistilledPipeline(_Pipeline):  # type: ignore[no-redef]
            def __init__(
                self,
                model_paths: Any,
                spatial_upsampler_path: str,
                loras: list[Any],
                quantization: Any = None,
                offload_mode: Any = OffloadMode.NONE,
            ) -> None:
                calls.constructed.append(
                    {
                        "model_paths": model_paths,
                        "spatial_upsampler_path": spatial_upsampler_path,
                        "loras": loras,
                        "quantization": quantization,
                        "offload_mode": offload_mode,
                    }
                )
                calls.pipelines.append(self)

    def encode_video(**kwargs: Any) -> None:
        calls.encodes.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"\x00\x00\x00\x18ftypisom" + b"STUB-NOT-A-VIDEO")

    @contextlib.contextmanager
    def inference_mode() -> Iterator[None]:
        calls.inference_mode_entries += 1
        yield

    def empty_cache() -> None:
        calls.empty_cache += 1

    def module(name: str, **attributes: Any) -> types.ModuleType:
        created = types.ModuleType(name)
        created.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, created)
        return created

    cuda = types.SimpleNamespace(
        OutOfMemoryError=StubOutOfMemoryError, is_available=lambda: True, empty_cache=empty_cache
    )
    module("torch", inference_mode=inference_mode, cuda=cuda)
    module("ltx_pipelines")
    module("ltx_pipelines.distilled", DistilledPipeline=DistilledPipeline)
    module("ltx_pipelines.utils")
    module("ltx_pipelines.utils.model_paths", ModelPaths=StubModelPaths)
    module("ltx_pipelines.utils.quantization_factory", QuantizationKind=QuantizationKind)
    module("ltx_pipelines.utils.types", OffloadMode=OffloadMode)
    module("ltx_pipelines.utils.media_io", encode_video=encode_video)
    module("ltx_core")
    module("ltx_core.loader")
    module("ltx_core.loader.registry", ModelRegistry=StubRegistry)
    module("ltx_core.model")
    module("ltx_core.model.video_vae", get_video_chunks_number=lambda frames, tiling: 3)
    return calls


def make_model_files(root: Path, models_subdir: str = "models/ltx-2.5") -> None:
    from ltx_server.ltx_binding import MODEL_FILES

    for relative in MODEL_FILES.values():
        target = root / models_subdir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
