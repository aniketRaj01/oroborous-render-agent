"""RENDER_MODE=resident: every LTX model is built ONCE per process and stays in memory.

LTX's DistilledPipeline rebuilds the text encoder, the transformer (twice), the upsampler and the
decoders from the weight files on every clip, then disposes them. Here the pipeline object is built
as usual (its constructor only creates builders), each model is built exactly once through the
pipeline's own builders, and the pipeline's block objects are replaced by small stand-ins that hand
out the resident model instead of building and disposing one. Then LTX's own `__call__` runs
unchanged, so the seeding, sigmas, noise and denoising loops are LTX's, not ours.

Where each model lives:
  transformer (fp8)            GPU; parked in host RAM while Gemma encodes and (optionally) while
                               the video decodes
  Gemma text encoder           host RAM (pinned if possible); lent to the GPU for the prompt encode
  embeddings processor, spatial upsampler + video encoder, video decoder, audio decoder, vocoder
                               GPU, always

Every call into torch or ltx_* is in this file or ltx_binding.py and is imported lazily. Nothing
here was run against a real checkout: each assumption carries `VERIFY-ON-INSTANCE: N`, N being the
item in docs/ltx-binding-notes.md.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from ltx_server.config import Settings
from ltx_server.logs import event
from ltx_server.ltx_binding import LtxBinding
from ltx_server.renderer import RenderRequest

GIB = 1024**3
FALLBACK_HINT = "RENDER_MODE=pipeline is the fallback"


class ResidentModeError(RuntimeError):
    """The resident path cannot serve this request; the message names the fallback."""


def process_memory_gib() -> tuple[float, float]:
    """(current, peak) resident set size of this process in GiB; (0, 0) where /proc is missing."""
    try:
        fields = dict(
            line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines()
        )
        return (
            int(fields["VmRSS"].split()[0]) * 1024 / GIB,
            int(fields["VmHWM"].split()[0]) * 1024 / GIB,
        )
    except (OSError, KeyError, ValueError, IndexError):
        return (0.0, 0.0)


class Phases:
    """Times named phases, logs one event for each, and keeps this clip's seconds per phase."""

    def __init__(self) -> None:
        self.clip: dict[str, float] = {}
        self._peak_gpu_gib = 0.0

    def begin_clip(self) -> None:
        self.clip = {}
        self._peak_gpu_gib = 0.0
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.record(name, time.monotonic() - started)

    def record(self, name: str, seconds: float) -> None:
        self.clip[name] = round(self.clip.get(name, 0.0) + seconds, 2)
        allocated = reserved = peak = 0.0
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / GIB
            reserved = torch.cuda.memory_reserved() / GIB
            peak = torch.cuda.max_memory_allocated() / GIB
            torch.cuda.reset_peak_memory_stats()
        self._peak_gpu_gib = max(self._peak_gpu_gib, peak)
        rss, _ = process_memory_gib()
        event(
            logging.INFO,
            "phase",
            phase=name,
            seconds=round(seconds, 2),
            gpu_allocated_gib=round(allocated, 1),
            gpu_reserved_gib=round(reserved, 1),
            gpu_peak_gib=round(peak, 1),
            rss_gib=round(rss, 1),
        )

    def finish_clip(self) -> None:
        _, rss_peak = process_memory_gib()
        event(
            logging.INFO,
            "clip_memory",
            gpu_peak_gib=round(self._peak_gpu_gib, 1),
            rss_peak_gib=round(rss_peak, 1),
        )


class Resident:
    """The built models and where they currently are. Moves them; never rebuilds or disposes."""

    def __init__(self, torch: Any, device: Any, park_for_decode: bool, phases: Phases) -> None:
        self.torch = torch
        self.device = device
        self.park_for_decode = park_for_decode
        self.phases = phases
        self.transformer: Any = None
        self.gemma: Any = None
        self.embeddings: Any = None
        self.video_encoder: Any = None
        self.upsampler: Any = None
        self.video_decoder: Any = None
        self.audio_decoder: Any = None
        self.vocoder: Any = None
        self.transformer_at = "unknown"  # gpu | host | unknown (a move was interrupted)
        self.gemma_host: list[tuple[Any, Any]] = []
        self.stage_entries = 0
        self.upsamples = 0
        self.video_decodes = 0

    def begin_clip(self) -> None:
        self.stage_entries = self.upsamples = self.video_decodes = 0
        self.phases.begin_clip()

    # --- moving models. Tensors are re-pointed in place (`.data`), so the module objects, their
    # --- hooks and the references LTX holds to them stay valid.

    def _tensors(self, module: Any) -> list[Any]:
        return [*module.parameters(), *module.buffers()]

    def _free_gpu_cache(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def transformer_to_host(self) -> None:
        if self.transformer_at == "host":
            return
        with self.phases.phase("park_transformer"), self.torch.inference_mode(False):
            self.transformer_at = "unknown"
            # VERIFY-ON-INSTANCE: 19
            for tensor in self._tensors(self.transformer):
                tensor.data = tensor.data.to("cpu")
            self.transformer_at = "host"
            self._free_gpu_cache()

    def transformer_to_gpu(self) -> None:
        if self.transformer_at == "gpu":
            return
        with self.phases.phase("unpark_transformer"), self.torch.inference_mode(False):
            self.transformer_at = "unknown"
            # VERIFY-ON-INSTANCE: 19
            for tensor in self._tensors(self.transformer):
                tensor.data = tensor.data.to(self.device)
            self.transformer_at = "gpu"

    def gemma_to_gpu(self) -> None:
        with self.phases.phase("gemma_to_gpu"), self.torch.inference_mode(False):
            # VERIFY-ON-INSTANCE: 19
            for tensor, host in self.gemma_host:
                tensor.data = host.to(self.device, non_blocking=True)
            self.torch.cuda.synchronize()

    def gemma_to_host(self) -> None:
        with self.phases.phase("gemma_to_host"), self.torch.inference_mode(False):
            for tensor, host in self.gemma_host:
                tensor.data = host  # the host copy was never dropped, so this is a pointer swap
            self._free_gpu_cache()

    # --- loading, once ---------------------------------------------------------------------

    def load(self, pipeline: Any) -> None:
        torch, phases = self.torch, self.phases
        # VERIFY-ON-INSTANCE: 17
        encoder = pipeline.prompt_encoder
        stage = pipeline.stage
        dtype = torch.bfloat16
        with phases.phase("load_gemma"):
            # Built on the CPU so the 24 GB never touches the GPU, then pinned if possible.
            # VERIFY-ON-INSTANCE: 21
            self.gemma = encoder._text_encoder_builder.build(
                device=torch.device("cpu"), dtype=dtype
            ).eval()
            self._pin_gemma()
        with phases.phase("load_transformer"):
            self.transformer = stage._build_transformer()
            self.transformer_at = "gpu"
        with phases.phase("load_small_models"):
            self.embeddings = encoder._build_embeddings_processor()
            upsampler = pipeline.upsampler
            self.video_encoder = upsampler._encoder_builder.build(
                device=self.device, dtype=dtype
            ).eval()
            self.upsampler = upsampler._upsampler_builder.build(
                device=self.device, dtype=dtype
            ).eval()
            self.video_decoder = (
                pipeline.video_decoder._prepared_builder()
                .build(device=self.device, dtype=dtype)
                .eval()
            )
            audio = pipeline.audio_decoder
            self.audio_decoder = audio._decoder_builder.build(
                device=self.device, dtype=dtype
            ).eval()
            self.vocoder = audio._vocoder_builder.build(device=self.device, dtype=dtype).eval()

    def _pin_gemma(self) -> None:
        pinned = True
        for tensor in self._tensors(self.gemma):
            host = tensor.data
            if pinned:
                try:
                    # VERIFY-ON-INSTANCE: 20
                    host = host.pin_memory()
                except RuntimeError:
                    pinned = False
                    event(logging.WARNING, "gemma_pin_failed", note="continuing unpinned")
            tensor.data = host
            self.gemma_host.append((tensor, host))
        event(logging.INFO, "gemma_host_copy", tensors=len(self.gemma_host), pinned=pinned)

    # --- the stand-ins ---------------------------------------------------------------------

    def install(self, pipeline: Any) -> None:
        # VERIFY-ON-INSTANCE: 18
        pipeline.prompt_encoder = _PromptEncoder(self)
        pipeline.upsampler = _Upsampler(self)
        pipeline.video_decoder = _VideoDecoder(self, pipeline.video_decoder)
        pipeline.audio_decoder = _AudioDecoder(self)
        pipeline.stage._transformer_ctx = self.transformer_ctx

    @contextmanager
    def transformer_ctx(self, **_ignored: object) -> Iterator[Any]:
        self.stage_entries += 1
        if self.stage_entries > 2:
            raise ResidentModeError(
                f"this clip planned more than one chunk; resident mode serves single-chunk "
                f"clips only ({FALLBACK_HINT})"
            )
        self.transformer_to_gpu()
        with self.phases.phase(f"stage_{self.stage_entries}"):
            yield self.transformer  # no dispose: the same object serves both stages and all clips


class _PromptEncoder:
    def __init__(self, resident: Resident) -> None:
        self._r = resident

    def __call__(
        self, prompts: list[str], *, enhance_first_prompt: bool = False, **_ignored: object
    ) -> list[Any]:
        r = self._r
        if enhance_first_prompt:
            raise ResidentModeError(f"prompt enhancement is not supported ({FALLBACK_HINT})")
        r.transformer_to_host()  # Gemma and the transformer do not fit on the GPU together
        r.gemma_to_gpu()
        try:
            with r.phases.phase("encode_text"):
                # VERIFY-ON-INSTANCE: 21
                raw_outputs = r.gemma.encode(list(prompts))
        finally:
            r.gemma_to_host()
        with r.phases.phase("encode_embeddings"):
            return [r.embeddings.process_hidden_states(hs, mask) for hs, mask in raw_outputs]


class _Upsampler:
    def __init__(self, resident: Resident) -> None:
        self._r = resident

    def __call__(self, latent: Any) -> Any:
        r = self._r
        r.upsamples += 1
        if r.upsamples > 1:
            raise ResidentModeError(f"more than one chunk to upsample ({FALLBACK_HINT})")
        # VERIFY-ON-INSTANCE: 22
        from ltx_core.model.upsampler import upsample_video

        with r.phases.phase("upsample"):
            return upsample_video(
                latent=latent, video_encoder=r.video_encoder, upsampler=r.upsampler
            )


class _VideoDecoder:
    """Keeps the original's attributes (checkpoint_path, ...); replaces only __call__."""

    def __init__(self, resident: Resident, original: Any) -> None:
        self._r = resident
        self._original = original

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def __call__(
        self,
        latent: Any,
        tiling_config: Any = None,
        generator: Any = None,
        *,
        dtype: Any = None,
        keyframes: Any = None,
        device_fn: Any = None,
    ) -> Iterator[Any]:
        r = self._r
        r.video_decodes += 1
        if r.video_decodes > 1:
            raise ResidentModeError(f"more than one chunk to decode ({FALLBACK_HINT})")
        if r.park_for_decode:
            r.transformer_to_host()
        latent = latent.to(dtype=dtype if dtype is not None else r.torch.bfloat16)
        extra: dict[str, Any] = {}
        if keyframes is not None:
            extra["keyframes"] = keyframes
        if device_fn is not None:
            extra["device_fn"] = device_fn
        with r.phases.phase("decode_video"):
            # VERIFY-ON-INSTANCE: 22
            yield from r.video_decoder.decode_video(latent, tiling_config, generator, **extra)


class _AudioDecoder:
    def __init__(self, resident: Resident) -> None:
        self._r = resident

    def __call__(self, latent: Any) -> Any:
        r = self._r
        # VERIFY-ON-INSTANCE: 22
        from ltx_core.model.audio_vae import decode_audio

        with r.phases.phase("decode_audio"):
            return decode_audio(latent, r.audio_decoder, r.vocoder)


class ResidentBinding(LtxBinding):
    def __init__(self, settings: Settings) -> None:
        # The resident path owns where the weights live: no offload, no registry (the registry
        # hands back disposed models), whatever the pipeline-mode settings say.
        super().__init__(replace(settings, offload_mode="none", use_registry=False))
        self._park_for_decode = settings.park_transformer_for_decode
        self._phases = Phases()
        self._resident: Resident | None = None

    def build_pipeline(self) -> Any:
        import torch

        # VERIFY-ON-INSTANCE: 16
        pipeline = super().build_pipeline()  # builders only; no weight is read yet
        resident = Resident(torch, pipeline.device, self._park_for_decode, self._phases)
        with torch.inference_mode(False):
            resident.load(pipeline)
        resident.install(pipeline)
        self._resident = resident
        self._phases.finish_clip()
        return pipeline

    def generate(self, pipeline: Any, request: RenderRequest, output_path: Path) -> bool:
        resident = self._resident
        if resident is None:
            raise RuntimeError("the models are not loaded")
        resident.begin_clip()
        try:
            return super().generate(pipeline, request, output_path)
        finally:
            self._phases.finish_clip()

    def _encode(self, output: Any, request: RenderRequest, output_path: Path) -> None:
        with self._phases.phase("write_mp4"):
            super()._encode(output, request, output_path)

    def last_phase_seconds(self) -> Mapping[str, float]:
        return dict(self._phases.clip)
