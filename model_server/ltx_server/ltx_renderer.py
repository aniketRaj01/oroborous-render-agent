"""LtxRenderer: the Renderer the server uses on the instance. Knows nothing about LTX itself;
every call into torch or ltx_* goes through a Binding (see ltx_binding.py)."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from ltx_server.config import Settings
from ltx_server.logs import event
from ltx_server.renderer import RendererOutOfMemory, RenderRequest, RenderResult

WARMUP_PROMPT = "a quiet landscape at dawn"


class Binding(Protocol):
    def prepare_environment(self) -> None: ...

    def build_pipeline(self) -> Any: ...

    def generate(self, pipeline: Any, request: RenderRequest, output_path: Path) -> bool: ...

    def free_gpu_cache(self) -> None: ...

    def is_out_of_memory(self, exc: BaseException) -> bool: ...


class LtxRenderer:
    def __init__(self, settings: Settings, binding: Binding | None = None) -> None:
        self._settings = settings
        self._binding = binding
        self._pipeline: Any = None

    def _bound(self) -> Binding:
        if self._binding is None:
            from ltx_server.ltx_binding import LtxBinding  # imports no torch or ltx_* itself

            self._binding = LtxBinding(self._settings)
        return self._binding

    def load(self) -> None:
        if self._pipeline is not None:
            raise RuntimeError("the pipeline is already built; it is built once per process")
        binding = self._bound()
        binding.prepare_environment()
        # VERIFY-ON-INSTANCE: 10
        self._pipeline = binding.build_pipeline()  # built once; every request reuses it

    def warm_up(self) -> None:
        """One small generation so the first real clip does not pay the first-pass disk read."""
        if not self._settings.warmup:
            event(logging.INFO, "warm_up_skipped")
            return
        pipeline = self._loaded()
        binding = self._bound()
        request = RenderRequest(
            job_id=0,
            prompt=WARMUP_PROMPT,
            width=self._settings.warmup_width,
            height=self._settings.warmup_height,
            num_frames=self._settings.warmup_frames,
            fps=self._settings.default_fps,
            seed=self._settings.default_seed,
        )
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="ltx-warmup-") as directory:
            binding.generate(pipeline, request, Path(directory) / "warmup.mp4")
        binding.free_gpu_cache()
        event(
            logging.INFO,
            "warm_up_generation",
            seconds=round(time.monotonic() - started, 1),
            width=request.width,
            height=request.height,
            frames=request.num_frames,
        )

    def render(self, request: RenderRequest, tmp_path: Path) -> RenderResult:
        pipeline = self._loaded()
        binding = self._bound()
        try:
            has_audio = binding.generate(pipeline, request, tmp_path)
        except Exception as exc:
            if binding.is_out_of_memory(exc):
                raise RendererOutOfMemory(type(exc).__name__) from exc
            raise
        return RenderResult(
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            fps=request.fps,
            has_audio=has_audio,
        )

    def after_clip(self) -> None:
        self._bound().free_gpu_cache()

    def recover(self) -> None:
        self._bound().free_gpu_cache()

    def _loaded(self) -> Any:
        if self._pipeline is None:
            raise RuntimeError("the model is not loaded")
        return self._pipeline
