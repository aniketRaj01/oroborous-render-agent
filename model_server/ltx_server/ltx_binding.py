"""The thin layer over LTX-2. EVERY call into torch or ltx_* lives in this file.

Nothing here is imported or executed until a method runs, so the server shell and the tests work
on a machine without torch or LTX. The code was written against the public LTX-2 source (see
docs/ltx-binding-notes.md for the commit that was read) and has NOT been run against a real
checkout. Each line that depends on an unverified assumption carries a
`VERIFY-ON-INSTANCE: N` marker; N is the item number in docs/ltx-binding-notes.md.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ltx_server.config import Settings
from ltx_server.imageinfo import image_size
from ltx_server.logs import event
from ltx_server.renderer import RenderRequest

# VERIFY-ON-INSTANCE: 2
MODEL_FILES = {
    "transformer": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "upsampler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}


def encode_fps(fps: float) -> int | float:
    """LTX's encode_video is annotated `fps: int`; hand it an int whenever the rate is whole."""
    return int(fps) if float(fps).is_integer() else fps


class LtxBinding:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # --- environment ---------------------------------------------------------------------------

    def prepare_environment(self) -> None:
        """Must run before torch is first imported."""
        # VERIFY-ON-INSTANCE: 13
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ["HF_HUB_OFFLINE"] = "1"
        if "LD_LIBRARY_PATH" in os.environ:
            # VERIFY-ON-INSTANCE: 13
            event(
                logging.WARNING, "ld_library_path_set", note="unset it in the unit; too late here"
            )
            del os.environ["LD_LIBRARY_PATH"]

    # --- building the pipeline -----------------------------------------------------------------

    def model_paths(self) -> dict[str, str]:
        root = self._settings.models_dir
        paths = {key: str(root / relative) for key, relative in MODEL_FILES.items()}
        missing = [key for key, path in paths.items() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"LTX model file(s) not found: {', '.join(missing)}")
        return paths

    def build_pipeline(self) -> Any:
        """Build ONE DistilledPipeline, with the options the CLI run uses, for reuse."""
        import inspect

        # VERIFY-ON-INSTANCE: 1
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
        from ltx_pipelines.utils.types import OffloadMode

        paths = self.model_paths()
        # VERIFY-ON-INSTANCE: 3
        model_paths = ModelPaths.from_split(
            transformer_path=paths["transformer"],
            text_encoder_path=paths["text_encoder"],
            video_vae_path=paths["video_vae"],
            audio_vae_path=paths["audio_vae"],
        )
        quantization = None
        if self._settings.quantization != "none":
            # VERIFY-ON-INSTANCE: 4
            kind = QuantizationKind(self._settings.quantization)
            quantization = kind.to_policy(checkpoint_path=paths["transformer"])

        # VERIFY-ON-INSTANCE: 5
        accepted = inspect.signature(DistilledPipeline.__init__).parameters
        arguments: dict[str, Any] = {
            "model_paths": model_paths,
            "spatial_upsampler_path": paths["upsampler"],
            "loras": [],
            "quantization": quantization,
        }
        # VERIFY-ON-INSTANCE: 7
        arguments["offload_mode"] = OffloadMode(self._settings.offload_mode)
        if self._settings.use_registry and "registry" in accepted:
            # VERIFY-ON-INSTANCE: 6
            from ltx_core.loader.registry import ModelRegistry

            arguments["registry"] = ModelRegistry(cache_weights=True, cache_models=True)
        event(
            logging.INFO,
            "building_pipeline",
            offload=self._settings.offload_mode,
            quantization=self._settings.quantization,
            registry="registry" in arguments,
        )
        return DistilledPipeline(**arguments)

    # --- generating ----------------------------------------------------------------------------

    def generate(self, pipeline: Any, request: RenderRequest, output_path: Path) -> bool:
        """Run the pipeline and encode video and audio to `output_path`; True if it has audio."""
        import torch

        # VERIFY-ON-INSTANCE: 12
        with torch.inference_mode():
            output = self._call_pipeline(pipeline, request)
            self._encode(output, request, output_path)
        # VERIFY-ON-INSTANCE: 15
        return bool(output.audio is not None)

    def _call_pipeline(self, pipeline: Any, request: RenderRequest) -> Any:
        # VERIFY-ON-INSTANCE: 8
        return pipeline(
            prompt=request.prompt,
            seed=request.seed,
            height=request.height,
            width=request.width,
            frame_rate=request.fps,
            images=self._condition_images(request),
            num_frames=request.num_frames,
        )

    def _condition_images(self, request: RenderRequest) -> list[Any]:
        """[] without a first-frame image (exactly the call made before images existed)."""
        if request.condition_image_path is None:
            return []
        # VERIFY-ON-INSTANCE: 23
        from ltx_pipelines.utils.types import ImageConditioningInput

        size = image_size(request.condition_image_path)
        event(
            logging.INFO,
            "condition_image",
            path=str(request.condition_image_path),
            image_width=size[0] if size else "unknown",
            image_height=size[1] if size else "unknown",
            render_width=request.width,
            render_height=request.height,
            same_aspect=(
                size[0] * request.height == size[1] * request.width if size else "unknown"
            ),
            strength=request.condition_strength,
        )
        # The pipeline resizes (fills, then center-crops) the image to the render size, and
        # fills in crf itself when it is left out.
        return [
            ImageConditioningInput(
                path=str(request.condition_image_path),
                frame_idx=0,
                strength=request.condition_strength,
            )
        ]

    def _encode(self, output: Any, request: RenderRequest, output_path: Path) -> None:
        # VERIFY-ON-INSTANCE: 9
        from ltx_core.model.video_vae import get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video

        encode_video(
            video=output.video,
            fps=encode_fps(request.fps),
            audio=output.audio,
            output_path=str(output_path),
            video_chunks_number=get_video_chunks_number(output.num_frames, output.tiling_config),
        )

    def last_phase_seconds(self) -> Mapping[str, float]:
        return {}

    # --- memory --------------------------------------------------------------------------------

    def free_gpu_cache(self) -> None:
        """Release cached GPU blocks that nothing uses. Never touches the weights."""
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            # VERIFY-ON-INSTANCE: 14
            torch.cuda.empty_cache()

    def is_out_of_memory(self, exc: BaseException) -> bool:
        if isinstance(exc, MemoryError):
            return True
        torch = sys.modules.get("torch")
        # VERIFY-ON-INSTANCE: 11
        return torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
