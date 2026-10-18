# LTX binding: what to verify on the instance

`model_server/ltx_server/ltx_binding.py` (pipeline mode) and `resident_binding.py` (resident mode) are the only files that touch torch or LTX-2. It was
written from the **public** LTX-2 source and has never been run against a real checkout (there is no
GPU, torch or LTX where it was written). Every line that depends on something not yet confirmed is
marked `# VERIFY-ON-INSTANCE: N` in the code, and `N` is the item number below. A test
(`tests/model_server/test_binding_notes.py`) fails if a marker has no item here or an item has no
marker, so the two cannot drift apart.

**Source that was read:** `Lightricks/LTX-2`, public `main`, commit
`be6d6d0a69d11a0a399c462ed92606692e5f8697`, read on 2026-10-18. The instance's checkout may be older
or newer. Record it:

```sh
git -C ~/LTX-2 log -1            # paste the hash and date here:  ______________________________
```

## Step 0: one command that checks most signatures

Run it with the LTX virtualenv's Python (`~/LTX-2/.venv/bin/python`). It prints what the binding
assumes next to what is installed. Anything that raises or looks different is an item below to fix.

```python
import inspect
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.loader.registry import ModelRegistry
from ltx_core.model.video_vae import get_video_chunks_number

print(inspect.signature(DistilledPipeline.__init__))
print(inspect.signature(DistilledPipeline.__call__))
print(inspect.signature(ModelPaths.from_split))
print(inspect.signature(ModelRegistry.__init__))
print(inspect.signature(encode_video))
print(inspect.signature(get_video_chunks_number))
print(list(QuantizationKind), list(OffloadMode))
```

Expected (from the commit above): `__init__(self, model_paths, spatial_upsampler_path, loras, device=None,
quantization=None, registry=None, compilation_config=None, offload_mode=OffloadMode.NONE,
alloc_trim_strategy=..., prompt_enhancer_gemma_root=None, diffvae_optimization=...)`; `__call__(self,
prompt, seed, height, width, frame_rate, images, num_frames=..., ...)`;
`ModelRegistry.__init__(self, *, cache_weights=True, cache_models=True)`;
`encode_video(video, fps, audio, output_path, video_chunks_number, ...)`;
`OffloadMode` values `none`, `cpu`, `disk`.

## The checklist

Each item: the assumption, where it sits in the code, the exact files to open under `~/LTX-2`, what to
confirm, and what to change if it is different. Tick them off in order.

### 1. Module paths and names

- **Assumption:** these imports exist: `ltx_pipelines.distilled.DistilledPipeline`,
  `ltx_pipelines.utils.model_paths.ModelPaths`, `ltx_pipelines.utils.quantization_factory.QuantizationKind`,
  `ltx_pipelines.utils.types.OffloadMode`.
- **Code:** `LtxBinding.build_pipeline`, first import block.
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/distilled.py` (the `import` lines and the
  `DistilledPipeline` class), `packages/ltx-pipelines/src/ltx_pipelines/utils/model_paths.py`, `packages/ltx-pipelines/src/ltx_pipelines/utils/quantization_factory.py`,
  `packages/ltx-pipelines/src/ltx_pipelines/utils/types.py`.
- **Confirm:** step 0 runs without an `ImportError`.
- **If different:** only the import lines in `build_pipeline` change.

### 2. Model file names and locations

- **Assumption:** the five files below `$LTX_ROOT/models/ltx-2.5/` are the ones the existing
  `render_test.sh` passes (names copied from the task description, not from the script):
  `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`,
  `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`,
  `vae/ltx-2.5-video-vae-bf16.safetensors`, `vae/ltx-2.5-audio-vae-bf16.safetensors`,
  `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`.
- **Code:** `MODEL_FILES` in `ltx_binding.py`.
- **Open:** the working `render_test.sh` (every `--*-path` argument) and
  `ls -R ~/LTX-2/models/ltx-2.5`.
- **Confirm:** all five paths match byte for byte, including the version suffixes.
- **If different:** edit `MODEL_FILES`; nothing else.

### 3. Building `ModelPaths` from separate files

- **Assumption:** `ModelPaths.from_split(transformer_path=, text_encoder_path=, video_vae_path=,
  audio_vae_path=)` is what the CLI builds for `--transformer-path --text-encoder-path --video-vae-path
  --audio-vae-path`, and that leaving `duration_head_path` unset is fine because the server always
  passes an explicit `num_frames`.
- **Code:** `build_pipeline` (the `ModelPaths.from_split` call).
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/utils/model_paths.py`
  (`from_split`, `model_paths_from_namespace`) and `packages/ltx-pipelines/src/ltx_pipelines/utils/args.py` (`_resolve_model_paths`).
- **Confirm:** the CLI run with your flags takes the split branch (not `from_monolith`), and
  `DistilledPipeline.__init__` accepts `duration_head_path=None` (it builds `DurationPredictor` from it;
  the source comment says it is `None` on LTX 2.5 / gemma4 checkpoints).
- **If different:** pass `duration_head_path=` or switch to `from_monolith`.

### 4. The fp8-cast quantization policy

- **Assumption:** `QuantizationKind("fp8-cast").to_policy(checkpoint_path=<transformer file>)` equals what
  `--quantization fp8-cast` produces with `--transformer-path`.
- **Code:** `build_pipeline` (the `QuantizationKind` lines).
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/utils/quantization_factory.py` and `packages/ltx-pipelines/src/ltx_pipelines/utils/args.py` (`_resolve_quantization`:
  it uses `--checkpoint-path`, else `--distilled-checkpoint-path`, else `--transformer-path`),
  `packages/ltx-core/src/ltx_core/quantization/fp8_cast.py`.
- **Confirm:** the checkpoint path given to `to_policy` is the transformer file in your CLI run.
- **If different:** change which path is passed.

### 5. `DistilledPipeline` constructor arguments

- **Assumption:** keyword arguments `model_paths`, `spatial_upsampler_path`, `loras` (empty list),
  `quantization`, `offload_mode`, and optionally `registry`; the binding passes `registry` only if
  the constructor's signature has that parameter (`inspect.signature`). Everything else keeps its default
  (`device=None`, `compilation_config=None`, `alloc_trim_strategy=TRIM`, `diffvae_optimization=...`),
  as the CLI's `main()` does.
- **Code:** `build_pipeline` (the `arguments` dict).
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/distilled.py`: `DistilledPipeline.__init__` and `main()`.
- **Confirm:** step 0 shows those parameter names; `main()` passes the same ones.
- **If different:** adjust the keyword names. If the instance's constructor has no `registry`, item 6 is moot.

### 6. The weight registry (the biggest unknown for RAM)

- **Assumption:** passing `ModelRegistry(cache_weights=True, cache_models=True)` as `registry=` keeps
  the checkpoint tensors in host RAM across requests, so clips after the first do not re-read ~65 GB
  from disk.
- **What the public source says (and where it is ambiguous):**
  - `registry=None` is *not* "no cache": `DiffusionStage.from_checkpoint` substitutes
    `ModelRegistry(cache_models=True, cache_weights=False)` (`blocks.py`, around lines 338 and 347). So
    by default model *shells* are cached and **weights are re-read on every call**.
  - `cache_weights=True` stores each loaded state dict in a dict **with no eviction**
    (`registry.py`). Everything that receives the registry is cached: the transformer (after the fp8 cast,
    keyed by the quantization ops), the Gemma text encoder, both VAEs and the upsampler. A rough
    estimate is ~22 GB (fp8 transformer) + ~24 GB (bf16 Gemma) + a few GB, **about 50 GB**, against a
    `MemoryMax` of 55 GB on a 61 GB machine. This is an estimate, not a measurement.
  - `OffloadMode.CPU` is described as "first pass reads from disk; subsequent passes reuse the CPU
    cache" (`types.py`, `optimization.md`). It is not stated whether "passes" means denoising passes inside
    one call or successive calls; streaming stages are also built and disposed per call (`blocks.py`,
    `_streaming_model`).
- **Code:** `build_pipeline` (the `ModelRegistry` lines); toggle with `USE_REGISTRY=0`.
- **Open:** `packages/ltx-core/src/ltx_core/loader/registry.py`,
  `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py` (`DiffusionStage.from_checkpoint`,
  `_streaming_model`, `PromptEncoder`), `packages/ltx-pipelines/docs/optimization.md`.
- **Confirm:** with the registry on, clip 2 is much faster than clip 1 *and* RAM stays under the floor
  (`docs/model-server-runbook.md`, measurement). Check that `registry=` really reaches the stages you care about.
- **If different:** the decision table in the runbook (turn the registry off, or `OFFLOAD_MODE=disk`).

### 7. `OffloadMode` values

- **Assumption:** `OffloadMode("cpu" | "disk" | "none")` works (the enum values are those strings).
- **Code:** `build_pipeline` (`OffloadMode(self._settings.offload_mode)`).
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/utils/types.py` (`class OffloadMode`).
- **Confirm:** step 0 prints the three members with those values.
- **If different:** map the setting to the enum by name instead.

### 8. Calling the pipeline

- **Assumption:** `pipeline(prompt=, seed=, height=, width=, frame_rate=, images=[], num_frames=)` returns a
  `PipelineOutput` with `.video` (an iterator of tensors), `.audio`, `.num_frames` and `.tiling_config`;
  `images=[]` means no image conditioning; `frame_rate` takes the float fps.
- **Code:** `LtxBinding.generate` (the `pipeline(...)` call).
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/distilled.py` (`__call__`, `stream_chunks`, `main()`), `packages/ltx-pipelines/src/ltx_pipelines/utils/types.py` (`PipelineOutput`).
- **Confirm:** names and order match `__call__`; `main()` goes through `stream_chunks` and
  `pipeline_output_from_chunks`, which `__call__` wraps (the binding relies on `__call__` being equivalent).
- **If different:** use `stream_chunks` plus `pipeline_output_from_chunks` exactly as `main()` does.

### 9. Encoding video and audio to the mp4

- **Assumption:** `encode_video(video=output.video, fps=<int>, audio=output.audio, output_path=<str>,
  video_chunks_number=get_video_chunks_number(output.num_frames, output.tiling_config))` is what the CLI
  does (SDR, `color_space=None`), the container comes from the `.mp4` extension (which is why temporary
  names end in `.mp4`), and `fps` is passed as an `int` when whole (the annotation says `int`).
- **Code:** `generate` (the `encode_video` call and its import line), `encode_fps`.
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/utils/media_io/encode.py` (`encode_video`),
  `packages/ltx-pipelines/src/ltx_pipelines/utils/media_io/__init__.py`, `packages/ltx-core/src/ltx_core/model/video_vae/__init__.py`
  (`get_video_chunks_number`), `packages/ltx-pipelines/src/ltx_pipelines/distilled.py` (`main()`, the last lines).
- **Confirm:** a clip written by the server plays, has audio, and `ffprobe` reports the requested
  size and frame rate. Check a fractional fps (23.976) if you ever use one.
- **If different:** adjust the arguments; `encode_video` is the only place output is written.

### 10. Reusing one pipeline object across requests

- **Assumption:** building `DistilledPipeline` once and calling it for every request is correct and is
  what makes later clips cheaper (with item 6's caching). The server never rebuilds it.
- **Code:** `LtxRenderer.load` (`ltx_renderer.py`).
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/distilled.py` (`__init__` stores stages; `__call__` runs them) and `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py`
  (do `DiffusionStage`, `PromptEncoder`, `VideoDecoder` build and dispose models inside each call?).
- **Confirm:** the measurement (clip 2 vs clip 1) and the journal: no "Building transformer" line should
  mean "re-reading weights from disk" for clip 2 unless that is what the cache is supposed to avoid.
- **If different:** the cost is per call regardless; see the runbook decision table.

### 11. The out-of-memory exception type

- **Assumption:** a CUDA out-of-memory raises `torch.cuda.OutOfMemoryError`. (The shell also treats any
  exception *named* `OutOfMemoryError`, and `MemoryError`, as out of memory.)
- **Code:** `LtxBinding.is_out_of_memory`.
- **Open:** the torch version in the LTX venv (`python -c "import torch; print(torch.__version__,
  torch.cuda.OutOfMemoryError)"`).
- **Confirm:** it prints a class. Also note the earlier report: the video decoder logs a CUDA OOM
  *warning* and recovers by itself; that is not an exception and not a failed clip.
- **If different:** extend `is_out_of_memory`.

### 12. Calling LTX from the server's request threads

- **Assumption:** building the pipeline in the background loading thread and then calling it from the
  HTTP server's per-request threads is fine, and `torch.inference_mode()` (thread-local, as in the CLI's
  `main()`) must be entered in the thread that renders, which `generate` does.
- **Code:** `generate` (the `with torch.inference_mode():`).
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/distilled.py` (`main` is decorated with `@torch.inference_mode()`); look for module-level
  or thread-local state in `ltx_core` (CUDA streams, compile caches) that assumes one thread.
- **Confirm:** warm-up (loading thread) and the first real clip (request thread) both succeed, and
  GPU use is not doubled.
- **If different:** run all LTX calls on one dedicated worker thread (a change inside `LtxRenderer` only).

### 13. Process environment

- **Assumption:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` must be set before torch first
  initialises CUDA (the binding sets it before any import and never overrides an explicit value);
  `HF_HUB_OFFLINE=1`; and `LD_LIBRARY_PATH` must be unset. Unsetting it *inside* the process is too late
  for the dynamic loader, so the systemd unit does it (`UnsetEnvironment=`); the binding only warns.
- **Code:** `LtxBinding.prepare_environment`; `deploy/systemd/ltx-server.service`.
- **Open:** how `render_test.sh` is launched (its environment) and `systemctl show ltx-server -p Environment`.
- **Confirm:** `cat /proc/$(pgrep -f 'ltx_server')/environ | tr '\0' '\n' | grep -E 'LD_LIBRARY|ALLOC|HF_HUB'`.
- **If different:** fix the unit, not the code.

### 14. Freeing GPU memory between clips

- **Assumption:** `gc.collect()` plus `torch.cuda.empty_cache()` after every clip releases only cached,
  unused GPU blocks and never touches the weights (cached weights live in the registry in host RAM).
  The pipeline's default `AllocatorTrimStrategy.TRIM` already runs `cleanup_memory()` after each stage.
- **Code:** `LtxBinding.free_gpu_cache`.
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py` (`_streaming_model`, `gpu_model`, `cleanup_memory`) and
  `packages/ltx-core/src/ltx_core/allocator_trim_strategy.py`.
- **Confirm:** `nvidia-smi` memory between clips returns to the post-load level and does not creep up
  over many clips (the measurement's GPU column).
- **If different:** nothing to change unless the GPU column grows clip after clip, which means a leak.

### 15. Whether a clip has audio

- **Assumption:** `output.audio is not None` means the mp4 has an audio track (LTX-2 generates audio with
  video), and the response's `hasAudio` reports it.
- **Code:** end of `generate`.
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/utils/types.py` (`PipelineOutput.audio: Audio`), `packages/ltx-pipelines/src/ltx_pipelines/utils/media_io/encode.py`
  (how `encode_video` handles `audio`).
- **Confirm:** `ffprobe -show_streams clip.mp4` lists an audio stream and the response says `true`.
- **If different:** probe the file for an audio stream instead.

## Resident mode (`RENDER_MODE=resident`): items 16 to 22

`resident_binding.py` builds every model once and keeps it in memory, then runs LTX's own
`DistilledPipeline.__call__` with its block objects replaced by stand-ins that hand out the resident
models. It was written from the same commit as above and has not been run. Do items 1 to 15 first
(it reuses that code), then these. Open `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py`,
`.../distilled.py`, `.../chunks/` and `packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py`.

### 16. Constructing `DistilledPipeline` reads no weights

- **Assumption:** the constructor only creates builders (and reads safetensors headers), so it is cheap
  and the models can then be built one by one.
- **Code:** `ResidentBinding.build_pipeline`.
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/distilled.py` (`DistilledPipeline.__init__`).
- **Confirm:** with `RENDER_MODE=resident`, the `phase` log line for `load_gemma` comes after the
  constructor returns, and constructing takes seconds and no RAM growth.
- **If different:** build the blocks by hand instead of through the pipeline constructor.

### 17. The private builder attributes used to build each model once

- **Assumption:** these exist and build what their names say: `prompt_encoder._text_encoder_builder`,
  `prompt_encoder._build_embeddings_processor()`, `stage._build_transformer()` (an `X0Model`, quantized),
  `upsampler._encoder_builder` / `_upsampler_builder`, `video_decoder._prepared_builder()`,
  `audio_decoder._decoder_builder` / `_vocoder_builder`. (`blocks.py`: `PromptEncoder`, `DiffusionStage`,
  `VideoUpsampler`, `VideoDecoder`, `AudioDecoder`.) The vocoder dtype is bf16 here; LTX uses fp32 only on MPS.
- **Code:** `Resident.load`.
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py` (`PromptEncoder`, `DiffusionStage`, `VideoUpsampler`, `VideoDecoder`, `AudioDecoder`).
- **Confirm:** `load` finishes; `nvidia-smi` shows about 22 GB (fp8 transformer) plus a few GB; the
  server RSS shows about 24 GB (Gemma). The `phase` lines for `load_gemma`, `load_transformer` and `load_small_models` are in the journal.
- **If different:** rename the attribute; the builders are underscore-private, so a newer LTX may move them.

### 18. Replacing the blocks on the pipeline object is honoured

- **Assumption:** `DistilledPipeline.__call__` reaches the models only through `self.prompt_encoder`,
  `self.stage` (via `stage._transformer_ctx(**kwargs)`), `self.upsampler`, `self.video_decoder` and
  `self.audio_decoder`, with these call shapes: `prompt_encoder([prompt], enhance_first_prompt=,
  enhance_static_cache=, enhance_prompt_image=)`, `upsampler(latent)`, `video_decoder(latent,
  tiling_config, generator, dtype=, keyframes=)`, `audio_decoder(latent)`; `video_decoder.checkpoint_path`
  and `.diffvae_optimization` are read before decoding.
- **Code:** `Resident.install` and the four stand-in classes.
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/distilled.py` (`__call__`, `stream_chunks`) and `packages/ltx-pipelines/src/ltx_pipelines/chunks/operators.py` (`denoise_chunks`, `spatially_upsample_chunks`, `decode_chunks`).
- **Confirm:** one clip renders in resident mode and the journal has `stage_1`, `upsample`, `stage_2`,
  `decode_video`, `decode_audio` phases and no "Building transformer" lines from LTX.
- **If different:** match the new call shapes in the stand-ins; do not copy LTX logic into them.

### 19. Moving a built model between host and GPU by re-pointing `.data`

- **Assumption:** `tensor.data = tensor.data.to(device)` for every parameter and buffer moves a built
  model (the fp8 transformer and Gemma) without rebuilding it, keeps its dtype (fp8 stays fp8) and
  leaves forward hooks and quantization wrappers working; the fp8 transformer has no tensors that must
  stay on one device.
- **Code:** `Resident.transformer_to_host`, `transformer_to_gpu`, `gemma_to_gpu`, `gemma_to_host`.
- **Open:** `packages/ltx-core/src/ltx_core/quantization/` (the fp8-cast policy) and `packages/ltx-pipelines/src/ltx_pipelines/utils/gpu_model.py`.
- **Confirm:** the `park_transformer` / `unpark_transformer` / `gemma_to_gpu` phase lines take seconds
  (not the tens of seconds a rebuild takes), no weight file is read after `ready` (watch `iostat` or
  `strace -e openat` on the server), and clip 2 gives a correct picture.
- **If different:** use `module.to(device)` on the whole module instead (same effect, new tensors), and
  check the quantization policy's module ops for device-bound state.

### 20. Pinned host memory for Gemma

- **Assumption:** `pin_memory()` on 24 GB of tensors succeeds under `LimitMEMLOCK=infinity` and
  `MemoryMax=55G`, and speeds up `gemma_to_gpu`.
- **Code:** `Resident._pin_gemma`.
- **Open:** nothing in LTX; check `ulimit -l` in the unit (`LimitMEMLOCK=infinity`) and the `gemma_host_copy` log line (the `torch` pin call).
- **Confirm:** the `gemma_host_copy` line says `pinned=True`; if it says `pinned=False` the server carried on
  unpinned (slower copies, still correct).
- **If different:** nothing to change unless the pin makes RSS exceed the budget; then drop the pin call.

### 21. Gemma built on the CPU and used through `encode`

- **Assumption:** the text encoder builder can build on `torch.device("cpu")`, and the module has
  `encode(prompts)` returning `(hidden_states, mask)` pairs that `embeddings_processor.process_hidden_states`
  accepts (`PromptEncoder.__call__` in `blocks.py`).
- **Code:** `Resident.load` (`load_gemma`), `_PromptEncoder.__call__`.
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py` (`PromptEncoder.__call__`, `_build_text_encoder`).
- **Confirm:** `encode_text` runs on the GPU after `gemma_to_gpu` with no device-mismatch error.
- **If different:** build Gemma on the GPU at load, then move it to the host once.

### 22. Upsampler, video decoder and audio decoder entry points

- **Assumption:** `ltx_core.model.upsampler.upsample_video(latent=, video_encoder=, upsampler=)`,
  `ltx_core.model.audio_vae.decode_audio(latent, decoder, vocoder)` and
  `video_decoder.decode_video(latent, tiling_config, generator, **extra)` exist as `blocks.py` uses them.
- **Code:** `_Upsampler`, `_AudioDecoder`, `_VideoDecoder` in `resident_binding.py`.
- **Open:** `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py` (`VideoUpsampler.__call__`, `VideoDecoder.__call__`, `AudioDecoder.__call__`).
- **Confirm:** step 0 style import check, then one clip. Compare with pipeline mode using
  `model_server/tools/compare_clips.py`.
- **If different:** only the import lines and argument names change.
