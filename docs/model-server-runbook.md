# Model server runbook

The **model server** (`model_server/ltx_server`, run as `python -m ltx_server`) is the program that
actually renders clips on the GPU instance. It builds the LTX pipeline **once**, keeps the weights
cached in RAM, and renders one clip per request on `127.0.0.1` for the render agent. Its HTTP contract
is [`model-server-contract.md`](model-server-contract.md); everything it assumes about LTX itself is
in [`ltx-binding-notes.md`](ltx-binding-notes.md) and must be checked on the instance before trusting it.

Why it exists: running `python -m ltx_pipelines.distilled` once per clip re-reads about 65 GB of
weights every time. This server pays that cost once at start-up (and in a warm-up clip), then each
request only renders. Whether that really works, and what it costs in RAM, is what the measurement
below decides.

## What runs where

```
render-agent.service  --POST /generate-->  ltx-server.service  (this server, port 8000, loopback only)
  writes nothing                              loads LTX once, renders, writes <scratch>/<job>-<attempt>.mp4
```

Both processes share the agent's scratch directory (`/tmp/render-agent` by default), so neither unit
uses `PrivateTmp`. The server's code travels with the agent's releases
(`/opt/render-agent/current/model_server`); the LTX checkout and its virtualenv are separate.

## Install the unit (once, with the rest of the image)

```sh
sudo deploy/install.sh --service-user <login-user> --ltx-root /home/<login-user>/LTX-2
```

`--ltx-root` is the LTX-2 checkout whose `.venv` has torch and LTX. It installs and enables
`ltx-server.service` and creates `/etc/render-agent/ltx-server.env` (all commented out). Preview with
`--dry-run`. Pre-load the first agent release as described in `instance-runbook.md` so that
`/opt/render-agent/current/model_server` exists before the first boot.

## First test, by hand

Do this before enabling anything, to see the server work with the fake renderer and then with LTX.

1. **Fake renderer** (no GPU, proves the plumbing):

   ```sh
   cd /opt/render-agent/current/model_server
   RENDERER=fake PORT=8000 python3 -m ltx_server        # prints "listening on 127.0.0.1:8000"
   ```

   In another terminal: `curl -s localhost:8000/health` gives `{"ok": true, "modelLoaded": true}`.
   Stop with Ctrl-C.
2. **Make the scratch directory** the server may write into (the agent normally creates it):
   `mkdir -p /tmp/render-agent`.
3. **Real LTX**, stopping the unit first so the port and the GPU are free:

   ```sh
   sudo systemctl stop ltx-server render-agent
   cd ~/LTX-2
   PYTHONPATH=/opt/render-agent/current/model_server HF_HUB_OFFLINE=1 \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True env -u LD_LIBRARY_PATH \
     .venv/bin/python -m ltx_server
   ```

   It starts listening at once; `/health` says `modelLoaded: false` until the weights are loaded **and** the
   warm-up clip has rendered (minutes the first time). While loading, `/generate` answers `503` with
   `"model is still loading"`.
4. **One real clip**, once `/health` says `true`:

   ```sh
   curl -s localhost:8000/generate -H 'content-type: application/json' -d '{
     "jobId": 1, "prompt": "a calm lake at dawn",
     "params": {"width": 576, "height": 1024, "numFrames": 97, "fps": 24, "seed": 42},
     "outputPath": "/tmp/render-agent/1-1.mp4"}'
   ffprobe /tmp/render-agent/1-1.mp4
   ```

   Expect `{"ok": true, ... "render": {...}}`, a playable mp4 with video and audio, and no
   `.…partial.mp4` file left in the directory.

If something in step 3 or 4 fails, work through `ltx-binding-notes.md` (the checklist exists for exactly
this) before changing anything else.

## Reading its logs

```sh
journalctl -u ltx-server -b                   # this boot
journalctl -u ltx-server -b | grep -E 'event=(ready|generate_done|generate_failed|too_many_oom)'
```

One line per event, `key=value`. The prompt is never logged; failures log the exception *type* and where
it happened (`where=file:line:function`), not its message, because messages can quote the prompt.

| Event | Meaning |
|---|---|
| `listening` | the HTTP server is up (the model may still be loading) |
| `building_pipeline` | LTX pipeline construction started (shows offload mode, quantization, whether the registry is on) |
| `model_loaded`, `warm_up_generation`, `warm_up_done`, `ready` | start-up phases with seconds; `ready` is when `/health` turns true |
| `generate_start` / `generate_done` | a clip: ids, size, frames, `render_s`, `duration_s`, `bytes` |
| `generate_failed` | `kind=oom` or `render_error`, `error_type`, `where` |
| `too_many_oom_exiting` | two out-of-memory failures in a row: the server exits (status 75) so systemd restarts it clean |
| `load_failed` | the model could not be built (status 3), usually a missing file or an LTX API mismatch |
| `signal_received`, `stopped`, `stopped_during_generation` | shutdown on SIGTERM |

## Measuring: should the weights stay in RAM?

`model_server/tools/measure.py` sends clips one after another and records memory every 5 seconds. Run it
while the server is up (by hand as above, or the unit):

```sh
cat > /tmp/prompts.txt <<'PROMPTS'
a calm lake at dawn
a busy market street at night
PROMPTS
python3 /opt/render-agent/current/model_server/tools/measure.py \
  --prompts /tmp/prompts.txt --count 2 --width 576 --height 1024 --num-frames 361 --fps 24 \
  --csv /tmp/measure-cpu.csv --ram-floor-gb 4
```

Use the real clip size you will render (the 15 second vertical clip above). It waits for `/health`, finds
the server process by name (or pass `--pid`), prints one line per clip and then a summary:

```
clip  wall_s  render_s  result
   1   412.0     380.5  ok
   2   168.0     150.2  ok

clip 1: 412.0 s    clips 2..2: 168.0 s on average    (+59% faster after the first)
lowest MemAvailable: 6.2 GB
peak server RSS:     47.1 GB
peak GPU memory:     39.6 GB
RAM floor (4.0 GB): never breached
```

(Numbers above are illustrative, not measurements.) The CSV has a row every 5 seconds and at each clip
start and end: `mem_available_mb, server_rss_mb, gpu_used_mb`. Watch it live with `tail -f`. Also note the
time to `ready` from the journal (`event=ready seconds=`), and run it twice, once per candidate setting.

### Decision table

Compare the defaults (`OFFLOAD_MODE=none`, `USE_REGISTRY=0`, `RENDER_MODE=pipeline`) with the alternatives by setting them in
`/etc/render-agent/ltx-server.env` and `sudo systemctl restart ltx-server` between runs.

| What the measurement shows | Meaning | What to change |
|---|---|---|
| Clip 2 is clearly faster than clip 1 (say more than 25 percent), the floor was never breached, peak RSS is comfortably under the 55 GB `MemoryMax` | weights are being reused and fit | Keep the defaults. Done. |
| Clip 2 is about as slow as clip 1 (under 10 percent faster) | the weights are being read again for every clip | This is expected in `pipeline` mode (every clip rebuilds the models). Try `RENDER_MODE=resident`. Do not turn the registry on: `USE_REGISTRY=1` gave noise or an out-of-memory failure (see "Configurations seen on the instance"). |
| `MemAvailable` fell below the floor, or `journalctl -u ltx-server` shows the unit killed by the memory limit | the RAM cache is too big for this machine | In this order: `USE_REGISTRY=0` (already the default), `RENDER_MODE=pipeline` if resident mode was on (its host RAM is about 46 GB), then `OFFLOAD_MODE=disk` (lowest RAM, slowest). Re-measure after each. Do **not** raise `MemoryMax` above about 57G: the OS and the SSM agent need the rest. |
| The unit is killed at exactly `MemoryMax` but `MemAvailable` never got low | the cgroup cap is lower than the working set, the machine has room | Raise `MemoryMax` in a drop-in (`systemctl edit ltx-server`) in 1 GB steps, staying under about 57G. |
| Clip 1 is much slower than clip 2 only because the first pass reads the disk | normal, the warm-up should absorb it | Make sure `WARMUP=1` (default). Then clip 1 should already be close to clip 2, and `ready` takes longer instead. If `ready` is too slow for the agent's 20 minute wait, raise `MODEL_SERVER_READY_TIMEOUT_MINUTES` on the agent. |
| `generate_failed kind=oom` on a normal clip, or `too_many_oom_exiting` | GPU memory, not RAM | Do not change RAM settings. Check `peak GPU memory`; in resident mode try `PARK_TRANSFORMER_FOR_DECODE=1`; in pipeline mode `OFFLOAD_MODE=cpu` uses less GPU memory but see the failed configurations above; shorten `MAX_FRAMES`/the clip; look at notes item 14 (a GPU memory leak shows as the GPU column growing clip after clip). |
| The warm-up itself fails with out of memory (`load_failed`) | the machine is too small for the warm-up size or the settings | `WARMUP=0`, or smaller `WARMUP_WIDTH/HEIGHT/FRAMES`; then re-check the first real clip. |

After deciding, record the choice (and the numbers) in the PR or a note next to the instance image, and
leave the other settings alone.

## Render modes

`RENDER_MODE` chooses how the clip is rendered. Set it in `/etc/render-agent/ltx-server.env`, then
`sudo systemctl restart ltx-server` (the server announces `ready` only after the warm-up clip).

| Mode | What it does | Per clip, measured on the L40S instance |
|---|---|---|
| `pipeline` (default) | LTX's `DistilledPipeline` as it ships: it builds the text encoder and the transformer (twice) from the weight files for every clip and disposes them | 248 to 256 s, warm-up 158 s |
| `resident` | builds every model once at start-up and keeps it in memory; LTX's own `__call__` runs unchanged with the resident models | target 75 to 100 s: **not yet measured** |

In `pipeline` mode about 190 s of each clip is building models; the denoising itself is stage 1 at about
13 s and stage 2 at about 35 s. Resident mode removes the rebuilding. The text encoder (about 24 GB)
lives in host RAM and is lent to the GPU for the prompt encode; the fp8 transformer (about 22 GB) stays
on the GPU, except that it is moved to host RAM while the text encoder is on the GPU (they do not fit
together in 46 GB) and, with `PARK_TRANSFORMER_FOR_DECODE=1`, while the video decodes. The other
models (embeddings processor, upsampler, decoders) stay on the GPU. Host RAM budget: text encoder plus
a parked transformer is about 46 GB, under the unit's `MemoryMax=55G`; `rss_gib` in the phase lines shows the
real figure. Resident mode serves single-chunk clips only (all of ours); anything else fails the request
with an error that names `RENDER_MODE=pipeline`. The 2 consecutive OOM rule still applies: after an
out-of-memory failure the server restarts rather than trust the state of the moved models.

### Configurations seen on the instance

Clip: 576x1024, 361 frames, 24 fps, seed 42 (15 s, vertical). Unit `MemoryMax=55G`, `LimitMEMLOCK=infinity`,
LTX checkout at commit `be6d6d0a69d11a0a399c462ed92606692e5f8697`, fp8-cast.

| Settings | Result |
|---|---|
| `OFFLOAD_MODE=none`, `USE_REGISTRY=0` | **known good**: correct output, 248 to 256 s per clip. These are now the defaults. |
| `OFFLOAD_MODE=cpu`, `USE_REGISTRY=1` | about 100 s per clip but **the output is noise**: different prompts gave byte-identical mp4s (a coloured block mosaic). Do not use. The likely cause is that LTX disposes a model's weights after use and the registry hands the emptied model back. |
| `OFFLOAD_MODE=none`, `USE_REGISTRY=1` | CUDA out of memory while loading (`sft_loader.py`, 18 MB free of 47.6 GB); never ready. Do not use. |

An earlier CLI run peaked at 39.6 GB of GPU memory during the video decode (with an allocator warning:
it asked for 6.8 GB with 6.1 GB free, then completed).

### Switching to resident mode, and proving it renders the same pictures

On the instance, with the unit stopped or after a restart in each mode. `<ltx-python>` is the LTX
virtualenv's python (`<clone>/.venv/bin/python`).

1. Pipeline mode (the default): render a prompt P with seed 42 and keep the file as `A.mp4`.
2. Add `RENDER_MODE=resident` to `/etc/render-agent/ltx-server.env`, restart, wait for `ready`
   (`journalctl -u ltx-server -f`). Render the same P and seed into `B.mp4`.
3. Compare:

   ```sh
   <ltx-python> /opt/render-agent/current/model_server/tools/compare_clips.py A.mp4 B.mp4
   ```

   It prints frame count, size, duration and audio for each file, then the mean absolute difference
   (0 to 255) and PSNR for five evenly spaced frames. Same frame count and size are required.
   **What "close" looks like:** the same seed and prompt run the same code on the same weights, so
   expect (near) identical frames: a mean absolute difference under about 1 and PSNR above about 40 dB.
   Identical files are fine. Differences of several grey levels, or a PSNR under 30 dB, mean the
   pictures differ: look at both clips, then at notes items 18 and 19. Noise-like frames or two prompts
   giving the same file mean the models were reused after being emptied: stop and report it.
4. Render two clips with different prompts in resident mode: the files must differ and each must show
   its own prompt.

### Reading the phase timings

Every phase of every clip is one log line, and the totals are on `generate_done`:

```
event=phase phase=encode_text seconds=2.1 gpu_allocated_gib=27.3 gpu_reserved_gib=29.0 gpu_peak_gib=27.8 rss_gib=47.2
...
event=clip_memory gpu_peak_gib=38.4 rss_peak_gib=48.0
event=generate_done job_id=... render_s=88.2 ... phases=park_transformer=3.1,gemma_to_gpu=4.0,encode_text=2.1,...
```

(Numbers are illustrative.) `journalctl -u ltx-server | grep -E 'event=(phase|clip_memory|generate_done)'`.
Phases, in order: `park_transformer`, `gemma_to_gpu`, `encode_text`, `gemma_to_host`, `encode_embeddings`,
`unpark_transformer`, `stage_1`, `upsample`, `stage_2`, `park_transformer` again (when the decode park is
on; its seconds add up under one name), `decode_video`, `decode_audio`, `write_mp4`. The model moves
should each take seconds; if one takes tens of seconds, a model is being rebuilt from disk (notes item
19). At start-up the `load_gemma`, `load_transformer` and `load_small_models` phases show what was loaded.
`gpu_peak_gib` is the high-water mark since the previous phase; `clip_memory` gives the whole clip.
Pipeline mode logs `generate_done` without the `phases` field.

Try `PARK_TRANSFORMER_FOR_DECODE=0` once: if `decode_video` completes without out-of-memory and
`gpu_peak_gib` stays clearly under the card's 46 GB, keep it off (it saves two moves, a few seconds).

## Where the server's code comes from

The unit's `PYTHONPATH` is `/opt/render-agent/current/model_server`: the code that ships with the agent
release. An instance that runs the server from a git clone instead (to test a branch before it is
released) overrides it in `/etc/render-agent/ltx-server.env`:

```
PYTHONPATH=<clone>/model_server
```

`<clone>` is the checkout of this repository on the instance, on the branch to test. Restart the unit
after a `git pull`. Remove the line to go back to the released code. The tool paths in this runbook use
`/opt/render-agent/current/model_server`; use `<clone>/model_server` instead in that case.

## Settings

All environment variables, with defaults (set them in `/etc/render-agent/ltx-server.env`):

| Variable | Default | Purpose |
|---|---|---|
| `HOST`, `PORT` | `127.0.0.1`, `8000` | bind address (loopback only, anything else is refused) |
| `RENDERER` | `ltx` | `ltx` or `fake` |
| `LTX_ROOT`, `MODELS_SUBDIR` | `~/LTX-2`, `models/ltx-2.5` | where the weights are |
| `RENDER_MODE` | `pipeline` | `pipeline` (LTX's pipeline as it ships) or `resident` (every model built once; see "Render modes") |
| `PARK_TRANSFORMER_FOR_DECODE` | `1` | resident mode only: move the transformer to host RAM before the video decode |
| `OFFLOAD_MODE` | `none` | `none`, `cpu` or `disk` (pipeline mode; resident mode always uses `none`) |
| `QUANTIZATION` | `fp8-cast` | LTX quantization kind, or `none` |
| `USE_REGISTRY` | `0` | pipeline mode only: cache checkpoint tensors in RAM (`registry=` of the pipeline). Leave it off: see the failed configurations below |
| `ALLOWED_OUTPUT_DIR` | `/tmp/render-agent` | clips may only be written inside it |
| `DEFAULT_WIDTH/HEIGHT/NUM_FRAMES/FPS/SEED` | `576/1024/361/24/42` | used when a request omits a param |
| `MAX_FRAMES` | `481` | longest accepted clip (frames must be 8k+1) |
| `WARMUP`, `WARMUP_WIDTH/HEIGHT/FRAMES` | `1`, `256/256/9` | start-up warm-up clip |
| `MAX_CONSECUTIVE_OOMS` | `2` | out-of-memory failures in a row before the server exits for a clean restart |
| `SHUTDOWN_GRACE_SECONDS` | `30` | how long SIGTERM lets a running clip finish |
| `MAX_BODY_BYTES` | `262144` | request body limit |

A bad value stops the server at start-up (exit 2) naming the variable but never printing its value.

## Failure modes

| Symptom | Likely cause |
|---|---|
| Agent exits 5 right after starting | the model server never became ready within 20 minutes; check `journalctl -u ltx-server` for `load_failed` or a slow load |
| `load_failed error_type=FileNotFoundError` | a weight file is missing or named differently: notes item 2 |
| `load_failed error_type=ImportError`/`TypeError` | the instance's LTX differs from the one the binding was written against: notes items 1, 3, 5 |
| the unit restarts every ~30 s | `Restart=on-failure`; read the last `event=` lines; `start request repeated too quickly` means the 10-per-30-minutes limit was hit |
| clips with no audio | notes items 9 and 15 |
| `503 model is still loading` from the agent's side | the agent normally waits; only a hand-made request sees it |
