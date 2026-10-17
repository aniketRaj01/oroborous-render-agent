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

Compare `OFFLOAD_MODE=cpu` (default) with the alternatives by setting it in
`/etc/render-agent/ltx-server.env` and `sudo systemctl restart ltx-server` between runs.

| What the measurement shows | Meaning | What to change |
|---|---|---|
| Clip 2 is clearly faster than clip 1 (say more than 25 percent), the floor was never breached, peak RSS is comfortably under the 55 GB `MemoryMax` | weights are being reused and fit | Keep the defaults. Done. |
| Clip 2 is about as slow as clip 1 (under 10 percent faster) | the weights are being read again for every clip | Confirm `registry=true` in the `building_pipeline` log line (`USE_REGISTRY=1`, and the constructor accepted it: notes items 5, 6, 10). If it is on and clip 2 is still slow, the cache is not reaching the stage that matters; try `OFFLOAD_MODE=none` if GPU memory allows (peak GPU is ~39.6 of 46 GB, so probably not) and otherwise accept the per-clip disk read; the operating system's page cache will help partly. |
| `MemAvailable` fell below the floor, or `journalctl -u ltx-server` shows the unit killed by the memory limit | the RAM cache is too big for this machine | In this order: `USE_REGISTRY=0` (no weight cache; every clip re-reads from disk), then `OFFLOAD_MODE=disk` (lowest RAM, slowest). Re-measure after each. Do **not** raise `MemoryMax` above about 57G: the OS and the SSM agent need the rest. |
| The unit is killed at exactly `MemoryMax` but `MemAvailable` never got low | the cgroup cap is lower than the working set, the machine has room | Raise `MemoryMax` in a drop-in (`systemctl edit ltx-server`) in 1 GB steps, staying under about 57G. |
| Clip 1 is much slower than clip 2 only because the first pass reads the disk | normal, the warm-up should absorb it | Make sure `WARMUP=1` (default). Then clip 1 should already be close to clip 2, and `ready` takes longer instead. If `ready` is too slow for the agent's 20 minute wait, raise `MODEL_SERVER_READY_TIMEOUT_MINUTES` on the agent. |
| `generate_failed kind=oom` on a normal clip, or `too_many_oom_exiting` | GPU memory, not RAM | Do not change RAM settings. Check `peak GPU memory`; try `OFFLOAD_MODE=cpu` if it was `none`; shorten `MAX_FRAMES`/the clip; look at notes item 14 (a GPU memory leak shows as the GPU column growing clip after clip). |
| The warm-up itself fails with out of memory (`load_failed`) | the machine is too small for the warm-up size or the settings | `WARMUP=0`, or smaller `WARMUP_WIDTH/HEIGHT/FRAMES`; then re-check the first real clip. |

After deciding, record the choice (and the numbers) in the PR or a note next to the instance image, and
leave the other settings alone.

## Settings

All environment variables, with defaults (set them in `/etc/render-agent/ltx-server.env`):

| Variable | Default | Purpose |
|---|---|---|
| `HOST`, `PORT` | `127.0.0.1`, `8000` | bind address (loopback only, anything else is refused) |
| `RENDERER` | `ltx` | `ltx` or `fake` |
| `LTX_ROOT`, `MODELS_SUBDIR` | `~/LTX-2`, `models/ltx-2.5` | where the weights are |
| `OFFLOAD_MODE` | `cpu` | `cpu`, `disk` or `none` |
| `QUANTIZATION` | `fp8-cast` | LTX quantization kind, or `none` |
| `USE_REGISTRY` | `1` | cache checkpoint tensors in RAM (`registry=` of the pipeline) |
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
