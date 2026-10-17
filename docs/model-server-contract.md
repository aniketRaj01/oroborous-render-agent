# Model server contract (agent side: implemented; server side: to be built)

The render agent talks to a **model server** on the same machine. The model server keeps the video
model loaded in GPU memory and renders one clip per request. This document is the spec for the
future model-server implementation; the agent side (`src/render_agent/model_server/`) already
implements it, and `scripts/fake_model_server.py` is a stand-in that follows it.

## Ground rules

- **Loopback only.** The server listens on `127.0.0.1` (or `::1`). The agent refuses a
  `MODEL_SERVER_URL` that is not a loopback address. No authentication: the network boundary is the
  machine. The server must not bind to any other interface.
- **Same machine, same user, same disk.** No video bytes travel over HTTP. The agent tells the server
  where to write the file; the agent's scratch directory is mode `0700` and owned by the agent's
  user, so the server must run as that user.
- **One generation at a time.** The agent is strictly sequential: it never sends a second
  `/generate` while one is outstanding. The server should still enforce it (see `busy`).
- **JSON.** Bodies are UTF-8 JSON, `content-type: application/json`. Field names are camelCase.
  Unknown request fields are rejected with `invalid_request`; the agent ignores unknown response
  fields.
- **The prompt is opaque.** It is the exact text from the content pipeline. The server must not
  change it, and should not write it to logs.

## GET /health

```
200 {"ok": true, "modelLoaded": true}
```

The server is **ready** only when `modelLoaded` is `true`. While the model is still loading the
server may answer `200` with `modelLoaded: false`, answer with a non-200 status, or not accept
connections yet; the agent treats all three as "not ready yet". It polls every
`MODEL_SERVER_HEALTH_POLL_SECONDS` (default 5 s, each request has a 5 s timeout) and gives up after
`MODEL_SERVER_READY_TIMEOUT_MINUTES` (default 20), exiting with code 5. The agent makes its first
Job API claim only after the server is ready.

## POST /generate

Synchronous: the response is sent when the clip has been written.

Request:

```json
{
  "jobId": 412,
  "prompt": "exact prompt text",
  "params": { "width": 576, "height": 1024, "numFrames": 361, "fps": 24, "seed": 42 },
  "outputPath": "/tmp/render-agent/412-1.mp4"
}
```

| Field | Rule |
|---|---|
| `jobId` | Integer. Informational (logging, tracing). The server must not cache results by it. |
| `prompt` | String. Render exactly this. |
| `params` | Object containing **only the keys the job has**: any of `width`, `height`, `numFrames`, `fps`, `seed`. An **empty object** when the job has no stored parameters. The server applies its own defaults for every missing key. |
| `outputPath` | Absolute path ending in `.mp4`, inside the agent's scratch directory. The **server writes the file here**. |

Success, `200`:

```json
{
  "ok": true,
  "outputPath": "/tmp/render-agent/412-1.mp4",
  "render": {
    "renderSeconds": 142.6,
    "width": 576, "height": 1024, "numFrames": 361, "fps": 24,
    "durationSeconds": 15.04, "hasAudio": true
  }
}
```

- `outputPath` must equal the requested path exactly; anything else is treated as a failed render.
- `render` describes the clip actually produced (not the request). `renderSeconds` is GPU render
  time. The agent forwards `render` to the Job API when it completes the job.
- The file must be **complete and closed before the response is sent**. Write it to a temporary
  name in the same directory and rename it, so a crash never leaves a half-written `.mp4` under the
  final name.
- The agent verifies the file before uploading: it must exist, be non-empty, and have an `ftyp` box
  at byte offset 4 (a valid MP4/ISO-BMFF header). Otherwise the clip fails as `verify_failed`.
- The agent deletes the file after the clip (success or failure). The server must neither keep nor
  reuse it.

### Errors

```json
{ "ok": false, "error": { "code": "oom", "message": "CUDA out of memory ..." } }
```

| HTTP | `code` | Meaning | Agent reports to the Job API as |
|---|---|---|---|
| 500 | `oom` | GPU or host memory exhausted | `oom` |
| 500 | `render_error` | Any other failure while rendering | `render_error` |
| 400 | `invalid_request` | Body is not valid JSON or fails validation, `outputPath` not acceptable | `render_error` |
| 409 | `busy` | A generation is already running; the running one is not disturbed | `render_error` |

Any other response (non-JSON body, wrong shape, unknown status) is a protocol error and is reported
as `render_error`. `message` should be short and must not contain the prompt. Unknown routes answer
`404` with the same envelope and code `invalid_request`.

## Timeouts and failure behaviour

- The agent waits up to `MODEL_SERVER_GENERATE_TIMEOUT_SECONDS` (default 1200 s) for `/generate`.
  On timeout it closes the connection and reports `render_error`. **The server should stop the
  generation when the client disconnects**; if it cannot, it must answer `busy` until it is free,
  and every such answer counts as a failed clip for the agent's consecutive-failure limit.
- A connection refused or dropped mid-run is a failed clip (`render_error`), not a crash.
- After `MAX_CONSECUTIVE_FAILURES` clip failures in a row (default 3) the agent stops the whole run
  with exit code 5 rather than burn through every job's attempts, so a model server that is down or
  broken stops the run quickly. A server that reliably recovers between requests is not penalised.
- The server may be restarted at any time (the agent will see failed clips, not a crash). It does
  not need to persist anything.

## Compatibility

A breaking change to this contract requires new paths (for example `/v2/generate`), not a silent
change to these.

## Try it without a GPU

```sh
python scripts/fake_model_server.py --port 8000
```

The fake writes a tiny file with a valid-looking `ftyp` box. It is **not** a playable video. See the
docstring for the environment variables that add delay and inject failures.
