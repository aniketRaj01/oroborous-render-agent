# Oroborous Render Agent — CLAUDE.md

Guardrails and conventions for anyone (human or agent) working in this repo. Read this before writing code.

**This repository is PUBLIC.** Everything committed here is visible to the world, forever (git history included). Read the "Public repo: what must never be committed" section before every commit.

## What this is

The render agent is a small, long-running Python process that runs on a GPU EC2 instance (g6e, us-east-1) that is switched on for about an hour a day. It is the bridge between the Oroborous content pipeline and the local video model:

1. Ask the **Render Agent Job API** (an AWS Lambda behind an IAM-authenticated Function URL, in the `oroborous-account` repo) for clips to render.
2. Render each clip, one at a time, by calling the **local model server** (a separate process on the same machine that keeps the video model loaded in memory).
3. Upload each finished clip to S3.
4. Tell the Job API the clip is done (or failed).
5. Stop when there is no work left or when its deadline passes.

It contains no content logic, no prompts, no personas and no pipeline decisions. It does what the Job API tells it to, and reports what happened.

## What is out of scope for this repo (for now)

- The model server itself (loading weights, GPU inference). The agent talks to it through a narrow interface over localhost HTTP, and tests use a fake implementation. Where the model server will live is not decided yet; do not add it here without asking.
- Booting and stopping the instance (separate Lambdas in `oroborous-account`).
- Joining clips, narration, captions, publishing (the existing pipeline does these).
- The Job API Lambda. Its contract is a private Google Doc, "Oroborous: GPU Render Agent Job API Contract". The user gives you the link in the session prompt. Treat it as the source of truth for request/response shapes, error codes, lease and retry rules. Do not copy the doc's link into this repo, and if the contract and your assumptions disagree, stop and ask.

## Gathering context (do this first, every new session)

1. Read this file and `README.md`.
2. Read the Job API contract doc (link provided in your prompt).
3. Run `git log --oneline` and `git branch -a` to see what is really merged. Trust the repo over anyone's memory of it.

## Public repo: what must never be committed

Never commit, paste, log, or put in a test fixture, a comment, a docstring, a commit message, a PR description or an issue:

- API keys, tokens, passwords, AWS access keys or session tokens, private keys, `.pem` files.
- The Job API **Function URL**. It is the entry point to a private system.
- AWS account ids, ARNs, IAM role names, S3 bucket names, SSM parameter paths that reveal real names, database URLs, instance ids, IP addresses, security group ids.
- Real persona names, account handles, prompts, or content from the pipeline.
- Anything copied from the private Google Docs beyond the short, generic facts needed to write the code.

How sensitive configuration reaches the agent at runtime:

- **On the instance:** read from SSM Parameter Store at start-up (instance role, no keys), or from a root-owned file on the instance disk (mode 600, outside the repo and outside any directory this repo is checked out into), for example `/etc/render-agent/agent.env`. The agent must work with either source, selected by configuration, and must never print the values.
- **Locally:** a `.env` file in the repo root, which is gitignored. `.env.example` lists variable *names* with obviously fake placeholder values (`https://example.invalid`, `changeme`), nothing real.
- Config is validated at start-up with a typed settings model (pydantic-settings). A missing or malformed value fails immediately with a message that names the variable but never echoes its value.
- Logging: never log request headers, signed URLs, environment dumps or full config objects. Redact anything that looks like a URL host for the Job API, and any credential. A `repr()` of the settings object must not leak secrets (use `SecretStr` / `repr=False` for them).
- No `print` debugging of config or responses that contain URLs or tokens.

Before every commit: `git diff --staged` and read it. If you are about to paste a real credential or URL into a file, stop. If a secret is ever committed by mistake, do not just delete it in a later commit: tell the user immediately, because the secret must be rotated and the history rewritten.

A pre-commit setup (ruff, mypy, plus a secret scanner such as gitleaks) must exist from the first PR and may not be bypassed (`--no-verify` is forbidden).

## Configuration

Settings are a typed `pydantic-settings` model (`src/render_agent/config.py`), validated at start-up. Sources, highest precedence first:

1. The process environment.
2. A dotenv-style config file whose path is in `RENDER_AGENT_CONFIG_FILE` (for example `/etc/render-agent/agent.env`). If it is readable by group or others, start-up fails.
3. A local `.env` (gitignored) in the working directory.
4. SSM Parameter Store under the prefix in `RENDER_AGENT_SSM_PREFIX`, read with `get_parameters_by_path` (`WithDecryption`) in the region from `RENDER_AGENT_SSM_REGION`. The last path segment maps to a setting (`.../job-api-url` is `JOB_API_URL`); unknown names are ignored.

The Job API URL is stored in an SSM SecureString (`/oroborous/render-agent/job-api-url`, region via `RENDER_AGENT_SSM_REGION`) or in the root-only file `/etc/render-agent/agent.env`, and locally in `.env`. Never in the repo.

Rules: `JOB_API_URL` must be https (http only for a loopback host); `JOB_API_SIGNING=none` is accepted only for loopback hosts; `AGENT_ID` is required, has no random default, and must be stable across restarts because the Job API resumes jobs only for the same agent id. Validation errors list setting names and error types, never values.

## Operating decisions for the run loop

Implemented in `src/render_agent/runner.py`; run with `python -m render_agent run [--max-clips N]`. The model server's HTTP contract is in `docs/model-server-contract.md`.

- The work list is **in memory**, filled from `/v1/claim` (`deadlineAt` = the run deadline, `maxPlans` = `MAX_PLANS_PER_CLAIM`, default 10). The Job API is the durable queue; there is no local queue file. Plans are worked in the order returned, clips by position.
- `AGENT_ID` is stable across restarts, so a restarted agent resumes its own jobs. A SIGTERM or SIGINT stops cleanly after the current step: the agent does not call `fail` for an in-flight render (the same agent id resumes it), removes the scratch file and exits 0. An upload followed by its `complete` call counts as one step.
- The first claim waits until the model server reports healthy (`/health` polled every `MODEL_SERVER_HEALTH_POLL_SECONDS`, default 5, for up to `MODEL_SERVER_READY_TIMEOUT_MINUTES`, default 20; exit 5 if it never is).
- When the in-memory list is empty, claim again at once. An empty claim means wait `EMPTY_CLAIM_WAIT_SECONDS` (60) and retry. Exit 0 after `IDLE_EXIT_MINUTES` (5) of consecutive empty claims; finding work resets the timer. A claim that fails after the client's retries counts as an empty one.
- Never start a clip that cannot finish before the run deadline (`RUN_DEADLINE_MINUTES`, default 55, measured from process start). The estimate is the mean of the last 5 clip times (start to complete) times 1.25, or `CLIP_TIME_ESTIMATE_SECONDS` (300) before any exist. No claim is made when a clip would not fit. Claimed clips that are not started are left to lease expiry and counted as skipped.
- Per clip: `start`, render into `SCRATCH_DIR/<jobId>-<attempt>.mp4`, verify (exists, non-empty, `ftyp` at offset 4), upload to the bucket, region and key from the claim, `complete` with the verified size, delete the local file (always, success or failure).
- Failure mapping to the Job API: model server `oom` is `oom`; any other model server error, a generate timeout or an unreachable server is `render_error`; verification problems are `verify_failed`; upload errors are `upload_error`. A failed clip never kills the run.
- **Consecutive-failure rule:** after `MAX_CONSECUTIVE_FAILURES` (3) clip failures in a row (any completed clip resets the count) the run stops with exit 5. A model server that is unreachable mid-run counts as a clip failure.
- `start` errors (`max_attempts_reached`, `wrong_state`, `not_your_lease`, `lease_expired`, `job_not_found`) drop the job: it is not failed, and it is ignored for the rest of the run even if a later claim returns it. `complete` errors: `object_missing` or `size_mismatch` re-upload once and retry once, else the job is failed as `upload_error`; `key_mismatch` is logged loudly and the job failed as `verify_failed`; `not_your_lease` and `lease_expired` drop the job. Transient `complete` failures are retried with backoff for up to `COMPLETE_RETRY_BUDGET_MINUTES` (10), never past the run deadline, then logged and the job given up.
- **Chained clips** (optional claim fields `startImage` and `lastFrameUpload`, optional `lastFrame` on
  `complete`; all `{bucket, region, key}` / `{bucket, key, sizeBytes}`, PNG files, keys chosen by the API):
  before rendering a clip with `startImage` the agent reuses the previous clip's local frame, or downloads
  it into `SCRATCH_DIR` and sends it to the model server as `conditionImagePath`; a missing start frame
  fails the clip (`render_error`), it is never rendered from text alone. After a clip with `lastFrameUpload`
  is verified, the agent extracts its last frame with ffmpeg, uploads it as `image/png`, and sends
  `lastFrame` with `complete`; a failure there is an `upload_error` for the clip. After a chained clip fails
  or is given up, later chained clips of that plan are not started in this run.
- A Job API auth error (HTTP 403) is systemic: exit 3 at once. Other unexpected errors exit 4 with a logged traceback.
- Exit codes: 0 normal end (idle, deadline, `--max-clips`, signal), 2 configuration error, 3 Job API auth error, 4 unexpected error, 5 systemic failure (model server never ready, or too many consecutive failures). `--max-clips N` stops after N clips have finished (done or failed).
- Logs: one line per event (job id, plan id, position, attempt, durations, outcome) and a final `run_summary` line (done, failed, skipped, dropped, runtime, reason). Prompts are never logged.

## Locked stack decisions: do not relitigate without discussion

- **Language:** Python `>=3.10` (CI runs 3.10 and 3.12). Confirm against the Python version on the GPU instance and document the choice in the README. Use `from __future__ import annotations`; no 3.11-only features. No Python 2-isms, no untyped code.
- **Packaging:** `pyproject.toml`, `src/` layout (`src/render_agent/`), one installable package. Dependencies pinned with a lock file. Use `uv` unless the first PR gives a concrete reason not to.
- **Dependencies are minimal and justified.** Expected: `pydantic` v2 and `pydantic-settings` for models and config, `boto3` for S3, SSM and SigV4 signing, one HTTP client (`httpx`). Anything else needs a one-line reason in the PR. No frameworks (no FastAPI, no Celery, no asyncio task frameworks, no agent frameworks). The agent is a simple loop. There is no web framework and the agent opens no listener: it only makes outbound calls (Job API, localhost model server, S3, SSM).
- **Concurrency:** the GPU renders one clip at a time. Keep the main loop sequential and simple. A background upload thread is allowed only if it is shown to matter; the default is render, then upload, then complete.
- **Typing:** `mypy --strict` clean. Pydantic models at every external boundary: Job API responses, model-server responses, config. No bare `dict[str, Any]` flowing through business logic.
- **Lint/format:** `ruff` (lint and format). Enforced pre-commit and in CI. Do not hand-format around it.
- **Testing:** `pytest`, written alongside the code (unlike the older Node repo). No network, no AWS and no GPU in unit tests: use fakes for the Job API, the model server and S3 (an in-process fake or `moto`). A separate, clearly marked, opt-in integration check may run against a locally served Job API.
- **Logging:** the standard `logging` module with structured, one-line records (job id, plan id, clip position, attempt, durations). No secrets (see above). Log enough to debug a failed run from the instance's journal alone.

## Folder structure

```
src/render_agent/   # the package: config, job API client, model-server client, S3 uploader, the run loop, entry point
tests/              # mirrors src/render_agent/, plus fakes under tests/fakes/
deploy/             # systemd unit files, boot-time bootstrap script, install notes: what gets copied onto the instance
scripts/            # developer helpers (run the agent against local fakes, etc.), not part of the deployed agent
docs/               # design notes, runbook for the instance. No secrets, no real identifiers.
```

Keep the entry point thin: it should read like a table of contents (load config, build clients, run the loop, exit with the right code). The behaviour lives in modules that take their dependencies as arguments, so they are testable with fakes.

## Behaviour rules the agent must keep

These come from the operating model, so they are not optional:

- **Never loses a job silently.** Every claimed job ends in exactly one of: completed, failed (reported to the Job API with a reason), or left to lease expiry because the process or instance died. No swallowed exceptions that skip reporting.
- **Idempotent and restart-safe.** The process can be killed at any moment (it will be: the instance is stopped daily). On restart it claims again and resumes cleanly. Starting, completing and failing the same job twice must be harmless.
- **Deadline-aware.** The agent has a hard deadline for each run (set by configuration). It must not start a clip it cannot finish before the deadline, and it must exit cleanly instead of being cut off mid-render.
- **One bad clip must not kill the run.** A failed render is reported and the loop moves on; a systematic failure (model server down, repeated identical failures) stops the run with a clear log line instead of burning through every job.
- **Verify before reporting success.** A clip is only reported complete after it has been uploaded and its size checked. Never report done on faith.
- **Local-only model server.** The model server is addressed on localhost. The agent never exposes a network listener of its own.
- **Fail loudly at start-up** on bad configuration; **fail safely at run time** (report, back off, continue or stop deliberately).
- **Time and retries are injectable.** No bare `time.sleep` or `datetime.now()` in logic that needs testing; pass a clock/sleeper.

## Deployment model

Built in `deploy/` and `scripts/`; the operator's procedures are in `docs/instance-runbook.md`.

- The instance gets the agent by pulling this **public** repository over HTTPS at every boot. No credentials are stored for it.
- It runs a **pinned, released version**: the newest git tag `vX.Y.Z` (compared as semantic versions) whose commit is an ancestor of `origin/main` and whose `pyproject.toml` version equals the tag, never whatever is on `main`. `RENDER_AGENT_PIN` runs one exact tag instead (manual rollback). Released versions are immutable: a tag moved to another commit is refused.
- The agent runs as a systemd service (`deploy/systemd/render-agent.service`, restart on failure, no restart on exit codes 2, 3 and 5). `deploy/install.sh` prepares the machine once and `deploy/bootstrap.py` (Python standard library only, run as `ExecStartPre`) prepares the release at each boot. Neither contains secrets or real identifiers: those come from SSM or `/etc/render-agent/agent.env` at runtime. `bootstrap.py` and the units are installed by `install.sh`, not by a release, so a change under `deploy/` reaches an instance only when `install.sh` is run again.
- Python dependencies are **installed at boot, not baked into the image**, from `requirements.lock` (hashes, no dev dependencies, exported from `uv.lock` by `scripts/export_requirements.sh`; CI fails when the committed file is out of date). The virtualenv is rebuilt only when the hash of `requirements.lock` changes. Trade-off: a changed lock needs PyPI at boot, and the bootstrap falls back to the current version when that fails; pre-loading the first release while the image is built (runbook step 4) makes the first boot independent of the network.
- Nothing in this repo may assume the instance has internet access beyond GitHub, PyPI (only when `requirements.lock` changed), and AWS endpoints.

### Release design

- A release is cut with `scripts/release.py vX.Y.Z` (annotated tag, pushed alone, after checking the tag format, the pyproject version, a clean tree, `main` equal to the remote, `requirements.lock` in sync and green CI; `--dry-run` changes nothing).
- The bootstrap keeps a bare mirror, exports a tag to `releases/<tag>` (with its commit in `RELEASE`), builds `venvs/<sha256-12 of requirements.lock>` with `pip install --require-hashes --only-binary=:all: --no-deps` followed by `pip check`, and runs `python -m render_agent selfcheck` in it. Only then does it switch `current` and `current-venv` (atomic symlink replacement) and keep the old pair as `previous` and `previous-venv`. It keeps the 3 newest releases plus `previous`.
- Any failure while preparing a new tag (fetch, tag checks, install, selfcheck, the 5 minute time budget) is logged and the agent starts on the current pair; the bootstrap fails the start only when no usable pair exists.
- `agentVersion` is the tag plus the short SHA (`vX.Y.Z+<12 hex>`), written to `/run/render-agent/version.env` as `AGENT_VERSION` and sent with every claim.

## Code style guardrails

- No premature abstraction. Three similar lines beats a speculative interface. Exception: the model server is accessed through one small interface because tests need a fake.
- No comments explaining *what* code does; names should do that. Comment only a non-obvious *why*: a workaround, a platform quirk, a constraint from an external system.
- Fail loudly. Do not catch broad exceptions just to log and continue. Catch specific exceptions, at the place that can do something about them (report the failure, retry, stop).
- Every external call (Job API, model server, S3, SSM) goes through a typed client module. No raw HTTP or boto3 calls scattered in the run loop.
- Retries with backoff and a cap, on the specific errors that are retryable (network, 5xx, throttling), never on validation or 4xx contract errors. Respect the Job API's error codes (`not_your_lease`, `lease_expired`, `max_attempts_reached` and the like) with the behaviour the contract defines.

## PR workflow

- Never commit directly to `main`. Branch, open a PR, keep PRs small and independently testable. Branch naming: `short-kebab-description`.
- A PR description states what changed, how it was verified (commands and output), and anything not verified. It must not contain secrets or real identifiers.
- Commits are small and in the imperative. Use the attribution trailer the harness gives you.
- Resist bundling "while I'm in here" cleanups into a feature PR.

## What a worker session may and may not do

- **No real AWS calls**, no real S3, SSM or Lambda, and no calls to the deployed Job API, from a development session. Use fakes, `moto`, and the local Job API harness the user points you to. The user performs real-environment tests and deployments.
- **No GPU or paid external services** are called from here. The agent has no reason to talk to Pollinations, ElevenLabs, Polly or any LLM API, so none of those clients belongs in this repo.
- **No deployments**, no AMI work, no instance changes, no changes to IAM. If a task seems to need one, write down what is needed in the PR and stop.
- Do not edit docs or tracking sheets outside this repo.

## When in doubt

If a task seems to require deviating from any of the above (a new dependency, a framework, a new AWS service, storing something sensitive, a change to how the agent is deployed), say so explicitly and explain why before doing it, rather than silently working around this document.
