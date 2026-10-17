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

## Locked stack decisions: do not relitigate without discussion

- **Language:** Python 3.11+. Confirm against the Python version on the GPU instance before pinning, and document the choice in the README. No Python 2-isms, no untyped code.
- **Packaging:** `pyproject.toml`, `src/` layout (`src/render_agent/`), one installable package. Dependencies pinned with a lock file. Use `uv` unless the first PR gives a concrete reason not to.
- **Dependencies are minimal and justified.** Expected: `pydantic` v2 and `pydantic-settings` for models and config, `boto3` for S3, SSM and SigV4 signing, one HTTP client (`httpx`). Anything else needs a one-line reason in the PR. No frameworks (no FastAPI, no Celery, no asyncio task frameworks, no agent frameworks). The agent is a simple loop.
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

## Deployment model (constraints; the design happens in the first session)

- The instance gets the agent by pulling this **public** repository over HTTPS at boot. No credentials are stored for it.
- It must run a **pinned, released version** (a tag or commit that was deliberately released), not whatever is on `main` at that moment, and it must fall back to the last known-good checkout if the pull fails. The release mechanism (tag naming, pointer) is designed in a PR, not improvised.
- The agent runs as a systemd service with a restart policy; the bootstrap script and unit files live in `deploy/`. They contain no secrets and no real identifiers: those come from SSM or the on-disk config file at runtime.
- Python dependencies must install reproducibly from the lock file. Decide in a PR whether they are baked into the machine image or installed at boot, and document the trade-off.
- Nothing in this repo may assume the instance has internet access beyond GitHub, PyPI (if used at boot), and AWS endpoints.

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
