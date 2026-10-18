# oroborous-render-agent

The render agent for Oroborous: a small Python service that runs on a GPU instance, asks a job API for video clips to render, renders them with a local model server, uploads them to S3 and reports the result.

This repository is public. It holds no secrets and no environment-specific identifiers; those are supplied at runtime (see `CLAUDE.md`).

## Layout

| Path | Purpose |
|---|---|
| `src/render_agent/` | The agent package |
| `tests/` | Tests (fakes for the job API, model server and S3) |
| `deploy/` | systemd units and boot-time bootstrap |
| `scripts/` | Developer helpers |
| `model_server/` | The model server that renders clips on the GPU instance (standard library shell + LTX binding) |
| `docs/` | Design notes and the instance runbooks |

## Setup

Requires Python 3.10 or newer (CI runs 3.10 and 3.12) and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                      # installs the package and dev tools from uv.lock
uv run pre-commit install    # ruff, ruff-format, mypy, gitleaks, key and file-size checks
cp .env.example .env         # then edit; .env is gitignored and must never be committed
```

## Checks and tests

```sh
uv run pre-commit run --all-files   # everything CI runs for linting, typing and secrets
uv run ruff check . && uv run ruff format --check .
uv run mypy                         # strict
uv run pytest                       # no network, no AWS, no GPU
```

## The `check` command

```sh
uv run python -m render_agent check --dry-run   # validate configuration only, call nothing
uv run python -m render_agent check             # one signed claim (maxPlans=1), see below
```

`check` without `--dry-run` really claims work: any returned clips are leased to `AGENT_ID` until the printed `leaseExpiresAt`. Exit codes: 0 ok, 2 configuration error, 3 Job API or auth error, 4 unexpected error. Configuration sources and precedence are described in the Configuration section of `CLAUDE.md`.

## The `run` command

```sh
uv run python -m render_agent run [--max-clips N]
```

Waits for the local model server, then claims work from the Job API and renders, uploads and reports it clip by clip until the queue is idle, the run deadline arrives, or a systemic failure stops it. Exit codes: 0 normal end, 2 configuration error, 3 Job API auth error, 4 unexpected error, 5 systemic failure. The model server's HTTP contract is in [`docs/model-server-contract.md`](docs/model-server-contract.md); the loop's behaviour and settings are described in `CLAUDE.md` ("Operating decisions for the run loop").

For local experiments without a GPU:

```sh
uv run python scripts/fake_model_server.py        # stdlib-only stand-in; its "videos" are fake
uv run python scripts/demo_fake_run.py            # the real loop against in-process fakes, on a fake clock
```

`run` really claims and completes jobs: only point it at a throwaway plan (see the PR notes for the manual test).

## Deployment

At every boot the GPU instance runs the newest released version (`vX.Y.Z` tag on `main`), installed
reproducibly from `requirements.lock`, self-checked before use, with the last good version as
fallback. The pieces:

| Path | Purpose |
|---|---|
| `deploy/bootstrap.py` | boot-time fetch, verify, install, self-check and atomic switch (standard library only) |
| `deploy/systemd/` | `render-agent.service` and the opt-in dead-man timer |
| `deploy/install.sh` | one-time machine setup (`--root DIR` stages into a directory, `--dry-run` previews) |
| `scripts/release.py` | cuts a release tag after checking every precondition |
| `scripts/export_requirements.sh` | regenerates (or `--check`s) `requirements.lock` from `uv.lock` |
| `python -m render_agent selfcheck` | imports the package and every dependency; no settings, no network |

Setup, releasing, rollback, logs and the first-boot checklist are in
[`docs/instance-runbook.md`](docs/instance-runbook.md).

## Model server

`model_server/` holds `ltx_server`, the local server that keeps the LTX pipeline loaded and renders one
clip per request for the agent. Start with [`docs/model-server-runbook.md`](docs/model-server-runbook.md)
(first test, logs, the RAM measurement and its decision table); the HTTP contract is
[`docs/model-server-contract.md`](docs/model-server-contract.md) and what must be checked against the
real LTX checkout is [`docs/ltx-binding-notes.md`](docs/ltx-binding-notes.md). No GPU is needed to test it:
`RENDERER=fake python -m ltx_server` (with `model_server/` on `PYTHONPATH`) runs it with a fake renderer. `RENDER_MODE=resident` builds each LTX model once and keeps it in memory
(opt-in until verified on the instance; see the runbook's "Render modes").

## Working on this repo

Read [`CLAUDE.md`](CLAUDE.md) first. It defines the stack, the rules for never committing secrets or identifiers, the agent's behaviour requirements and the PR workflow.
