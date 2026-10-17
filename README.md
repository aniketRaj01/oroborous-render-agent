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
| `docs/` | Design notes and the instance runbook |

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

## Working on this repo

Read [`CLAUDE.md`](CLAUDE.md) first. It defines the stack, the rules for never committing secrets or identifiers, the agent's behaviour requirements and the PR workflow.
