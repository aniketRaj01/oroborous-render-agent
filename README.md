# oroborous-render-agent

The render agent for Oroborous: a small Python service that runs on a GPU instance, asks a job API for video clips to render, renders them with a local model server, uploads them to S3 and reports the result.

> **Status:** scaffold only. No code yet.

This repository is public. It holds no secrets and no environment-specific identifiers; those are supplied at runtime (see `CLAUDE.md`).

## Layout

| Path | Purpose |
|---|---|
| `src/render_agent/` | The agent package |
| `tests/` | Tests (fakes for the job API, model server and S3) |
| `deploy/` | systemd units and boot-time bootstrap |
| `scripts/` | Developer helpers |
| `docs/` | Design notes and the instance runbook |

## Working on this repo

Read [`CLAUDE.md`](CLAUDE.md) first. It defines the stack, the rules for never committing secrets or identifiers, the agent's behaviour requirements and the PR workflow.
