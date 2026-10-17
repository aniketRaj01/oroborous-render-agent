"""Demo: the real run loop against fakes, on a fake clock. No network, AWS, GPU or model.

Scenario (two process lifetimes of the same agent id):
  plans 101 (2 clips), 102 (2 clips), 103 (1 clip)
  run 1  one render failure (101/1), one out-of-memory (102/0), then the deadline arrives and
         a clip that cannot finish in time is skipped, not started
  run 2  a restart: the skipped clip comes back as "resumed", then the queue stays empty and
         the agent exits on the idle limit

Usage: uv run python scripts/demo_fake_run.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from fakes.clock import FakeClock
from fakes.fake_job_api import FakeJobApi
from fakes.fake_model_server import FakeModelServer, Reject
from fakes.fake_uploader import FakeUploader
from render_agent.config import build_settings
from render_agent.job_api.client import JobApiClient, RetryPolicy
from render_agent.job_api.signing import UnsignedSigner
from render_agent.logging_setup import configure_logging
from render_agent.runner import Runner, RunResult
from render_agent.scratch import prepare_scratch_dir

AGENT_ID = "demo-agent"
FAKE_URL = "https://fake-id.lambda-url.us-east-1.on.aws"


class FakeTimeFilter(logging.Filter):
    """Stamp log records with the fake clock so the timeline reads like a real run."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self._clock = clock

    def filter(self, record: logging.LogRecord) -> bool:
        record.created = self._clock.now().timestamp()
        return True


def make_runner(
    settings_env: dict[str, str], clock: FakeClock, api: FakeJobApi, model: FakeModelServer
) -> Runner:
    settings = build_settings(settings_env)
    client = JobApiClient(
        base_url=FAKE_URL,
        agent_id=AGENT_ID,
        agent_version="demo",
        http=httpx.Client(transport=api.transport),
        signer=UnsignedSigner(),
        clock=clock,
        retry=RetryPolicy(max_retries=4, base_delay=0.5, max_delay=10, jitter=0.5),
    )
    return Runner(
        settings=settings,
        job_api=client,
        model_server=model,
        uploader=FakeUploader(api, clock),
        clock=clock,
    )


def banner(text: str) -> None:
    print(f"\n=== {text} ===", flush=True)


def show(result: RunResult) -> None:
    print(
        f"-> exit={result.exit_code} reason={result.reason} done={result.done} "
        f"failed={result.failed} skipped={result.skipped} dropped={result.dropped} "
        f"runtime={result.runtime_seconds:.0f}s",
        flush=True,
    )


def main() -> None:
    clock = FakeClock()
    configure_logging(stream=sys.stdout)
    logging.getLogger().handlers[0].addFilter(FakeTimeFilter(clock))
    logging.getLogger("render_agent.job_api").setLevel(logging.WARNING)  # keep the demo short

    api = FakeJobApi(clock)
    plans = [api.add_plan(101, clips=2), api.add_plan(102, clips=2), api.add_plan(103, clips=1)]
    model = FakeModelServer(clock, render_seconds=240.0)
    model.script(plans[0].jobs[1].job_id, Reject("render_error", "decoder crashed"))
    model.script(plans[1].jobs[0].job_id, Reject("oom", "CUDA out of memory"))

    scratch = tempfile.mkdtemp(prefix="render-agent-demo-")
    env = {
        "JOB_API_URL": FAKE_URL,
        "AGENT_ID": AGENT_ID,
        "SCRATCH_DIR": scratch,
        "RUN_DEADLINE_MINUTES": "20",
        "IDLE_EXIT_MINUTES": "3",
    }
    prepare_scratch_dir(Path(scratch))

    banner("run 1: deadline 20 min, 5 clips queued (render 240 s + upload 10 s each)")
    show(make_runner(env, clock, api, model).run(clock.now()))

    clock.advance(60)
    banner("run 2: the agent restarted with the same agent id (1 minute later)")
    show(make_runner(env, clock, api, model).run(clock.now()))

    banner("final job states")
    for plan in plans:
        for job in plan.jobs:
            print(
                f"plan {plan.plan_id} position {job.position}: {job.status} "
                f"(attempts {job.attempts}, last error {job.last_fail_code or 'none'})"
            )


if __name__ == "__main__":
    main()
