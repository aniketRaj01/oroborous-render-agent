from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import httpx

from render_agent.clock import Clock
from render_agent.config import Settings
from render_agent.job_api.client import create_job_api_client
from render_agent.model_server.client import ModelServer, create_model_server
from render_agent.runner import Runner, RunResult
from render_agent.scratch import prepare_scratch_dir
from render_agent.uploader import S3Uploader, Uploader


def run_agent(
    settings: Settings,
    *,
    started_at: datetime,
    clock: Clock,
    should_stop: Callable[[], bool],
    max_clips: int | None,
    job_api_transport: httpx.BaseTransport | None = None,
    model_server: ModelServer | None = None,
    uploader: Uploader | None = None,
) -> RunResult:
    """Build the real collaborators (unless given) and run the loop. Raises ConfigError."""
    prepare_scratch_dir(settings.scratch_dir)
    runner = Runner(
        settings=settings,
        job_api=create_job_api_client(settings, clock=clock, transport=job_api_transport),
        model_server=model_server or create_model_server(settings),
        uploader=uploader or S3Uploader(),
        clock=clock,
        should_stop=should_stop,
        max_clips=max_clips,
    )
    return runner.run(started_at)
