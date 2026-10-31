from __future__ import annotations

import platform
from datetime import timedelta
from typing import TextIO

from render_agent.clock import Clock
from render_agent.config import Settings
from render_agent.frames import ffmpeg_available
from render_agent.job_api.client import JobApi

DEFAULT_LEASE_MINUTES = 2


def write_summary(out: TextIO, settings: Settings, sources: tuple[str, ...]) -> None:
    out.write("render-agent check\n")
    out.write(f"  python:        {platform.python_version()}\n")
    out.write(f"  agent id:      {settings.agent_id}\n")
    out.write(f"  agent version: {settings.agent_version}\n")
    out.write(f"  signing:       {settings.job_api_signing}\n")
    out.write(f"  region:        {settings.region or 'n/a'}\n")
    out.write(f"  job api host:  {settings.host_summary()}\n")
    out.write(f"  config from:   {', '.join(sources) if sources else 'none'}\n")


def run_claim(
    out: TextIO,
    settings: Settings,
    client: JobApi,
    clock: Clock,
    *,
    lease_minutes: int,
) -> None:
    deadline_at = clock.now() + timedelta(minutes=lease_minutes)
    response = client.claim(deadline_at=deadline_at, max_plans=1)
    out.write("\nclaim ok\n")
    out.write(f"  server time:      {response.server_time:%Y-%m-%dT%H:%M:%SZ}\n")
    out.write(f"  lease expires at: {response.lease_expires_at:%Y-%m-%dT%H:%M:%SZ}\n")
    out.write(f"  totals:           plans={response.totals.plans} clips={response.totals.clips}\n")
    for plan in response.plans:
        out.write(
            f"  plan {plan.plan_id}: clips {len(plan.clips)} of {plan.clips_total} to render "
            f"({plan.clips_already_done} already done)\n"
        )
        for clip in plan.clips:
            out.write(
                f"    job {clip.job_id} position {clip.position} "
                f"attempts {clip.attempts} resumed {str(clip.resumed).lower()}"
                f"{' chained' if clip.is_chained else ''}\n"
            )
    if any(clip.is_chained for plan in response.plans for clip in plan.clips) and (
        not ffmpeg_available()
    ):
        out.write(
            "\nWARNING: chained clips need ffmpeg to extract last frames and it was not found "
            "(apt install ffmpeg).\n"
        )
    if response.totals.clips:
        out.write(
            f"\nWARNING: {response.totals.clips} job(s) are now leased to {settings.agent_id} "
            f"until {response.lease_expires_at:%Y-%m-%dT%H:%M:%SZ}. They are not rendered and "
            "will not be claimable by another agent until then (a restart with the same agent "
            "id reclaims them at once).\n"
        )
    else:
        out.write("\nNothing was leased (no work available).\n")
