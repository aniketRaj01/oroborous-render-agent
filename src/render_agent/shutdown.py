"""Power the instance off when the agent has run out of work (opt-in, AUTO_SHUTDOWN_ON_IDLE).

The agent never powers anything off itself: it runs unprivileged with NoNewPrivileges. After an
idle exit it writes a marker file; render-agent.service then runs deploy/poweroff-if-requested.sh
as root (ExecStopPost, "+" prefix), which powers off only when that marker exists. The operating
system poweroff makes EC2 stop the instance (its shutdown behaviour is "stop"); no AWS API is used.
Only the idle exit (or the optional MAX_RUN_MINUTES end) gets here: every other end of a run
(signal, --max-clips, errors) never writes it.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path

from render_agent.clock import Clock
from render_agent.config import Settings
from render_agent.model_server.client import ModelServer
from render_agent.runner import REASON_IDLE, REASON_MAX_RUN, RunResult
from render_agent.scratch import unfinished_output_files

logger = logging.getLogger("render_agent")

NO_SHUTDOWN_FILE = Path("/etc/render-agent/no-shutdown")
POWEROFF_MARKER = Path("/run/render-agent/poweroff-requested")

# Outcomes, returned so callers and checks can tell what happened.
DISABLED = "disabled"
BOOTSTRAP_FALLBACK = "bootstrap_fallback"
GUARD_FILE = "guard_file"
NOT_IDLE = "not_idle"
WORK_IN_FLIGHT = "work_in_flight"
STOP_REQUESTED = "stop_requested"
REQUESTED = "requested"
REQUEST_FAILED = "request_failed"


def write_marker(path: Path = POWEROFF_MARKER) -> None:
    path.write_text("idle\n", encoding="utf-8")


def clear_marker(path: Path = POWEROFF_MARKER) -> None:
    """A marker left by an earlier run must never power off the next one."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers + logger.handlers:
        handler.flush()


def request_poweroff_after_idle(
    result: RunResult,
    *,
    settings: Settings,
    model_server: ModelServer,
    clock: Clock,
    should_stop: Callable[[], bool],
    request: Callable[[], None] = write_marker,
    guard_file: Path = NO_SHUTDOWN_FILE,
) -> str:
    if result.reason not in (REASON_IDLE, REASON_MAX_RUN):
        return NOT_IDLE
    if not settings.auto_shutdown_on_idle:
        return DISABLED
    if settings.bootstrap_fallback:
        logger.warning("idle_exit_staying_up", extra={"why": "the bootstrap fell back"})
        return BOOTSTRAP_FALLBACK
    if guard_file.exists():
        logger.info("idle_exit_staying_up", extra={"why": "no-shutdown file present"})
        return GUARD_FILE
    problems = _in_flight(result, settings, model_server)
    if problems:
        logger.warning("idle_exit_staying_up", extra={"why": f"work may be in flight: {problems}"})
        return WORK_IN_FLIGHT
    if should_stop():
        return STOP_REQUESTED

    logger.info("idle exit: powering off instance")
    _flush_logs()
    clock.sleep(settings.auto_shutdown_grace_seconds)
    if should_stop():
        logger.info("idle_exit_staying_up", extra={"why": "stop requested during the grace wait"})
        return STOP_REQUESTED
    if guard_file.exists():
        logger.info("idle_exit_staying_up", extra={"why": "no-shutdown file appeared"})
        return GUARD_FILE
    try:
        request()
    except OSError as exc:
        logger.error("poweroff_request_failed", extra={"error_type": type(exc).__name__})
        return REQUEST_FAILED
    _flush_logs()
    return REQUESTED


def _in_flight(result: RunResult, settings: Settings, model_server: ModelServer) -> str:
    """Idle implies nothing is in flight; say so loudly if that is not what we see."""
    found = []
    if result.reason == REASON_IDLE and result.skipped:
        # The work list is empty at an idle exit. (MAX_RUN_MINUTES may leave claimed clips to
        # lease expiry on purpose; the next boot claims them again.)
        found.append("claimed clips were left unstarted")
    if unfinished_output_files(settings.scratch_dir):
        found.append("a clip output file is still in the scratch directory")
    if not model_server.is_ready():
        found.append("the model server is not ready")
    return ", ".join(found)
