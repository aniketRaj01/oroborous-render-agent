from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from types import FrameType
from typing import TextIO

import httpx

from render_agent.app import run_agent
from render_agent.check import DEFAULT_LEASE_MINUTES, run_claim, write_summary
from render_agent.clock import Clock, InterruptibleClock, SystemClock
from render_agent.config import ConfigError
from render_agent.config_sources import SsmFetch, fetch_ssm_parameters, load_settings
from render_agent.exit_codes import EXIT_API, EXIT_CONFIG, EXIT_OK, EXIT_UNEXPECTED
from render_agent.job_api.client import create_job_api_client
from render_agent.job_api.errors import JobApiError
from render_agent.logging_setup import configure_logging
from render_agent.model_server.client import ModelServer
from render_agent.uploader import Uploader

logger = logging.getLogger("render_agent")


def _lease_minutes(raw: str) -> int:
    value = int(raw)
    if not 1 <= value <= 170:
        raise argparse.ArgumentTypeError("must be between 1 and 170")
    return value


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="render-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check", help="validate configuration and make one signed claim against the Job API"
    )
    check.add_argument(
        "--dry-run", action="store_true", help="validate configuration only, call nothing"
    )
    check.add_argument(
        "--lease-minutes",
        type=_lease_minutes,
        default=DEFAULT_LEASE_MINUTES,
        help="deadline for the check claim, in minutes from now (default: %(default)s)",
    )
    run = commands.add_parser(
        "run", help="claim work, render it with the local model server, upload and report"
    )
    run.add_argument(
        "--max-clips",
        type=_positive_int,
        default=None,
        help="stop after N clips have been finished (done or failed); for manual testing",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    out: TextIO | None = None,
    clock: Clock | None = None,
    transport: httpx.BaseTransport | None = None,
    ssm_fetch: SsmFetch = fetch_ssm_parameters,
    model_server: ModelServer | None = None,
    uploader: Uploader | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    out = out if out is not None else sys.stdout
    configure_logging()
    if args.command == "run":
        return _run(
            args.max_clips,
            environ=environ,
            out=out,
            clock=clock,
            transport=transport,
            ssm_fetch=ssm_fetch,
            model_server=model_server,
            uploader=uploader,
            should_stop=should_stop,
        )
    clock = clock if clock is not None else SystemClock()
    try:
        loaded = load_settings(environ=environ, ssm_fetch=ssm_fetch)
    except ConfigError as exc:
        out.write(f"{exc}\n")
        return EXIT_CONFIG
    try:
        write_summary(out, loaded.settings, loaded.sources)
        if args.dry_run:
            out.write("\ndry run: no request was sent.\n")
            return EXIT_OK
        client = create_job_api_client(loaded.settings, clock=clock, transport=transport)
        run_claim(out, loaded.settings, client, clock, lease_minutes=args.lease_minutes)
    except JobApiError as exc:
        out.write(f"\nJob API error ({type(exc).__name__}): {exc}\n")
        return EXIT_API
    except Exception as exc:
        # Last-resort boundary of the CLI: report the type only, never the message.
        logger.error("check_unexpected_error", extra={"error_type": type(exc).__name__})
        out.write(f"\nUnexpected error ({type(exc).__name__}). See the log for details.\n")
        return EXIT_UNEXPECTED
    return EXIT_OK


def _run(
    max_clips: int | None,
    *,
    environ: Mapping[str, str] | None,
    out: TextIO,
    clock: Clock | None,
    transport: httpx.BaseTransport | None,
    ssm_fetch: SsmFetch,
    model_server: ModelServer | None,
    uploader: Uploader | None,
    should_stop: Callable[[], bool] | None,
) -> int:
    interrupt = threading.Event()
    clock = clock if clock is not None else InterruptibleClock(interrupt)
    started_at = clock.now()
    try:
        loaded = load_settings(environ=environ, ssm_fetch=ssm_fetch)
    except ConfigError as exc:
        out.write(f"{exc}\n")
        return EXIT_CONFIG

    def on_signal(signum: int, frame: FrameType | None) -> None:
        logger.info("signal_received", extra={"signal": signal.Signals(signum).name})
        interrupt.set()

    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        result = run_agent(
            loaded.settings,
            started_at=started_at,
            clock=clock,
            should_stop=should_stop or interrupt.is_set,
            max_clips=max_clips,
            job_api_transport=transport,
            model_server=model_server,
            uploader=uploader,
        )
    except ConfigError as exc:
        out.write(f"{exc}\n")
        return EXIT_CONFIG
    except Exception:
        # Last-resort boundary of the process: log the traceback (redacted) and exit 4.
        logger.exception("run_unexpected_error")
        return EXIT_UNEXPECTED
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return result.exit_code
