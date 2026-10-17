from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from typing import TextIO

import httpx

from render_agent.check import DEFAULT_LEASE_MINUTES, run_claim, write_summary
from render_agent.clock import Clock, SystemClock
from render_agent.config import ConfigError
from render_agent.config_sources import SsmFetch, fetch_ssm_parameters, load_settings
from render_agent.job_api.client import create_job_api_client
from render_agent.job_api.errors import JobApiError
from render_agent.logging_setup import configure_logging

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_API = 3
EXIT_UNEXPECTED = 4

logger = logging.getLogger("render_agent")


def _lease_minutes(raw: str) -> int:
    value = int(raw)
    if not 1 <= value <= 170:
        raise argparse.ArgumentTypeError("must be between 1 and 170")
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
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    out: TextIO | None = None,
    clock: Clock | None = None,
    transport: httpx.BaseTransport | None = None,
    ssm_fetch: SsmFetch = fetch_ssm_parameters,
) -> int:
    args = build_parser().parse_args(argv)
    out = out if out is not None else sys.stdout
    clock = clock if clock is not None else SystemClock()
    configure_logging()
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


if __name__ == "__main__":
    sys.exit(main())
