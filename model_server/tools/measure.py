#!/usr/bin/env python3
"""Measure the model server on the instance: how long each clip takes and how memory behaves.

Standard library only. Run it while the server is up (`python -m ltx_server`):

    python measure.py --prompts prompts.txt --count 2 --width 576 --height 1024 --num-frames 361

It sends the clips one after another to POST /generate, and every few seconds records the machine's
available RAM (/proc/meminfo), the server process's resident memory (/proc/<pid>/status) and the
GPU memory in use (nvidia-smi) into a CSV. At the end it prints one line per clip and a summary:
how clip 1 compares with clips 2 to N, the lowest available RAM, and whether it fell below a safety
floor. docs/model-server-runbook.md explains how to turn those numbers into a decision.

The prompts file holds one prompt per line (blank lines and lines starting with # are skipped).
Prompts are sent to the server but are never printed or written to the CSV.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlsplit

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_DIR = "/tmp/render-agent"
CSV_HEADER = ["timestamp", "elapsed_s", "phase", "mem_available_mb", "server_rss_mb", "gpu_used_mb"]
KB_PER_MB = 1024
COMMAND_TIMEOUT_SECONDS = 10.0

EXIT_OK = 0
EXIT_CLIP_FAILED = 1
EXIT_USAGE = 2


# --- reading the machine -------------------------------------------------------------------------


def parse_mem_available_mb(meminfo: str) -> float | None:
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
    return int(match[1]) / KB_PER_MB if match else None


def parse_rss_mb(status: str) -> float | None:
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.MULTILINE)
    return int(match[1]) / KB_PER_MB if match else None


def parse_gpu_used_mb(nvidia_smi_output: str) -> float | None:
    """Sum of `memory.used` (MiB) over the GPUs in `nvidia-smi --query-gpu=memory.used`."""
    values = [int(line) for line in nvidia_smi_output.split() if line.isdigit()]
    return float(sum(values)) if values else None


def read_mem_available_mb() -> float | None:
    try:
        return parse_mem_available_mb(Path("/proc/meminfo").read_text())
    except OSError:
        return None


def read_rss_mb(pid: int) -> float | None:
    try:
        return parse_rss_mb(Path(f"/proc/{pid}/status").read_text())
    except OSError:
        return None


def read_gpu_used_mb() -> float | None:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        completed = subprocess.run(
            [smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_gpu_used_mb(completed.stdout) if completed.returncode == 0 else None


def find_server_pid() -> int | None:
    """The pid of `python -m ltx_server`, found with pgrep; None if not found or ambiguous."""
    pgrep = shutil.which("pgrep")
    if pgrep is None:
        return None
    try:
        completed = subprocess.run(
            [pgrep, "-f", "--", "-m ltx_server( |$)"],
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    pids = [int(p) for p in completed.stdout.split() if p.isdigit() and int(p) != os.getpid()]
    return pids[0] if len(pids) == 1 else None


# --- sampling ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    elapsed_s: float
    phase: str
    mem_available_mb: float | None
    server_rss_mb: float | None
    gpu_used_mb: float | None


@dataclass
class Sources:
    mem_available_mb: Callable[[], float | None] = read_mem_available_mb
    server_rss_mb: Callable[[], float | None] = lambda: None
    gpu_used_mb: Callable[[], float | None] = read_gpu_used_mb


class Sampler:
    """Records a Sample every `interval` seconds, and on demand, into a CSV and into memory."""

    def __init__(self, sources: Sources, sink: IO[str], interval: float) -> None:
        self._sources = sources
        self._writer = csv.writer(sink)
        self._sink = sink
        self._interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = time.monotonic()
        self.phase = "start"
        self.samples: list[Sample] = []
        self._writer.writerow(CSV_HEADER)

    def start(self) -> None:
        self.sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=COMMAND_TIMEOUT_SECONDS + 5)
        self.sample()

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.sample()

    def sample(self) -> None:
        taken = Sample(
            elapsed_s=round(time.monotonic() - self._started, 2),
            phase=self.phase,
            mem_available_mb=self._sources.mem_available_mb(),
            server_rss_mb=self._sources.server_rss_mb(),
            gpu_used_mb=self._sources.gpu_used_mb(),
        )
        with self._lock:
            self.samples.append(taken)
            self._writer.writerow(
                [
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    taken.elapsed_s,
                    taken.phase,
                    *(
                        "" if v is None else round(v, 1)
                        for v in (taken.mem_available_mb, taken.server_rss_mb, taken.gpu_used_mb)
                    ),
                ]
            )
            self._sink.flush()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample()


# --- talking to the server -----------------------------------------------------------------------


@dataclass(frozen=True)
class ClipResult:
    index: int
    wall_seconds: float
    render_seconds: float | None
    error_code: str | None
    size_bytes: int | None = None

    @property
    def ok(self) -> bool:
        return self.error_code is None


def _connection(url: str, timeout: float) -> http.client.HTTPConnection:
    parts = urlsplit(url)
    return http.client.HTTPConnection(
        parts.hostname or "127.0.0.1", parts.port or 80, timeout=timeout
    )


def server_is_ready(url: str) -> bool:
    try:
        connection = _connection(url, 5)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            return response.status == 200 and bool(json.loads(response.read()).get("modelLoaded"))
        finally:
            connection.close()
    except (OSError, ValueError, http.client.HTTPException):
        return False


def wait_until_ready(
    url: str,
    timeout: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    poll: float = 5.0,
) -> bool:
    deadline = clock() + timeout
    while clock() < deadline:
        if server_is_ready(url):
            return True
        sleep(poll)
    return False


def generate(
    url: str, job_id: int, prompt: str, params: dict[str, Any], output_path: str, timeout: float
) -> tuple[int, dict[str, Any] | None]:
    body = json.dumps(
        {"jobId": job_id, "prompt": prompt, "params": params, "outputPath": output_path}
    ).encode()
    connection = _connection(url, timeout)
    try:
        connection.request(
            "POST", "/generate", body=body, headers={"content-type": "application/json"}
        )
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    return response.status, parsed if isinstance(parsed, dict) else None


# --- the summary ---------------------------------------------------------------------------------


@dataclass
class Summary:
    clips: list[ClipResult]
    samples: list[Sample]
    ram_floor_mb: float
    lines: list[str] = field(default_factory=list)

    @property
    def first_wall(self) -> float | None:
        return self.clips[0].wall_seconds if self.clips and self.clips[0].ok else None

    @property
    def later_wall_mean(self) -> float | None:
        later = [c.wall_seconds for c in self.clips[1:] if c.ok]
        return statistics.fmean(later) if later else None

    @property
    def min_available_mb(self) -> float | None:
        values = [s.mem_available_mb for s in self.samples if s.mem_available_mb is not None]
        return min(values) if values else None

    @property
    def peak_rss_mb(self) -> float | None:
        values = [s.server_rss_mb for s in self.samples if s.server_rss_mb is not None]
        return max(values) if values else None

    @property
    def peak_gpu_mb(self) -> float | None:
        values = [s.gpu_used_mb for s in self.samples if s.gpu_used_mb is not None]
        return max(values) if values else None

    @property
    def floor_breached(self) -> bool | None:
        lowest = self.min_available_mb
        return None if lowest is None else lowest < self.ram_floor_mb


def _gb(value_mb: float | None) -> str:
    return "n/a" if value_mb is None else f"{value_mb / 1024:.1f} GB"


def render_summary(summary: Summary) -> list[str]:
    lines = ["", "clip  wall_s  render_s  result"]
    for clip in summary.clips:
        render = "n/a" if clip.render_seconds is None else f"{clip.render_seconds:.1f}"
        result = "ok" if clip.ok else f"failed ({clip.error_code})"
        lines.append(f"{clip.index:>4}  {clip.wall_seconds:>6.1f}  {render:>8}  {result}")
    first, later = summary.first_wall, summary.later_wall_mean
    lines.append("")
    if first is not None and later is not None:
        change = (first - later) / first * 100 if first else 0.0
        lines.append(
            f"clip 1: {first:.1f} s    clips 2..{len(summary.clips)}: {later:.1f} s on average"
            f"    ({change:+.0f}% faster after the first)"
        )
    else:
        lines.append("clip timing comparison: needs at least two successful clips")
    lines.append(f"lowest MemAvailable: {_gb(summary.min_available_mb)}")
    lines.append(f"peak server RSS:     {_gb(summary.peak_rss_mb)}")
    lines.append(f"peak GPU memory:     {_gb(summary.peak_gpu_mb)}")
    breached = summary.floor_breached
    floor = f"{summary.ram_floor_mb / 1024:.1f} GB"
    if breached is None:
        lines.append(f"RAM floor ({floor}): not measured (no /proc/meminfo)")
    else:
        lines.append(f"RAM floor ({floor}): {'BREACHED' if breached else 'never breached'}")
    return lines


# --- command line --------------------------------------------------------------------------------


def load_prompts(path: Path) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise ValueError("the prompts file has no prompts")
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure clip times and memory on the model server."
    )
    parser.add_argument(
        "--prompts", type=Path, required=True, help="text file, one prompt per line"
    )
    parser.add_argument(
        "--count", type=int, help="number of clips (default: one per prompt; prompts cycle)"
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="model server URL (default %(default)s)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="a directory the server may write to (its ALLOWED_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("measure.csv"), help="where to write samples"
    )
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between samples")
    parser.add_argument(
        "--ram-floor-gb", type=float, default=4.0, help="safety floor for MemAvailable"
    )
    parser.add_argument("--pid", type=int, help="server process id (default: found with pgrep)")
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="seconds to wait for one clip"
    )
    parser.add_argument(
        "--ready-timeout", type=float, default=1800.0, help="seconds to wait for the model to load"
    )
    parser.add_argument(
        "--keep-files", action="store_true", help="do not delete the rendered clips"
    )
    for name in ("width", "height", "num-frames", "seed"):
        parser.add_argument(f"--{name}", type=int, help=f"{name} parameter (default: the server's)")
    parser.add_argument("--fps", type=float, help="fps parameter (default: the server's)")
    return parser


def params_from(args: argparse.Namespace) -> dict[str, Any]:
    names = {
        "width": "width",
        "height": "height",
        "numFrames": "num_frames",
        "fps": "fps",
        "seed": "seed",
    }
    return {
        key: getattr(args, attribute)
        for key, attribute in names.items()
        if getattr(args, attribute) is not None
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    sources: Sources | None = None,
    out: IO[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    stream = out if out is not None else sys.stdout
    try:
        prompts = load_prompts(args.prompts)
    except (OSError, ValueError) as exc:
        stream.write(f"cannot use the prompts file: {type(exc).__name__}\n")
        return EXIT_USAGE
    count = args.count if args.count is not None else len(prompts)
    if count < 1 or args.interval <= 0:
        stream.write("--count and --interval must be positive\n")
        return EXIT_USAGE

    if sources is None:
        pid = args.pid if args.pid is not None else find_server_pid()
        if pid is None:
            stream.write("note: server process not found; RSS will not be recorded (use --pid)\n")
        sources = Sources(
            server_rss_mb=(lambda: read_rss_mb(pid)) if pid is not None else (lambda: None)
        )

    stream.write("waiting for the model to load...\n")
    stream.flush()
    if not wait_until_ready(args.url, args.ready_timeout):
        stream.write("the server did not become ready in time\n")
        return EXIT_CLIP_FAILED

    params = params_from(args)
    results: list[ClipResult] = []
    with args.csv.open("w", newline="", encoding="utf-8") as sink:
        sampler = Sampler(sources, sink, args.interval)
        sampler.start()
        try:
            for index in range(1, count + 1):
                sampler.set_phase(f"clip {index} start")
                output = args.output_dir / f"measure-{os.getpid()}-{index}.mp4"
                started = time.monotonic()
                try:
                    status, body = generate(
                        args.url,
                        900000 + index,
                        prompts[(index - 1) % len(prompts)],
                        params,
                        str(output),
                        args.timeout,
                    )
                except (OSError, http.client.HTTPException) as exc:
                    status, body = 0, {"error": {"code": type(exc).__name__}}
                wall = time.monotonic() - started
                sampler.set_phase(f"clip {index} end")
                size = output.stat().st_size if output.is_file() else None
                if not args.keep_files:
                    output.unlink(missing_ok=True)
                if status == 200 and body is not None:
                    render = body.get("render", {}).get("renderSeconds")
                    result = ClipResult(
                        index, wall, float(render) if render is not None else None, None, size
                    )
                else:
                    code = ((body or {}).get("error") or {}).get("code", f"http {status}")
                    result = ClipResult(index, wall, None, str(code))
                results.append(result)
                stream.write(
                    f"clip {index}: {wall:.1f} s, "
                    + ("ok\n" if result.ok else f"failed ({result.error_code})\n")
                )
                stream.flush()
        finally:
            sampler.stop()
    summary = Summary(results, sampler.samples, args.ram_floor_gb * 1024)
    stream.write("\n".join(render_summary(summary)) + "\n")
    stream.write(f"\nsamples written to {args.csv}\n")
    return EXIT_OK if all(clip.ok for clip in results) else EXIT_CLIP_FAILED


if __name__ == "__main__":
    sys.exit(main())
