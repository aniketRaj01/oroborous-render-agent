from __future__ import annotations

import csv
import io
from pathlib import Path

import measure
import pytest
from measure import ClipResult, Sample, Sources, Summary, render_summary

from ltx_server.renderer import FakeRenderer
from ms_helpers import make_settings, running

MEMINFO = "MemTotal:       61000000 kB\nMemFree:         1000000 kB\nMemAvailable:   40960000 kB\n"
STATUS = "Name:\tpython\nVmPeak:\t 70000000 kB\nVmRSS:\t 36864000 kB\nThreads:\t12\n"


def test_parsers() -> None:
    assert measure.parse_mem_available_mb(MEMINFO) == 40960000 / 1024
    assert measure.parse_mem_available_mb("nothing here") is None
    assert measure.parse_rss_mb(STATUS) == 36864000 / 1024
    assert measure.parse_rss_mb("") is None
    assert measure.parse_gpu_used_mb("39600\n") == 39600
    assert measure.parse_gpu_used_mb("1000\n2000\n") == 3000
    assert measure.parse_gpu_used_mb("") is None and measure.parse_gpu_used_mb("N/A") is None


def test_reading_a_missing_process_or_file_is_none_not_an_error() -> None:
    assert measure.read_rss_mb(2**22 + 12345) is None


def samples(*available: float | None) -> list[Sample]:
    return [Sample(float(i), "p", a, 30000.0 + i, 40000.0) for i, a in enumerate(available)]


def test_the_summary_compares_clip_one_with_the_rest_and_checks_the_floor() -> None:
    clips = [
        ClipResult(1, 400.0, 380.0, None),
        ClipResult(2, 160.0, 150.0, None),
        ClipResult(3, 140.0, 130.0, None),
    ]
    summary = Summary(clips, samples(20000, 3000, 9000), ram_floor_mb=4096)
    assert summary.first_wall == 400 and summary.later_wall_mean == 150
    assert summary.min_available_mb == 3000 and summary.floor_breached is True
    text = "\n".join(render_summary(summary))
    assert "clip 1: 400.0 s" in text and "150.0 s on average" in text and "+62% faster" in text
    assert "BREACHED" in text and "lowest MemAvailable: 2.9 GB" in text


def test_the_floor_is_not_breached_when_ram_stays_high() -> None:
    summary = Summary([ClipResult(1, 10, 9, None)], samples(20000, 18000), ram_floor_mb=4096)
    assert summary.floor_breached is False
    assert "never breached" in "\n".join(render_summary(summary))


def test_failed_clips_are_shown_and_excluded_from_the_comparison() -> None:
    clips = [
        ClipResult(1, 100, 90, None),
        ClipResult(2, 5, None, "oom"),
        ClipResult(3, 50, 45, None),
    ]
    text = "\n".join(render_summary(Summary(clips, samples(9000), 4096)))
    assert "failed (oom)" in text and "clip 1: 100.0 s    clips 2..3: 50.0 s on average" in text


def test_a_single_clip_has_nothing_to_compare() -> None:
    text = "\n".join(render_summary(Summary([ClipResult(1, 10, 9, None)], samples(9000), 4096)))
    assert "needs at least two successful clips" in text


def test_without_meminfo_the_floor_is_reported_as_not_measured() -> None:
    summary = Summary([ClipResult(1, 10, 9, None)], samples(None), 4096)
    assert summary.floor_breached is None and "not measured" in "\n".join(render_summary(summary))


def test_prompts_file_skips_blank_lines_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "p.txt"
    path.write_text("# comment\n\nfirst prompt\n  second prompt  \n")
    assert measure.load_prompts(path) == ["first prompt", "second prompt"]
    path.write_text("# only a comment\n")
    with pytest.raises(ValueError, match="no prompts"):
        measure.load_prompts(path)


def fake_sources(available: list[float]) -> Sources:
    iterator = iter(available + [available[-1]] * 1000)
    return Sources(
        mem_available_mb=lambda: next(iterator),
        server_rss_mb=lambda: 30000.0,
        gpu_used_mb=lambda: 39600.0,
    )


def run_tool(
    tmp_path: Path, port: int, out: Path, sources: Sources, *extra: str
) -> tuple[int, str, Path]:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("SENTINEL-PROMPT first\nSENTINEL-PROMPT second\n")
    csv_path = tmp_path / "measure.csv"
    stream = io.StringIO()
    code = measure.main(
        [
            "--prompts",
            str(prompts),
            "--url",
            f"http://127.0.0.1:{port}",
            "--output-dir",
            str(out),
            "--csv",
            str(csv_path),
            "--interval",
            "0.02",
            "--ready-timeout",
            "10",
            "--timeout",
            "30",
            "--width",
            "256",
            "--height",
            "256",
            "--num-frames",
            "9",
            *extra,
        ],
        sources=sources,
        out=stream,
    )
    return code, stream.getvalue(), csv_path


def test_two_clips_against_the_real_server(tmp_path: Path) -> None:
    renderer = FakeRenderer(render_seconds=0.15)
    with running(make_settings(tmp_path), renderer) as server:
        code, text, csv_path = run_tool(
            tmp_path, server.port, server.out, fake_sources([30000, 9000, 8000])
        )
        leftovers = list(server.out.iterdir())
    assert code == 0 and leftovers == []  # clips are deleted after each measurement
    assert (
        renderer.renders == 2
        and renderer.requests[0].width == 256
        and renderer.requests[0].num_frames == 9
    )
    assert "clip 1:" in text and "clip 2:" in text and "lowest MemAvailable" in text
    assert "never breached" in text and "peak GPU memory:     38.7 GB" in text
    assert "SENTINEL" not in text
    rows = list(csv.reader(csv_path.open()))
    assert (
        rows[0] == measure.CSV_HEADER and len(rows) >= 6
    )  # start, 2x(start,end), end, plus timed ones
    assert "SENTINEL" not in csv_path.read_text()
    assert {row[2] for row in rows[1:]} >= {
        "clip 1 start",
        "clip 1 end",
        "clip 2 start",
        "clip 2 end",
    }
    assert all(row[4] == "30000.0" and row[5] == "39600.0" for row in rows[1:])


def test_a_ram_floor_breach_is_reported(tmp_path: Path) -> None:
    with running(make_settings(tmp_path), FakeRenderer()) as server:
        _, text, _ = run_tool(tmp_path, server.port, server.out, fake_sources([30000, 2000, 9000]))
    assert "BREACHED" in text


def test_a_failing_clip_makes_the_exit_status_non_zero(tmp_path: Path) -> None:
    with running(make_settings(tmp_path), FakeRenderer(script={2: "render_error"})) as server:
        code, text, _ = run_tool(tmp_path, server.port, server.out, fake_sources([30000]))
    assert code == 1 and "failed (render_error)" in text


def test_more_clips_than_prompts_cycle_through_them(tmp_path: Path) -> None:
    renderer = FakeRenderer()
    with running(make_settings(tmp_path), renderer) as server:
        code, _, _ = run_tool(
            tmp_path, server.port, server.out, fake_sources([30000]), "--count", "3"
        )
    assert code == 0 and [r.prompt for r in renderer.requests] == [
        "SENTINEL-PROMPT first",
        "SENTINEL-PROMPT second",
        "SENTINEL-PROMPT first",
    ]


def test_usage_errors(tmp_path: Path) -> None:
    out = io.StringIO()
    assert measure.main(["--prompts", str(tmp_path / "missing.txt")], out=out) == 2
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert measure.main(["--prompts", str(empty)], out=out) == 2
    assert "cannot use the prompts file" in out.getvalue()


def test_an_unreachable_server_is_reported_not_hung(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("one\n")
    out = io.StringIO()
    code = measure.main(
        [
            "--prompts",
            str(prompts),
            "--url",
            "http://127.0.0.1:1",
            "--ready-timeout",
            "0.1",
            "--csv",
            str(tmp_path / "m.csv"),
        ],
        sources=fake_sources([1]),
        out=out,
    )
    assert code == 1 and "did not become ready" in out.getvalue()
