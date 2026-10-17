"""Atomic output and confinement to ALLOWED_OUTPUT_DIR."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ltx_server import output
from ltx_server.renderer import FakeRenderer
from ms_helpers import generate_body, make_settings, request, running


def names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


# --- atomic output -------------------------------------------------------------------------------


def test_success_leaves_exactly_the_final_file_and_no_temp(tmp_path: Path) -> None:
    with running(make_settings(tmp_path), FakeRenderer()) as server:
        status, body, _ = request(server.port, "POST", "/generate", generate_body(server.path()))
    assert status == 200 and body["outputPath"] == server.path()
    assert names(server.out) == ["412-1.mp4"]
    assert Path(server.path()).read_bytes()[4:8] == b"ftyp"


def test_the_renderer_never_writes_to_the_final_name(tmp_path: Path) -> None:
    seen: list[Path] = []

    class Spy(FakeRenderer):
        def render(self, request, tmp_path):  # type: ignore[no-untyped-def]
            seen.append(tmp_path)
            assert not Path(server.path()).exists()
            return super().render(request, tmp_path)

    with running(make_settings(tmp_path), Spy()) as server:
        request(server.port, "POST", "/generate", generate_body(server.path()))
    assert len(seen) == 1
    assert seen[0].parent == server.out and seen[0] != Path(server.path())
    assert seen[0].name.startswith(".") and seen[0].name.endswith(".partial.mp4")
    assert output.TEMP_NAME.match(seen[0].name)


def test_a_crash_mid_write_leaves_no_file_under_the_final_name(tmp_path: Path) -> None:
    renderer = FakeRenderer(script={1: "mid_write"})
    with running(make_settings(tmp_path), renderer) as server:
        status, body, _ = request(server.port, "POST", "/generate", generate_body(server.path()))
    assert status == 500 and body["error"]["code"] == "render_error"
    assert names(server.out) == []  # neither the final file nor the half-written temp


def test_a_failure_keeps_an_existing_file_untouched(tmp_path: Path) -> None:
    renderer = FakeRenderer(script={1: "mid_write"})
    with running(make_settings(tmp_path), renderer) as server:
        Path(server.path()).write_bytes(b"previous content")
        request(server.port, "POST", "/generate", generate_body(server.path()))
        assert Path(server.path()).read_bytes() == b"previous content"
        status, _, _ = request(server.port, "POST", "/generate", generate_body(server.path()))
        assert status == 200 and Path(server.path()).read_bytes() != b"previous content"
    assert names(server.out) == ["412-1.mp4"]


def test_a_renderer_that_writes_nothing_is_a_render_error(tmp_path: Path) -> None:
    with running(make_settings(tmp_path), FakeRenderer(script={1: "no_output"})) as server:
        status, body, _ = request(server.port, "POST", "/generate", generate_body(server.path()))
    assert status == 500 and body["error"]["code"] == "render_error"
    assert names(server.out) == []


def test_commit_flushes_the_file_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "out.mp4"
    temp = output.temp_path_for(final)
    temp.write_bytes(b"data")
    order: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def recording_fsync(descriptor: int) -> None:
        order.append("fsync")
        real_fsync(descriptor)

    def recording_replace(source: object, destination: object) -> None:
        order.append("replace")
        real_replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    output.commit(temp, final)
    assert order[:2] == ["fsync", "replace"] and "fsync" in order[2:]  # then the directory
    assert final.read_bytes() == b"data" and not temp.exists()


def test_stale_partial_files_from_a_killed_run_are_removed_at_start(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    stale = settings.allowed_output_dir / ".412-1.0123456789ab.partial.mp4"
    stale.write_bytes(b"half")
    keep = [settings.allowed_output_dir / "412-1.mp4", settings.allowed_output_dir / ".notes.mp4"]
    for path in keep:
        path.write_bytes(b"x")
    with running(settings, FakeRenderer()):
        pass
    assert not stale.exists() and all(p.exists() for p in keep)


def test_temp_names_are_unique_and_end_in_mp4(tmp_path: Path) -> None:
    final = tmp_path / "412-1.mp4"
    generated = {output.temp_path_for(final) for _ in range(50)}
    assert len(generated) == 50 and all(
        p.suffix == ".mp4" and p.parent == tmp_path for p in generated
    )


# --- confinement ---------------------------------------------------------------------------------


def post(server_port: int, out: str) -> tuple[int, dict]:  # type: ignore[type-arg]
    status, body, _ = request(server_port, "POST", "/generate", generate_body(out))
    return status, body


def test_paths_outside_the_allowed_directory_are_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with running(make_settings(tmp_path), FakeRenderer()) as server:
        for target in (
            str(outside / "x.mp4"),
            str(server.out / ".." / "outside" / "x.mp4"),
            str(server.out / "sub" / ".." / ".." / "outside" / "x.mp4"),
            "/tmp/ltx-escape-probe.mp4",
        ):
            status, body = post(server.port, target)
            assert status == 400 and body["error"]["code"] == "invalid_request", target
    assert names(outside) == [] and not Path("/tmp/ltx-escape-probe.mp4").exists()


def test_a_symlinked_directory_that_leaves_the_allowed_area_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with running(make_settings(tmp_path), FakeRenderer()) as server:
        (server.out / "link").symlink_to(outside, target_is_directory=True)
        status, body = post(server.port, str(server.out / "link" / "x.mp4"))
    assert status == 400 and body["error"]["code"] == "invalid_request"
    assert names(outside) == []


def test_a_symlinked_file_that_leaves_the_allowed_area_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.mp4"
    victim.write_bytes(b"precious")
    with running(make_settings(tmp_path), FakeRenderer()) as server:
        (server.out / "412-1.mp4").symlink_to(victim)
        status, body = post(server.port, server.path())
    assert status == 400 and body["error"]["code"] == "invalid_request"
    assert victim.read_bytes() == b"precious"


def test_a_symlink_that_stays_inside_the_allowed_area_is_fine(tmp_path: Path) -> None:
    with running(make_settings(tmp_path), FakeRenderer()) as server:
        (server.out / "real").mkdir()
        (server.out / "alias").symlink_to(server.out / "real", target_is_directory=True)
        status, body = post(server.port, str(server.out / "alias" / "1-1.mp4"))
    assert status == 200 and body["outputPath"] == str(server.out / "alias" / "1-1.mp4")
    assert (server.out / "real" / "1-1.mp4").is_file()


def test_the_allowed_directory_itself_may_be_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "scratch-link"
    link.symlink_to(real, target_is_directory=True)
    with running(make_settings(tmp_path, allowed_output_dir=link), FakeRenderer()) as server:
        status, _ = post(server.port, str(link / "1-1.mp4"))
    assert status == 200 and (real / "1-1.mp4").is_file()


def test_a_directory_in_the_way_is_refused(tmp_path: Path) -> None:
    with running(make_settings(tmp_path), FakeRenderer()) as server:
        (server.out / "dir.mp4").mkdir()
        status, body = post(server.port, str(server.out / "dir.mp4"))
    assert status == 400 and body["error"]["code"] == "invalid_request"
