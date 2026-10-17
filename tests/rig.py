"""Wiring shared by the run loop tests: real client over the fake Job API, fake model server,
fake uploader, fake clock."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import httpx

from conftest import build_client, valid_env
from fakes.clock import FakeClock
from fakes.fake_job_api import FakeJobApi
from fakes.fake_model_server import FakeModelServer
from fakes.fake_uploader import FakeUploader
from render_agent.config import Settings, build_settings
from render_agent.runner import Runner, RunResult
from render_agent.scratch import prepare_scratch_dir


def envelope(status: int, code: str) -> httpx.Response:
    return httpx.Response(
        status, json={"ok": False, "error": {"code": code, "message": "scripted", "details": {}}}
    )


@dataclass
class Rig:
    clock: FakeClock
    fake_api: FakeJobApi
    model: FakeModelServer
    uploader: FakeUploader
    settings: Settings
    stop: bool = False
    stop_checks: list[int] = field(default_factory=list)

    @property
    def scratch(self) -> Path:
        return self.settings.scratch_dir

    def runner(self, max_clips: int | None = None, agent_id: str | None = None) -> Runner:
        return Runner(
            settings=self.settings,
            job_api=build_client(
                self.fake_api, self.clock, agent_id=agent_id or self.settings.agent_id
            ),
            model_server=self.model,
            uploader=self.uploader,
            clock=self.clock,
            should_stop=lambda: self.stop,
            max_clips=max_clips,
        )

    def run(self, max_clips: int | None = None) -> RunResult:
        return self.runner(max_clips).run(self.clock.now())

    def leftover_files(self) -> list[str]:
        return sorted(p.name for p in self.scratch.iterdir())


def make_rig(tmp_path: Path, **overrides: str) -> Rig:
    clock = FakeClock()
    fake_api = FakeJobApi(clock)
    settings = build_settings(
        {**valid_env(), "SCRATCH_DIR": str(tmp_path / "scratch"), **overrides}
    )
    prepare_scratch_dir(settings.scratch_dir)
    return Rig(
        clock=clock,
        fake_api=fake_api,
        model=FakeModelServer(clock),
        uploader=FakeUploader(fake_api, clock),
        settings=settings,
    )
