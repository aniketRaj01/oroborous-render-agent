from __future__ import annotations

from datetime import datetime, timedelta, timezone

START = datetime(2026, 10, 18, 6, 0, 0, tzinfo=timezone.utc)


class FakeClock:
    """Deterministic clock: sleeping advances time instantly and is recorded."""

    def __init__(self, start: datetime = START) -> None:
        self._now = start
        self._monotonic = 0.0
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._monotonic += seconds
