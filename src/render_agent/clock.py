from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Current time as a timezone-aware UTC datetime."""
        ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class InterruptibleClock(SystemClock):
    """Sleeps on an event, so a signal handler that sets the event ends the wait at once."""

    def __init__(self, interrupt: threading.Event) -> None:
        self._interrupt = interrupt

    def sleep(self, seconds: float) -> None:
        self._interrupt.wait(seconds)
