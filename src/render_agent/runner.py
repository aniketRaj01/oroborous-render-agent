from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from render_agent.clock import Clock
from render_agent.config import Settings
from render_agent.exit_codes import EXIT_AUTH, EXIT_OK, EXIT_SYSTEMIC
from render_agent.job_api.client import JobApi
from render_agent.job_api.errors import (
    AuthError,
    ContractError,
    JobApiError,
    KeyMismatchError,
    ObjectMissingError,
    SizeMismatchError,
    TransientError,
)
from render_agent.job_api.models import (
    Clip,
    CompleteUpload,
    FailErrorCode,
    Plan,
    RenderInfo,
)
from render_agent.model_server.client import (
    ModelServer,
    ModelServerError,
    ModelServerRejected,
    ModelServerTimeout,
)
from render_agent.model_server.models import (
    GenerateParams,
    GenerateRequest,
    ModelServerErrorCode,
    RenderReport,
)
from render_agent.redaction import safe_text
from render_agent.uploader import Uploader, UploadError

logger = logging.getLogger(__name__)

TIMING_WINDOW = 5
TIMING_SAFETY_FACTOR = 1.25
COMPLETE_BACKOFF_BASE_SECONDS = 5.0
COMPLETE_BACKOFF_MAX_SECONDS = 60.0
FAIL_MESSAGE_CHARS = 500
FTYP_OFFSET = 4

# Reasons a run ends. The exit code follows from the reason.
REASON_IDLE = "idle"
REASON_DEADLINE = "deadline"
REASON_SIGNAL = "signal"
REASON_MAX_CLIPS = "max_clips"
REASON_MODEL_SERVER_NOT_READY = "model_server_not_ready"
REASON_CONSECUTIVE_FAILURES = "consecutive_failures"
REASON_AUTH_ERROR = "auth_error"

_EXIT_CODES = {
    REASON_IDLE: EXIT_OK,
    REASON_DEADLINE: EXIT_OK,
    REASON_SIGNAL: EXIT_OK,
    REASON_MAX_CLIPS: EXIT_OK,
    REASON_MODEL_SERVER_NOT_READY: EXIT_SYSTEMIC,
    REASON_CONSECUTIVE_FAILURES: EXIT_SYSTEMIC,
    REASON_AUTH_ERROR: EXIT_AUTH,
}


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    reason: str
    done: int
    failed: int
    skipped: int
    dropped: int
    runtime_seconds: float


class _ClipFailure(Exception):
    """The clip failed in a way that is reported to the Job API with this code."""

    def __init__(self, code: FailErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _Dropped(Exception):
    """We do not (or may not) hold the job: nothing to report, move on."""


class _Interrupted(Exception):
    """A stop signal arrived. The job is left for the same agent id to resume."""


class _CompleteUnconfirmed(Exception):
    """Uploaded, but complete could not be confirmed within the retry budget."""


class Runner:
    def __init__(
        self,
        *,
        settings: Settings,
        job_api: JobApi,
        model_server: ModelServer,
        uploader: Uploader,
        clock: Clock,
        should_stop: Callable[[], bool] = lambda: False,
        max_clips: int | None = None,
    ) -> None:
        self._settings = settings
        self._job_api = job_api
        self._model_server = model_server
        self._uploader = uploader
        self._clock = clock
        self._should_stop = should_stop
        self._max_clips = max_clips

        self._work: deque[tuple[Plan, Clip]] = deque()
        # Jobs we gave up on this run. A later claim may return them again (we still hold the
        # lease); ignoring them keeps a persistently failing job from becoming a tight loop.
        self._given_up: set[int] = set()
        self._timings: deque[float] = deque(maxlen=TIMING_WINDOW)
        self._done = 0
        self._failed = 0
        self._dropped = 0
        self._skipped = 0
        self._consecutive_failures = 0
        self._started_at = clock.now()
        self._deadline = self._started_at

    # --- public entry point --------------------------------------------------------------------

    def run(self, started_at: datetime) -> RunResult:
        self._started_at = started_at
        self._deadline = started_at + timedelta(minutes=self._settings.run_deadline_minutes)
        logger.info(
            "run_start",
            extra={
                "agent_id": self._settings.agent_id,
                "agent_version": self._settings.agent_version,
                "deadline_at": self._deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "max_clips": self._max_clips if self._max_clips is not None else "none",
            },
        )
        try:
            reason = self._loop()
        except AuthError as exc:
            logger.error(
                "job_api_auth_error", extra={"route": exc.route, "status": exc.http_status}
            )
            reason = REASON_AUTH_ERROR
        except Exception:
            self._log_summary("unexpected_error")
            raise
        return self._finish(reason)

    # --- main loop -----------------------------------------------------------------------------

    def _loop(self) -> str:
        reason = self._wait_for_model_server()
        if reason is not None:
            return reason
        idle_since: datetime | None = None
        while True:
            if self._should_stop():
                return self._abandon_work(REASON_SIGNAL)
            if self._max_clips is not None and self._done + self._failed >= self._max_clips:
                return self._abandon_work(REASON_MAX_CLIPS)
            if not self._can_fit_a_clip():
                return self._abandon_work(REASON_DEADLINE)

            if self._work:
                plan, clip = self._work.popleft()
                stop_reason = self._process_clip(plan, clip)
                if stop_reason is not None:
                    return self._abandon_work(stop_reason)
                continue

            if self._claim():
                idle_since = None
                continue
            now = self._clock.now()
            idle_since = idle_since or now
            idle_seconds = (now - idle_since).total_seconds()
            if idle_seconds >= self._settings.idle_exit_minutes * 60:
                logger.info("idle_exit", extra={"idle_seconds": round(idle_seconds)})
                return REASON_IDLE
            self._clock.sleep(self._settings.empty_claim_wait_seconds)

    def _wait_for_model_server(self) -> str | None:
        give_up_at = self._clock.now() + timedelta(
            minutes=self._settings.model_server_ready_timeout_minutes
        )
        waited = False
        while True:
            if self._should_stop():
                return REASON_SIGNAL
            if self._model_server.is_ready():
                logger.info("model_server_ready", extra={"waited": str(waited).lower()})
                return None
            if self._clock.now() >= give_up_at:
                logger.error(
                    "model_server_not_ready",
                    extra={"timeout_minutes": self._settings.model_server_ready_timeout_minutes},
                )
                return REASON_MODEL_SERVER_NOT_READY
            waited = True
            self._clock.sleep(self._settings.model_server_health_poll_seconds)

    def _claim(self) -> bool:
        """Fill the work list from the Job API. True when it returned any clips."""
        try:
            response = self._job_api.claim(
                deadline_at=self._deadline, max_plans=self._settings.max_plans_per_claim
            )
        except TransientError as exc:
            logger.warning("claim_failed", extra={"error_type": type(exc).__name__})
            return False
        added = 0
        for plan in response.plans:
            for clip in sorted(plan.clips, key=lambda c: c.position):
                if clip.job_id not in self._given_up:
                    self._work.append((plan, clip))
                    added += 1
        logger.info(
            "claim",
            extra={
                "plans": response.totals.plans,
                "clips": response.totals.clips,
                "lease_expires_at": response.lease_expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
        return added > 0

    # --- deadline ------------------------------------------------------------------------------

    def _clip_estimate_seconds(self) -> float:
        if not self._timings:
            return self._settings.clip_time_estimate_seconds
        return sum(self._timings) / len(self._timings) * TIMING_SAFETY_FACTOR

    def _can_fit_a_clip(self) -> bool:
        needed = timedelta(seconds=self._clip_estimate_seconds())
        return self._clock.now() + needed <= self._deadline

    def _abandon_work(self, reason: str) -> str:
        """Clips claimed but not started are left to lease expiry; say so, one line each."""
        while self._work:
            plan, clip = self._work.popleft()
            self._skipped += 1
            logger.info(
                "clip_skipped",
                extra={
                    "job_id": clip.job_id,
                    "plan_id": plan.plan_id,
                    "position": clip.position,
                    "reason": reason,
                },
            )
        return reason

    # --- one clip ------------------------------------------------------------------------------

    def _process_clip(self, plan: Plan, clip: Clip) -> str | None:
        """Run one clip. Returns a reason when the whole run must stop, else None."""
        ids = {"job_id": clip.job_id, "plan_id": plan.plan_id, "position": clip.position}
        started = self._clock.monotonic()
        output: Path | None = None
        attempt = 0
        try:
            attempt = self._start(clip, ids)
            output = self._settings.scratch_dir / f"{clip.job_id}-{attempt}.mp4"
            self._check_stop()
            render = self._render(clip, output, ids, attempt)
            self._check_stop()
            self._verify(output)
            self._check_stop()
            upload_started = self._clock.monotonic()
            size = self._upload(clip, output)
            upload_seconds = self._clock.monotonic() - upload_started
            self._complete(clip, output, size, render, ids, attempt)
        except _Dropped:
            self._give_up(clip)
        except _Interrupted:
            logger.info("clip_interrupted", extra={**ids, "attempt": attempt})
            return REASON_SIGNAL
        except _CompleteUnconfirmed:
            self._give_up(clip)
        except _ClipFailure as failure:
            self._report_failure(clip, failure, ids, attempt)
            if self._consecutive_failures >= self._settings.max_consecutive_failures:
                logger.error(
                    "systemic_failure",
                    extra={
                        "consecutive_failures": self._consecutive_failures,
                        "limit": self._settings.max_consecutive_failures,
                    },
                )
                return REASON_CONSECUTIVE_FAILURES
        else:
            duration = self._clock.monotonic() - started
            self._timings.append(duration)
            self._done += 1
            self._consecutive_failures = 0
            logger.info(
                "clip_done",
                extra={
                    **ids,
                    "attempt": attempt,
                    "outcome": "done",
                    "render_s": round(render.render_seconds, 1),
                    "upload_s": round(upload_seconds, 1),
                    "size_bytes": size,
                    "duration_s": round(duration, 1),
                },
            )
        finally:
            if output is not None:
                output.unlink(missing_ok=True)
        return None

    def _give_up(self, clip: Clip) -> None:
        self._dropped += 1
        self._given_up.add(clip.job_id)

    def _check_stop(self) -> None:
        if self._should_stop():
            raise _Interrupted

    def _start(self, clip: Clip, ids: dict[str, int]) -> int:
        try:
            response = self._job_api.start(clip.job_id)
        except (ContractError, TransientError) as exc:
            logger.warning(
                "clip_dropped",
                extra={**ids, "step": "start", "error_code": exc.code or type(exc).__name__},
            )
            raise _Dropped from None
        logger.info(
            "clip_start",
            extra={**ids, "attempt": response.attempt, "resumed": str(clip.resumed).lower()},
        )
        return response.attempt

    def _render(self, clip: Clip, output: Path, ids: dict[str, int], attempt: int) -> RenderReport:
        request = GenerateRequest(
            job_id=clip.job_id,
            prompt=clip.prompt,
            params=GenerateParams.model_validate(
                clip.params.model_dump(exclude_none=True) if clip.params else {}
            ),
            output_path=str(output),
        )
        try:
            response = self._model_server.generate(request)
        except ModelServerRejected as exc:
            code = (
                FailErrorCode.OOM
                if exc.code == ModelServerErrorCode.OOM.value
                else FailErrorCode.RENDER_ERROR
            )
            raise _ClipFailure(code, self._failure_text(clip, str(exc))) from None
        except ModelServerTimeout:
            raise _ClipFailure(
                FailErrorCode.RENDER_ERROR,
                f"model server did not answer within "
                f"{self._settings.model_server_generate_timeout_seconds:g} s",
            ) from None
        except ModelServerError as exc:
            raise _ClipFailure(
                FailErrorCode.RENDER_ERROR, self._failure_text(clip, str(exc))
            ) from None
        if response.output_path != str(output):
            raise _ClipFailure(
                FailErrorCode.RENDER_ERROR, "model server wrote to a different path than asked"
            )
        logger.info(
            "clip_rendered",
            extra={**ids, "attempt": attempt, "render_s": round(response.render.render_seconds, 1)},
        )
        return response.render

    @staticmethod
    def _verify(output: Path) -> None:
        try:
            size = output.stat().st_size
            with output.open("rb") as handle:
                header = handle.read(FTYP_OFFSET + 4)
        except OSError:
            raise _ClipFailure(FailErrorCode.VERIFY_FAILED, "output file is missing") from None
        if size == 0:
            raise _ClipFailure(FailErrorCode.VERIFY_FAILED, "output file is empty")
        if header[FTYP_OFFSET:] != b"ftyp":
            raise _ClipFailure(
                FailErrorCode.VERIFY_FAILED, "output file is not an mp4 (no ftyp box)"
            )

    def _upload(self, clip: Clip, output: Path) -> int:
        try:
            return self._uploader.upload(
                output,
                bucket=clip.upload.bucket,
                region=clip.upload.region,
                key=clip.upload.key,
            )
        except UploadError as exc:
            raise _ClipFailure(FailErrorCode.UPLOAD_ERROR, str(exc)) from None

    # --- complete ------------------------------------------------------------------------------

    def _complete(
        self,
        clip: Clip,
        output: Path,
        size: int,
        render: RenderReport,
        ids: dict[str, int],
        attempt: int,
    ) -> None:
        info = RenderInfo(**render.model_dump())
        retry_until = min(
            self._deadline,
            self._clock.now() + timedelta(minutes=self._settings.complete_retry_budget_minutes),
        )
        reuploaded = False
        transient_failures = 0
        while True:
            try:
                self._job_api.complete(
                    clip.job_id,
                    upload=CompleteUpload(
                        bucket=clip.upload.bucket, key=clip.upload.key, size_bytes=size
                    ),
                    render=info,
                )
                return
            except (ObjectMissingError, SizeMismatchError) as exc:
                if reuploaded:
                    raise _ClipFailure(
                        FailErrorCode.UPLOAD_ERROR,
                        f"Job API could not verify the upload after a second try ({exc.code})",
                    ) from None
                reuploaded = True
                logger.warning("complete_reupload", extra={**ids, "error_code": exc.code})
                size = self._upload(clip, output)
            except KeyMismatchError:
                logger.error("complete_key_mismatch", extra={**ids, "attempt": attempt})
                raise _ClipFailure(
                    FailErrorCode.VERIFY_FAILED,
                    "Job API reported a bucket or key mismatch for this upload",
                ) from None
            except ContractError as exc:
                logger.warning(
                    "clip_dropped",
                    extra={**ids, "step": "complete", "error_code": exc.code or "unknown"},
                )
                raise _Dropped from None
            except TransientError as exc:
                transient_failures += 1
                if self._should_stop() or self._clock.now() >= retry_until:
                    logger.error(
                        "complete_unconfirmed",
                        extra={
                            **ids,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "tries": transient_failures,
                        },
                    )
                    raise _CompleteUnconfirmed from None
                delay = min(
                    COMPLETE_BACKOFF_MAX_SECONDS,
                    COMPLETE_BACKOFF_BASE_SECONDS * 2 ** (transient_failures - 1),
                )
                logger.warning(
                    "complete_retry",
                    extra={**ids, "error_type": type(exc).__name__, "wait_s": delay},
                )
                self._clock.sleep(delay)

    # --- failures ------------------------------------------------------------------------------

    def _failure_text(self, clip: Clip, text: str) -> str:
        if clip.prompt:
            text = text.replace(clip.prompt, "<prompt>")
        return safe_text(text, FAIL_MESSAGE_CHARS)

    def _report_failure(
        self, clip: Clip, failure: _ClipFailure, ids: dict[str, int], attempt: int
    ) -> None:
        self._failed += 1
        self._consecutive_failures += 1
        logger.warning(
            "clip_failed",
            extra={
                **ids,
                "attempt": attempt,
                "outcome": "failed",
                "error_code": failure.code.value,
                "detail": failure.message,
                "consecutive_failures": self._consecutive_failures,
            },
        )
        try:
            result = self._job_api.fail(clip.job_id, code=failure.code, message=failure.message)
        except AuthError:
            raise
        except JobApiError as exc:
            self._given_up.add(clip.job_id)
            logger.error(
                "fail_not_reported",
                extra={**ids, "error_code": exc.code or type(exc).__name__},
            )
            return
        logger.info(
            "clip_failure_reported",
            extra={
                **ids,
                "status": result.status.value,
                "attempts_left": result.attempts_left,
                "will_retry": str(result.will_retry).lower(),
            },
        )

    # --- finishing -----------------------------------------------------------------------------

    def _runtime_seconds(self) -> float:
        return (self._clock.now() - self._started_at).total_seconds()

    def _log_summary(self, reason: str) -> None:
        logger.info(
            "run_summary",
            extra={
                "done": self._done,
                "failed": self._failed,
                "skipped": self._skipped,
                "dropped": self._dropped,
                "runtime_s": round(self._runtime_seconds()),
                "reason": reason,
            },
        )

    def _finish(self, reason: str) -> RunResult:
        self._abandon_work(reason)
        self._log_summary(reason)
        return RunResult(
            exit_code=_EXIT_CODES[reason],
            reason=reason,
            done=self._done,
            failed=self._failed,
            skipped=self._skipped,
            dropped=self._dropped,
            runtime_seconds=self._runtime_seconds(),
        )
