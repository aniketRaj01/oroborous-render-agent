"""In-process fake of the Job API, served through httpx.MockTransport.

Implements the contract's core semantics over real JSON: plans grouped with one job per clip,
leases and deadline arithmetic, attempts counted at start, max attempts, same-agent resume,
lease-expiry reclaim, idempotent start/complete/fail, the error envelope and status codes.

Interpretation choices where the contract is silent (not exercised by the client):
- a plan is returned only if every one of its clips that still needs a render is claimable by the
  requesting agent; otherwise it is skipped whole, so a plan is never split.
- a plan counts as failed when any of its jobs is failed, and is then never returned.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import httpx

from fakes.clock import FakeClock

MAX_ATTEMPTS = 3
LEASE_MARGIN_SECONDS = 300
DEFAULT_LEASE_MINUTES = 90
MAX_LEASE_MINUTES = 180
MAX_PLANS_CEILING = 50
FAIL_CODES = {"render_error", "oom", "verify_failed", "upload_error", "unknown"}

FAKE_BUCKET = "fake-clip-bucket"
FAKE_REGION = "us-east-1"


@dataclass
class FakeJob:
    job_id: int
    plan_id: int
    position: int
    prompt: str
    params: dict[str, Any] | None
    status: str = "pending"
    attempts: int = 0
    claimed_by: str | None = None
    lease_expires_at: datetime | None = None
    agent_version: str | None = None
    storage_key: str | None = None
    last_error: str | None = None
    last_fail_code: str | None = None
    last_fail_message: str | None = None

    @property
    def key(self) -> str:
        return f"plans/{self.plan_id}/clip-{self.position}.mp4"


@dataclass
class FakePlan:
    plan_id: int
    account_id: int
    jobs: list[FakeJob] = field(default_factory=list)


@dataclass
class RecordedRequest:
    method: str
    path: str
    body: bytes
    headers: dict[str, str]


Fault = httpx.Response | Exception


class FakeJobApi:
    def __init__(self, clock: FakeClock, *, require_signature: bool = False) -> None:
        self.clock = clock
        self.require_signature = require_signature
        self.plans: list[FakePlan] = []
        self.s3_objects: dict[tuple[str, str], int] = {}
        self.requests: list[RecordedRequest] = []
        self._faults: dict[str, list[Fault]] = {}
        self._next_job_id = 400
        self.transport = httpx.MockTransport(self.handle)

    # --- test setup ----------------------------------------------------------------------------

    def add_plan(self, plan_id: int, *, clips: int = 2, account_id: int = 5) -> FakePlan:
        plan = FakePlan(plan_id=plan_id, account_id=account_id)
        for position in range(clips):
            self._next_job_id += 1
            plan.jobs.append(
                FakeJob(
                    job_id=self._next_job_id,
                    plan_id=plan_id,
                    position=position,
                    prompt=f"fake prompt {plan_id}/{position}",
                    params={"width": 576, "height": 1024, "numFrames": 361, "fps": 24, "seed": 42},
                )
            )
        self.plans.append(plan)
        return plan

    def job(self, job_id: int) -> FakeJob:
        for plan in self.plans:
            for job in plan.jobs:
                if job.job_id == job_id:
                    return job
        raise KeyError(job_id)

    def put_object(self, job: FakeJob, size: int) -> None:
        self.s3_objects[(FAKE_BUCKET, job.key)] = size

    def inject_fault(self, route: str, *faults: Fault) -> None:
        """Responses or exceptions returned for the next calls to `route`, before normal handling"""
        self._faults.setdefault(route, []).extend(faults)

    def calls(self, route: str) -> list[RecordedRequest]:
        return [r for r in self.requests if r.path == route]

    # --- transport -----------------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = request.content
        self.requests.append(
            RecordedRequest(request.method, request.url.path, body, dict(request.headers))
        )
        queued = self._faults.get(request.url.path)
        if queued:
            fault = queued.pop(0)
            if isinstance(fault, Exception):
                raise fault
            return fault
        if self.require_signature and not self._signature_ok(request):
            return httpx.Response(403, json={"Message": "The security token is invalid"})
        if request.method != "POST":
            return self._error(405, "method_not_allowed", "Only POST is supported")
        routes: dict[str, Callable[[dict[str, Any]], httpx.Response]] = {
            "/v1/claim": self._claim,
            "/v1/start": self._start,
            "/v1/complete": self._complete,
            "/v1/fail": self._fail,
        }
        handler = routes.get(request.url.path)
        if handler is None:
            return self._error(404, "not_found", "Unknown route")
        try:
            payload = json.loads(body)
        except ValueError:
            return self._error(400, "invalid_request", "Body is not valid JSON")
        if not isinstance(payload, dict):
            return self._error(400, "invalid_request", "Body must be a JSON object")
        return handler(payload)

    @staticmethod
    def _signature_ok(request: httpx.Request) -> bool:
        authorization = request.headers.get("authorization", "")
        return (
            authorization.startswith("AWS4-HMAC-SHA256 ")
            and request.headers.get("x-amz-content-sha256")
            == hashlib.sha256(request.content).hexdigest()
            and "x-amz-content-sha256" in authorization
        )

    # --- helpers -------------------------------------------------------------------------------

    @staticmethod
    def _error(status: int, code: str, message: str, details: Any = None) -> httpx.Response:
        return httpx.Response(
            status,
            json={
                "ok": False,
                "error": {"code": code, "message": message, "details": details or {}},
            },
        )

    def _format(self, value: datetime) -> str:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _validate(
        self, payload: dict[str, Any], rules: dict[str, type | tuple[type, ...]]
    ) -> httpx.Response | None:
        problems = {
            name: "required" if name not in payload else "wrong type"
            for name, kind in rules.items()
            if name not in payload
            or isinstance(payload[name], bool)
            or not isinstance(payload[name], kind)
        }
        agent_id = payload.get("agentId")
        if isinstance(agent_id, str) and not 1 <= len(agent_id) <= 64:
            problems["agentId"] = "length"
        if problems:
            return self._error(400, "invalid_request", "Validation failed", problems)
        return None

    def _lease_live(self, job: FakeJob) -> bool:
        return job.lease_expires_at is not None and job.lease_expires_at > self.clock.now()

    def _claimable(self, job: FakeJob, agent_id: str) -> bool:
        if job.status in ("done", "failed"):
            return False
        return job.claimed_by == agent_id or not self._lease_live(job)

    # --- routes --------------------------------------------------------------------------------

    def _claim(self, payload: dict[str, Any]) -> httpx.Response:
        rules: dict[str, type | tuple[type, ...]] = {"agentId": str, "agentVersion": str}
        if (invalid := self._validate(payload, rules)) is not None:
            return invalid
        max_plans = payload.get("maxPlans", MAX_PLANS_CEILING)
        if (
            isinstance(max_plans, bool)
            or not isinstance(max_plans, int)
            or not 1 <= max_plans <= 50
        ):
            return self._error(400, "invalid_request", "Validation failed", {"maxPlans": "range"})
        if len(payload["agentVersion"]) > 64:
            return self._error(
                400, "invalid_request", "Validation failed", {"agentVersion": "length"}
            )

        now = self.clock.now()
        deadline = self._parse_deadline(payload.get("deadlineAt"))
        if (
            deadline is None
            or deadline <= now
            or deadline > now + timedelta(minutes=MAX_LEASE_MINUTES)
        ):
            deadline = now + timedelta(minutes=DEFAULT_LEASE_MINUTES)
        lease_expires_at = deadline + timedelta(seconds=LEASE_MARGIN_SECONDS)
        agent_id = payload["agentId"]

        plans_out: list[dict[str, Any]] = []
        for plan in self.plans:
            if len(plans_out) >= max_plans:
                break
            if any(job.status == "failed" for job in plan.jobs):
                continue
            remaining = [job for job in plan.jobs if job.status != "done"]
            if not remaining or not all(self._claimable(job, agent_id) for job in remaining):
                continue
            clips = []
            for job in sorted(remaining, key=lambda j: j.position):
                resumed = job.claimed_by == agent_id
                if job.status == "generating":
                    job.status = "pending"
                job.claimed_by = agent_id
                job.lease_expires_at = lease_expires_at
                job.agent_version = payload["agentVersion"]
                clips.append(
                    {
                        "jobId": job.job_id,
                        "position": job.position,
                        "prompt": job.prompt,
                        "params": job.params,
                        "upload": {"bucket": FAKE_BUCKET, "region": FAKE_REGION, "key": job.key},
                        "attempts": job.attempts,
                        "resumed": resumed,
                    }
                )
            plans_out.append(
                {
                    "planId": plan.plan_id,
                    "accountId": plan.account_id,
                    "clipsTotal": len(plan.jobs),
                    "clipsAlreadyDone": len(plan.jobs) - len(remaining),
                    "clips": clips,
                }
            )
        return httpx.Response(
            200,
            json={
                "ok": True,
                "serverTime": self._format(now),
                "leaseExpiresAt": self._format(lease_expires_at),
                "totals": {
                    "plans": len(plans_out),
                    "clips": sum(len(p["clips"]) for p in plans_out),
                },
                "plans": plans_out,
            },
        )

    @staticmethod
    def _parse_deadline(raw: Any) -> datetime | None:
        if not isinstance(raw, str) or not raw.endswith("Z"):
            return None
        try:
            return datetime.fromisoformat(raw[:-1] + "+00:00")
        except ValueError:
            return None

    def _find(self, payload: dict[str, Any]) -> FakeJob | httpx.Response:
        try:
            return self.job(payload["jobId"])
        except KeyError:
            return self._error(404, "job_not_found", f"Job {payload['jobId']} not found")

    def _start(self, payload: dict[str, Any]) -> httpx.Response:
        if (invalid := self._validate(payload, {"agentId": str, "jobId": int})) is not None:
            return invalid
        job = self._find(payload)
        if isinstance(job, httpx.Response):
            return job
        agent_id = payload["agentId"]
        if job.status in ("done", "failed"):
            return self._error(409, "wrong_state", f"Job {job.job_id} is {job.status}")
        if job.claimed_by != agent_id:
            return self._error(
                409, "not_your_lease", f"Job {job.job_id} is leased to another agent"
            )
        if job.status == "generating":
            return self._start_body(job, already_started=True)
        if not self._lease_live(job):
            return self._error(409, "lease_expired", f"Lease for job {job.job_id} has ended")
        if job.attempts >= MAX_ATTEMPTS:
            job.status = "failed"
            return self._error(409, "max_attempts_reached", f"Job {job.job_id} used all attempts")
        job.status = "generating"
        job.attempts += 1
        return self._start_body(job, already_started=False)

    @staticmethod
    def _start_body(job: FakeJob, *, already_started: bool) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "ok": True,
                "jobId": job.job_id,
                "attempt": job.attempts,
                "maxAttempts": MAX_ATTEMPTS,
                "alreadyStarted": already_started,
            },
        )

    def _plan_progress(self, job: FakeJob) -> dict[str, Any]:
        plan = next(p for p in self.plans if p.plan_id == job.plan_id)
        done = sum(1 for j in plan.jobs if j.status == "done")
        return {
            "id": plan.plan_id,
            "clipsDone": done,
            "clipsTotal": len(plan.jobs),
            "ready": done == len(plan.jobs),
        }

    def _complete_body(self, job: FakeJob) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "ok": True,
                "jobId": job.job_id,
                "status": "done",
                "storageKey": job.storage_key,
                "plan": self._plan_progress(job),
            },
        )

    def _complete(self, payload: dict[str, Any]) -> httpx.Response:
        rules: dict[str, type | tuple[type, ...]] = {"agentId": str, "jobId": int, "upload": dict}
        if (invalid := self._validate(payload, rules)) is not None:
            return invalid
        upload = payload["upload"]
        size = upload.get("sizeBytes")
        if (
            not isinstance(upload.get("bucket"), str)
            or not isinstance(upload.get("key"), str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            return self._error(400, "invalid_request", "Validation failed", {"upload": "invalid"})
        job = self._find(payload)
        if isinstance(job, httpx.Response):
            return job
        if upload["bucket"] != FAKE_BUCKET or upload["key"] != job.key:
            return self._error(
                409, "key_mismatch", f"Unexpected bucket or key for job {job.job_id}"
            )
        if job.status == "done":
            return self._complete_body(job)
        if job.status != "generating":
            return self._error(409, "wrong_state", f"Job {job.job_id} is {job.status}")
        if job.claimed_by != payload["agentId"]:
            return self._error(
                409, "not_your_lease", f"Job {job.job_id} is leased to another agent"
            )
        actual = self.s3_objects.get((upload["bucket"], upload["key"]), 0)
        if actual <= 0:
            return self._error(409, "object_missing", "Object is missing or empty")
        if actual != size:
            return self._error(409, "size_mismatch", "Object size differs from sizeBytes")
        job.status = "done"
        job.storage_key = job.key
        job.claimed_by = None
        job.lease_expires_at = None
        return self._complete_body(job)

    def _fail_body(self, job: FakeJob) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "ok": True,
                "jobId": job.job_id,
                "status": job.status,
                "attemptsLeft": max(MAX_ATTEMPTS - job.attempts, 0),
                "willRetry": job.status == "pending",
            },
        )

    def _fail(self, payload: dict[str, Any]) -> httpx.Response:
        rules: dict[str, type | tuple[type, ...]] = {"agentId": str, "jobId": int, "error": dict}
        if (invalid := self._validate(payload, rules)) is not None:
            return invalid
        error = payload["error"]
        if error.get("code") not in FAIL_CODES or not isinstance(error.get("message"), str):
            return self._error(400, "invalid_request", "Validation failed", {"error": "invalid"})
        job = self._find(payload)
        if isinstance(job, httpx.Response):
            return job
        message = error["message"][:2000]
        if job.claimed_by != payload["agentId"]:
            return self._error(
                409, "not_your_lease", f"Job {job.job_id} is leased to another agent"
            )
        if (
            job.status in ("pending", "failed")
            and job.last_fail_code == error["code"]
            and job.last_fail_message == message
        ):
            return self._fail_body(job)
        if job.status != "generating":
            return self._error(409, "wrong_state", f"Job {job.job_id} is {job.status}")
        job.last_error = f"{error['code']}: {message}"
        job.last_fail_code = error["code"]
        job.last_fail_message = message
        job.status = "failed" if job.attempts >= MAX_ATTEMPTS else "pending"
        return self._fail_body(job)
