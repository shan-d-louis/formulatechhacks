"""Small async job queue for heavy SIDEWALL analytics work.

The public API talks to this abstraction instead of calling replay/model analytics
directly. That keeps the request path fast now, and leaves a clear replacement seam
for Redis/RQ, Celery, or Dramatiq later.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


JobCallable = Callable[["JobRecord"], Awaitable[Any]]


@dataclass
class JobRecord:
    """In-memory representation of one queued analytics job."""

    id: str
    kind: str
    status: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: float = 0.0
    error: str | None = None
    result: Any = None

    def public(self) -> dict[str, Any]:
        """Return the JSON-safe status fields exposed by the API."""
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "progress": self.progress,
            "error": self.error,
        }


class JobQueue:
    """Single-process queue for hackathon deployments and tests."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._work: asyncio.Queue[tuple[str, JobCallable]] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def enqueue(self, kind: str, fn: JobCallable) -> JobRecord:
        """Create a queued job and schedule it for background execution."""
        async with self._lock:
            self._ensure_worker()
            job = JobRecord(id=uuid4().hex, kind=kind)
            self._jobs[job.id] = job
            await self._work.put((job.id, fn))
            return job

    def get(self, job_id: str) -> JobRecord | None:
        """Look up a job by id."""
        return self._jobs.get(job_id)

    def reset_for_tests(self) -> None:
        """Clear state for tests without exposing mutation through the public API."""
        self._jobs.clear()
        self._work = asyncio.Queue()
        self._worker = None

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            job_id, fn = await self._work.get()
            job = self._jobs.get(job_id)
            if job is None:
                self._work.task_done()
                continue
            job.status = "running"
            job.started_at = datetime.now(UTC)
            job.progress = max(job.progress, 0.05)
            try:
                job.result = await fn(job)
                job.progress = 1.0
                job.status = "succeeded"
            except Exception as exc:  # noqa: BLE001 - worker must capture and report failures
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "failed"
            finally:
                job.finished_at = datetime.now(UTC)
                self._work.task_done()


JOBS = JobQueue()
