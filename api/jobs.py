"""Background job registry."""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .schemas import Stage

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _JobLogHandler(logging.Handler):
    def __init__(self, sink: list[str]):
        super().__init__(level=logging.INFO)
        self.sink = sink
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink.append(self.format(record))
        except Exception:
            self.handleError(record)


@dataclass
class Job:
    id: str
    stage: Stage
    run_name: str
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    result: dict[str, Any] | None = None

    def view(self) -> dict:
        return {
            "id": self.id,
            "stage": self.stage,
            "status": self.status,
            "run_name": self.run_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "logs": self.logs,
            "error": self.error,
            "result": self.result,
        }


class JobRegistry:
    def __init__(self, max_workers: int = 2):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="polymap-job")

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(
                self._jobs.values(), key=lambda j: j.started_at or "", reverse=True
            )

    def submit(
        self,
        stage: Stage,
        run_name: str,
        fn: Callable[[], Any],
        summarize: Callable[[Any], dict] | None = None,
    ) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], stage=stage, run_name=run_name)
        with self._lock:
            self._jobs[job.id] = job

        def _run():
            handler = _JobLogHandler(job.logs)
            root = logging.getLogger("polymap")
            root.addHandler(handler)
            job.status = "running"
            job.started_at = _now()
            try:
                value = fn()
                job.result = summarize(value) if summarize else None
                job.status = "done"
            except Exception as e:
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.logs.append(traceback.format_exc())
                log.exception("job %s (%s) failed", job.id, stage)
            finally:
                job.finished_at = _now()
                root.removeHandler(handler)

        self._pool.submit(_run)
        return job

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


registry = JobRegistry()
