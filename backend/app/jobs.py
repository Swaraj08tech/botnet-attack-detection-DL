"""Background jobs with real stages: dataset analysis, PDF reports and per-flow incident explanations.

Each kind of job has one worker thread (the model, the PDF writer and the local LLM each do one thing at a time); further jobs wait
in a queue whose position is shown. The browser polls a job's stage. Jobs live in memory; their results are stored in the database.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

log = logging.getLogger('botnet-api.jobs')


@dataclass
class Job:
    id: str
    key: str                              # what the job is about, e.g. "analysis:<upload id>" or "report:12"
    stages: list[str]
    status: str = 'queued'                # queued / running / done / failed
    stage: str = 'queued'
    detail: str = ''                      # sub-step shown under the stage, e.g. "attack type 2 of 4: DoS"
    error: str = ''
    result: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    finished: float | None = None


class JobRunner:
    def __init__(self, name: str, stages: list[str], keep: int = 300):
        self.name, self.stages = name, stages
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self.jobs: dict[str, Job] = {}
        self.latest: dict[str, str] = {}          # key -> latest job id
        self.lock = threading.Lock()
        self.keep = keep

    def submit(self, key: str, work) -> Job:
        """Run `work(progress)` in the background unless a job with the same key is already queued or running.

        `work` may return a dict, kept as the job result (for example the id of the stored analysis)."""
        with self.lock:
            current = self.jobs.get(self.latest.get(key, ''))
            if current and current.status in ('queued', 'running'):
                return current
            job = Job(id=uuid.uuid4().hex, key=key, stages=self.stages)
            self.jobs[job.id], self.latest[key] = job, job.id
            finished = [j for j in self.jobs.values() if j.finished]
            for old in finished[: max(0, len(self.jobs) - self.keep)]:
                self.jobs.pop(old.id, None)
        self.executor.submit(self._run, job, work)
        return job

    def _run(self, job: Job, work):
        job.status, job.stage = 'running', self.stages[0]

        def progress(stage: str, detail: str = ''):
            job.stage, job.detail = stage, detail

        try:
            job.result = work(progress) or {}
            job.status = 'done'
        except UserFacingError as error:
            job.status, job.error = 'failed', str(error)
        except Exception:                                                     # noqa: BLE001  details in the server log only
            log.exception('%s job %s failed', self.name, job.key)
            job.status, job.error = 'failed', 'An unexpected error occurred on the server. The details were written to the server log.'
        job.finished = time.time()

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def for_key(self, key: str) -> Job | None:
        return self.jobs.get(self.latest.get(key, ''))

    def describe(self, job: Job) -> dict:
        with self.lock:
            ahead = 0
            if job.status == 'queued':
                ahead = sum(1 for j in self.jobs.values() if j.status == 'queued' and j.created < job.created)
                ahead += any(j.status == 'running' for j in self.jobs.values())
        return {'job_id': job.id, 'key': job.key, 'status': job.status, 'stage': job.stage, 'detail': job.detail, 'stages': job.stages,
                'queue_position': ahead, 'error': job.error, 'result': job.result,
                'elapsed': round((job.finished or time.time()) - job.created, 1)}

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


class UserFacingError(RuntimeError):
    """An error whose message is safe and useful to show to the user."""
