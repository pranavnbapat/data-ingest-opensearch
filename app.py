# app.py

import logging
import os
import traceback
from collections import deque
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Optional, Dict, List, Any, cast
from uuid import uuid4

from fastapi import FastAPI, Body, BackgroundTasks, HTTPException
from pydantic import BaseModel, field_validator

from create_neural_search_index import run_index, MODEL_CONFIG

app = FastAPI(title="Neural Search Indexer")

OPENAPI_EXAMPLES = {
    "dev": {"summary": "Run in DEV", "value": {"env_mode": "DEV"}},
    "prd": {"summary": "Run in PRD", "value": {"env_mode": "PRD"}},
}

class EnvMode(str, Enum):
    DEV = "DEV"
    PRD = "PRD"

# -------- Job tracking --------
class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    success = "success"
    error = "error"

class Job(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    models: List[str] = []
    summary: Optional[dict] = None
    error: Optional[str] = None

JOBS: Dict[str, Job] = {}
JOB_LOGS: Dict[str, deque[str]] = {}
JOB_LOCK = Lock()

class PerJobLogHandler(logging.Handler):
    """Push all log lines into an in-memory ring buffer and (optionally) a file."""
    def __init__(self, job_id: str, max_lines: int = 4000, persist_dir: Optional[str] = "output/job-logs"):
        super().__init__()
        self.job_id = job_id
        self.buf = JOB_LOGS.setdefault(job_id, deque(maxlen=max_lines))
        self.persist_fp = None
        self.log_path = None
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self.log_path = os.path.join(persist_dir, f"{job_id}.log")
            try:
                self.persist_fp = open(self.log_path, "a", encoding="utf-8")
            except Exception:
                self.persist_fp = None

    def emit(self, record: logging.LogRecord):
        line = f"{datetime.utcnow().isoformat()}Z {record.levelname} {self.format(record)}"
        self.buf.append(line)
        if self.persist_fp:
            try:
                self.persist_fp.write(line + "\n")
                self.persist_fp.flush()
            except Exception:
                pass

    def close(self):
        try:
            if self.persist_fp:
                self.persist_fp.close()
        finally:
            super().close()


# -------- Request schema --------
class RunParams(BaseModel):
    env_mode: Optional[EnvMode] = None
    model_config = {"extra": "ignore"}


# -------- Routes --------
@app.get("/healthz")
def healthz():
    return {"ok": True}

def _run_job(job_id: str, env_mode_val: str):
    y = datetime.now().strftime("%Y")
    m = datetime.now().strftime("%m")
    log_dir = os.path.join("job-logs", env_mode_val, y, m)

    handler = PerJobLogHandler(job_id, persist_dir=log_dir)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        with JOB_LOCK:
            job = JOBS[job_id]
            job.status = JobStatus.running
            job.started_at = datetime.utcnow()

        logging.info("Starting index job (env=%s, models=%s)", env_mode_val, "ALL")
        summary = run_index(models=None, env_mode=env_mode_val)

        with JOB_LOCK:
            job.summary = summary
            job.status = JobStatus.success
            job.finished_at = datetime.utcnow()
        logging.info("Index job finished.")
    except Exception as e:
        err = f"{e.__class__.__name__}: {e}"
        tb = traceback.format_exc()
        with JOB_LOCK:
            job = JOBS[job_id]
            job.status = JobStatus.error
            job.error = err + "\n" + tb
            job.finished_at = datetime.utcnow()
        logging.exception("Index job failed: %s", err)
    finally:
        root_logger.removeHandler(handler)
        handler.close()

@app.post("/build_index/run")
def run_index_endpoint(
    params: Optional[RunParams] = Body(None, openapi_examples=cast(Dict[str, Any], OPENAPI_EXAMPLES)),
    bg: BackgroundTasks = None,
):
    if params is None:
        params = RunParams()

    # prevent concurrent runs
    with JOB_LOCK:
        if any(j.status == JobStatus.running for j in JOBS.values()):
            raise HTTPException(status_code=409, detail="Another index job is already running")

    env_mode_val = (params.env_mode.value if params.env_mode else (os.getenv("ENV_MODE") or "DEV")).upper()
    if env_mode_val not in {"DEV", "PRD"}:
        raise HTTPException(status_code=400, detail=f"Invalid ENV_MODE {env_mode_val!r}. Use DEV or PRD")

    job_id = uuid4().hex
    job = Job(
        id=job_id,
        status=JobStatus.queued,
        created_at=datetime.utcnow(),
        models=[],
    )
    with JOB_LOCK:
        JOBS[job_id] = job

    bg.add_task(_run_job, job_id, env_mode_val)
    return {"status": "scheduled", "job_id": job_id}

@app.get("/jobs")
def list_jobs():
    with JOB_LOCK:
        return [j.model_dump() for j in JOBS.values()]

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job.model_dump()

@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, tail: int = 200):
    buf = JOB_LOGS.get(job_id)
    if buf is None:
        raise HTTPException(status_code=404, detail="job not found")
    tail = max(1, min(tail, len(buf)))
    return {"job_id": job_id, "lines": list(buf)[-tail:]}

# Optional: basic logging setup for local runs
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
