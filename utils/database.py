"""
Persistent job storage for the PDF processing pipeline.

Dual backend:
  - Supabase PostgreSQL (production) — requires migration.sql to be run first.
  - JSON file on disk (fallback)     — works out of the box, thread-safe,
                                       atomic writes via .tmp + os.replace.

Backend is auto-detected lazily on the first call so that dotenv is loaded
by the time we try to connect to Supabase.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend state (discovered lazily)
# ---------------------------------------------------------------------------
_BACKEND = None       # "supabase" | "json" | None
_SUPABASE_TABLE = None
_JSON_LOCK = threading.Lock()
JSON_PATH = os.path.join("database", "processing_jobs.json")

VALID_STAGES = [
    "UPLOADED",
    "EXTRACTING",
    "CHUNKING",
    "EMBEDDING",
    "INDEXING",
    "COMPLETED",
    "FAILED",
]

# ── Backend detection (lazy) ──────────────────────────────────────────


def _detect_backend():
    """Try Supabase first; fall back to JSON.  Idempotent."""
    global _BACKEND, _SUPABASE_TABLE
    if _BACKEND is not None:
        return _BACKEND == "supabase"

    try:
        from utils.supabase_storage import get_client

        client = get_client()
        client.table("processing_jobs").select("job_id").limit(1).execute()
        _BACKEND = "supabase"
        _SUPABASE_TABLE = client.table("processing_jobs")
        logger.info("[DB] Using Supabase PostgreSQL backend")
        return True
    except Exception as exc:
        _BACKEND = "json"
        os.makedirs("database", exist_ok=True)
        logger.info("[DB] Supabase table not available (%s). Using JSON file "
                    "backend at %s", exc, JSON_PATH)
        return False


# ── JSON helpers (thread-safe, crash-safe writes) ──────────────────────


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json():
    if not os.path.exists(JSON_PATH):
        return {}
    with open(JSON_PATH, "r") as f:
        return json.load(f)


def _write_json(data):
    tmp = JSON_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, JSON_PATH)


# ── Public API ─────────────────────────────────────────────────────────


def create_job(pdf_name, storage_path):
    """Insert a new job row and return its job_id."""
    job_id = str(uuid4())
    now = _now()
    row = {
        "job_id": job_id,
        "pdf_name": pdf_name,
        "storage_path": storage_path,
        "status": "UPLOADED",
        "current_stage": "UPLOADED",
        "total_chunks": 0,
        "last_processed_chunk": 0,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
    }

    if _detect_backend():
        _SUPABASE_TABLE.insert(row).execute()
    else:
        with _JSON_LOCK:
            data = _read_json()
            data[job_id] = row
            _write_json(data)

    logger.info("[DB] Job %s created for %s", job_id, pdf_name)
    return job_id


def update_job_status(job_id, **kwargs):
    """Update one or more fields on an existing job."""
    allowed = {"status", "current_stage", "total_chunks",
               "last_processed_chunk", "error_message"}
    payload = {k: v for k, v in kwargs.items() if k in allowed}

    if not payload:
        return

    payload["updated_at"] = _now()

    # Log what changed (without leaking error_message content in success logs)
    log_fields = {k: v for k, v in payload.items() if k != "updated_at"}
    logger.info("[DB] Job %s update: %s", job_id, log_fields)

    if _detect_backend():
        _SUPABASE_TABLE.update(payload).eq("job_id", job_id).execute()
    else:
        with _JSON_LOCK:
            data = _read_json()
            job = data.get(job_id)
            if job is None:
                logger.warning("[DB] Job %s not found for update", job_id)
                return
            job.update(payload)
            job["updated_at"] = payload["updated_at"]
            _write_json(data)


def save_checkpoint(job_id, last_processed_chunk, **kwargs):
    """Convenience: update last_processed_chunk + optional extras in one call."""
    kwargs["last_processed_chunk"] = last_processed_chunk
    update_job_status(job_id, **kwargs)


def get_pending_jobs():
    """Return all jobs where status is NOT COMPLETED nor FAILED, oldest first."""
    if _detect_backend():
        result = (
            _SUPABASE_TABLE.select("*")
            .not_.in_("status", ["COMPLETED", "FAILED"])
            .order("created_at")
            .execute()
        )
        jobs = result.data or []
    else:
        with _JSON_LOCK:
            data = _read_json()
        jobs = [j for j in data.values()
                if j["status"] not in ("COMPLETED", "FAILED")]
        jobs.sort(key=lambda j: j["created_at"])

    logger.debug("[DB] get_pending_jobs: %d job(s)", len(jobs))
    return jobs


def get_failed_jobs():
    """Return all jobs with status == FAILED, oldest first."""
    if _detect_backend():
        result = (
            _SUPABASE_TABLE.select("*")
            .eq("status", "FAILED")
            .order("created_at")
            .execute()
        )
        jobs = result.data or []
    else:
        with _JSON_LOCK:
            data = _read_json()
        jobs = [j for j in data.values() if j["status"] == "FAILED"]
        jobs.sort(key=lambda j: j["created_at"])

    logger.debug("[DB] get_failed_jobs: %d job(s)", len(jobs))
    return jobs


def get_job(job_id):
    """Return a single job dict, or None."""
    if _detect_backend():
        result = _SUPABASE_TABLE.select("*").eq("job_id", job_id).execute()
        job = result.data[0] if result.data else None
    else:
        with _JSON_LOCK:
            data = _read_json()
        job = data.get(job_id)

    if job is None:
        logger.debug("[DB] Job %s not found", job_id)
    return job


def get_all_jobs():
    """Return every job as a list of dicts."""
    if _detect_backend():
        result = _SUPABASE_TABLE.select("*").order("created_at").execute()
        jobs = result.data or []
    else:
        with _JSON_LOCK:
            data = _read_json()
        jobs = list(data.values())

    logger.debug("[DB] get_all_jobs: %d job(s)", len(jobs))
    return jobs


def delete_job(job_id):
    """Remove a job row entirely."""
    if _detect_backend():
        _SUPABASE_TABLE.delete().eq("job_id", job_id).execute()
    else:
        with _JSON_LOCK:
            data = _read_json()
            data.pop(job_id, None)
            _write_json(data)
    logger.info("[DB] Job %s deleted", job_id)
