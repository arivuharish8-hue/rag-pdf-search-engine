"""
Upload entrypoint for the RabbitMQ + Celery PDF processing pipeline.

The Flask request thread does *only*:
  1. save the PDF locally,
  2. upload it to Supabase Storage,
  3. create the processing_jobs row,
  4. publish job_id to RabbitMQ.

All heavy work (download, extraction, chunking, embedding, FAISS indexing,
metadata + status updates) runs in Celery workers.  See ``tasks.py``.
"""

import logging
import os

from utils.database import (
    create_job,
    get_failed_jobs,
    get_pending_jobs,
    update_job_status,
)
from utils.supabase_storage import upload_file

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"


# ---------------------------------------------------------------------------
# Upload + job creation  (runs in the Flask request thread)
# ---------------------------------------------------------------------------


def start_processing(pdf_file, filename):
    """Save PDF locally, upload to Supabase Storage, create a DB job and
    enqueue it on RabbitMQ.

    Returns (object_name, job_id).  Returns immediately — the actual
    extraction / embedding / indexing happens in a Celery worker.
    """
    from uuid import uuid4
    from werkzeug.utils import secure_filename

    safe_name = secure_filename(filename)
    object_name = f"{uuid4().hex}_{safe_name}"

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    local_path = os.path.join(UPLOAD_FOLDER, object_name)

    pdf_file.save(local_path)
    upload_file(local_path, object_name)
    job_id = create_job(object_name, object_name)
    enqueue_job(job_id)

    logger.info("[Processing] Job %s created for %s", job_id, object_name)
    return object_name, job_id


# ---------------------------------------------------------------------------
# Queue helpers  (Flask side → RabbitMQ)
# ---------------------------------------------------------------------------


def enqueue_job(job_id):
    """Publish job_id to RabbitMQ for processing.

    Import of tasks is lazy to avoid a Celery/Flask import cycle.  If the
    broker is temporarily unreachable the upload still succeeds — the job is
    non-terminal in the DB, so ``resume_pending_jobs()`` re-enqueues it later.
    """
    from tasks import process_pdf_job

    try:
        process_pdf_job.delay(job_id)
        logger.info("[Queue] job %s received and sent to RabbitMQ", job_id)
    except Exception as exc:
        logger.error("[Queue] failed to enqueue job %s: %s", job_id, exc)


def resume_pending_jobs():
    """Re-enqueue every non-terminal job (crash / restart recovery).

    Replaces the old background-thread worker.  Idempotent: each job is only
    processed once per message thanks to the atomic claim in ``tasks.py``.
    Returns the number of jobs re-enqueued.
    """
    pending = get_pending_jobs()
    if not pending:
        logger.info("[Processing] No pending jobs to resume")
        return 0

    logger.info("[Processing] Resuming %d pending job(s) ...", len(pending))
    for job in pending:
        jid = job["job_id"]
        logger.info("[Processing]   %s: %s (stage=%s, checkpoint=%d)",
                    jid, job["pdf_name"], job["current_stage"],
                    job["last_processed_chunk"])
        enqueue_job(jid)
    return len(pending)


def retry_failed_jobs():
    """Manually re-queue FAILED jobs.

    Automatic retries are handled inside the Celery tasks via
    ``self.retry()``.  This helper is for operator use (e.g. after fixing the
    underlying cause) to push a FAILED job back into the queue.
    Returns the number of jobs re-enqueued.
    """
    failed = get_failed_jobs()
    if not failed:
        logger.info("[Processing] No failed jobs to retry")
        return 0

    logger.info("[Processing] Re-queueing %d failed job(s) ...", len(failed))
    for job in failed:
        jid = job["job_id"]
        update_job_status(jid, status="UPLOADED", current_stage="UPLOADED",
                          error_message=None)
        enqueue_job(jid)
        logger.info("[Processing]   %s: %s", jid, job["pdf_name"])
    return len(failed)
