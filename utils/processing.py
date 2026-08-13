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
import time

from utils.database import (
    claim_upload,
    create_job,
    get_failed_jobs,
    get_pending_jobs,
    get_stuck_uploads,
    release_upload,
    update_job_status,
)
from utils.supabase_storage import upload_file

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"

ENQUEUE_RETRIES = int(os.getenv("ENQUEUE_RETRIES", "3"))
ENQUEUE_RETRY_DELAY_SECONDS = 1


# ---------------------------------------------------------------------------
# Upload + job creation  (runs in the Flask request thread)
# ---------------------------------------------------------------------------


def start_processing(pdf_file, filename):
    """Save PDF locally, upload to Supabase Storage, create a DB job and
    enqueue it on RabbitMQ.

    The job is atomically claimed (UPLOADED → PROCESSING) before the publish
    so that only one publisher ever enqueues it and a backed-up queue can
    never cause a duplicate.  If the publish fails the claim is reverted and
    the job stays UPLOADED so the stuck-job reconciler retries it.

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
    try:
        upload_file(local_path, object_name)
    except Exception:
        # A failed storage upload leaves an orphaned file in uploads/ that is
        # not in Supabase, has no job row and is never indexed.  Remove the
        # local artifact and surface the failure so it is not silent.
        try:
            os.remove(local_path)
        except OSError:
            pass
        raise
    job_id = create_job(object_name, object_name)
    if not claim_upload(job_id):
        logger.warning("[Processing] Job %s already claimed elsewhere", job_id)
        return object_name, job_id

    try:
        enqueue_job(job_id)
    except Exception:
        release_upload(job_id)
        logger.error("[Processing] Job %s created but not enqueued — it will "
                     "be picked up by the stuck-job reconciler", job_id)
        raise

    logger.info("[Processing] Job %s created for %s", job_id, object_name)
    return object_name, job_id


# ---------------------------------------------------------------------------
# Queue helpers  (Flask side → RabbitMQ)
# ---------------------------------------------------------------------------


def enqueue_job(job_id, max_retries=None, retry_delay=None):
    """Publish job_id to RabbitMQ for processing.

    Retries a few times with a short backoff so a transient broker blip does
    not wedge the job.  Raises the last exception once retries are exhausted
    so callers can react (the job stays non-terminal in the DB and is retried
    by ``resume_stuck_uploads()``).

    Import of tasks is lazy to avoid a Celery/Flask import cycle.
    """
    from tasks import process_pdf_job

    max_retries = ENQUEUE_RETRIES if max_retries is None else max_retries
    retry_delay = ENQUEUE_RETRY_DELAY_SECONDS if retry_delay is None else retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            process_pdf_job.delay(job_id)
            logger.info("[Queue] job %s received and sent to RabbitMQ", job_id)
            return
        except Exception as exc:
            logger.warning("[Queue] enqueue attempt %d/%d failed for job %s: %s",
                           attempt, max_retries, job_id, exc)
            if attempt < max_retries:
                time.sleep(retry_delay)
    raise


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
        try:
            enqueue_job(jid)
        except Exception as exc:
            logger.error("[Processing]   %s: re-enqueue failed: %s", jid, exc)
    return len(pending)


def resume_stuck_uploads():
    """Re-enqueue jobs that were created but never claimed.

    A job stuck in UPLOADED means the original RabbitMQ publish failed (see
    ``enqueue_job``).  Each job is claimed atomically (UPLOADED → PROCESSING)
    before publishing, so at most one reconciler/worker enqueues it; if the
    publish fails again the claim is reverted and the next pass retries.
    Called periodically by the Flask reconciler thread.
    """
    stuck = get_stuck_uploads()
    if not stuck:
        logger.debug("[Processing] No stuck uploads to resume")
        return 0

    logger.info("[Processing] Resuming %d stuck upload(s) ...", len(stuck))
    n = 0
    for job in stuck:
        jid = job["job_id"]
        if not claim_upload(jid):
            continue
        try:
            enqueue_job(jid)
            n += 1
            logger.info("[Processing]   %s: re-enqueued %s", jid, job["pdf_name"])
        except Exception as exc:
            release_upload(jid)
            logger.error("[Processing]   %s: re-enqueue failed: %s", jid, exc)
    return n


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
        try:
            enqueue_job(jid)
            logger.info("[Processing]   %s: %s", jid, job["pdf_name"])
        except Exception as exc:
            logger.error("[Processing]   %s: re-enqueue failed: %s", jid, exc)
    return len(failed)
