"""
Background PDF processing pipeline with checkpoint / resume support.

Flow for each job:
  1. UPLOADED   → create_job()                      (done by start_processing)
  2. EXTRACTING → extract_text_from_pdf()
  3. CHUNKING   → chunk_text()                       (inside extract_text_from_pdf)
  4. EMBEDDING  → create_embeddings()
  5. INDEXING   → add_documents() to FAISS
  6. COMPLETED  → done

On crash / restart the BackgroundWorker picks up pending jobs and resumes
from the last successful checkpoint stored in the database backend.
"""

import logging
import os
import threading
import traceback as tb
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from utils.database import (
    create_job,
    delete_job,
    get_failed_jobs,
    get_job,
    get_pending_jobs,
    save_checkpoint,
    update_job_status,
)
from utils.supabase_storage import upload_file
from utils.pdf_utils import extract_text_from_pdf
from utils.embeddings import create_embeddings
from utils.faiss_db import (
    add_documents,
    get_pdf_chunk_count,
    remove_pdf,
    total_vectors,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_BATCH = 256
UPLOAD_FOLDER = "uploads"
POLL_INTERVAL = 5
EXTRACT_TIMEOUT = 300  # seconds — hard limit for PDF text extraction

# ---------------------------------------------------------------------------
# Upload + job creation  (runs in the Flask request thread)
# ---------------------------------------------------------------------------


def start_processing(pdf_file, filename):
    """Save PDF locally, upload to Supabase Storage, create a DB job.

    Returns immediately — the actual extraction / embedding / indexing
    happens asynchronously in a background thread.
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

    logger.info("[Processing] Job %s created for %s", job_id, object_name)
    return object_name, job_id


# ---------------------------------------------------------------------------
# Core processing  (runs in background threads)
# ---------------------------------------------------------------------------


def _extract_with_timeout(local_path, pdf_name, timeout=EXTRACT_TIMEOUT):
    """Run PDF extraction in a thread pool with a hard timeout."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(extract_text_from_pdf, local_path, pdf_name)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            raise TimeoutError(
                f"PDF extraction timed out after {timeout}s: {pdf_name}"
            )


def _process_job(job_id):
    """Run the full pipeline for one job with checkpoint-aware resume.

    Guarantees:
      - Per-stage exception handling; every error is logged with traceback.
      - The job is always marked FAILED on an unhandled error.
      - No duplicate FAISS vectors (inconsistency forces a clean restart).
      - Checkpoint saved after every batch.
    """
    job = get_job(job_id)
    if job is None:
        logger.warning("[Processing] Job %s not found, skipping", job_id)
        return

    pdf_name = job["storage_path"]
    local_path = os.path.join(UPLOAD_FOLDER, pdf_name)

    if not os.path.exists(local_path):
        update_job_status(job_id, status="FAILED",
                          error_message=f"Local file not found: {local_path}")
        logger.error("[Processing] %s: local file missing (%s)", pdf_name, local_path)
        return

    # Mark job as actively processing so get_pending_jobs() still returns it
    # but the status field reflects intent.
    update_job_status(job_id, status="PROCESSING")

    # ── EXTRACT + CHUNK ──────────────────────────────────────────────
    try:
        logger.info("[Processing] %s: stage EXTRACTING — extracting text", pdf_name)
        update_job_status(job_id, current_stage="EXTRACTING")
        chunks = _extract_with_timeout(local_path, pdf_name)
        logger.info("[Processing] %s: extraction returned %d chunks",
                    pdf_name, len(chunks))
    except Exception as exc:
        logger.exception("[Processing] %s: EXTRACTING failed", pdf_name)
        update_job_status(job_id, status="FAILED",
                          current_stage="EXTRACTING",
                          error_message=f"EXTRACTING failed: {exc}")
        return

    total = len(chunks)

    if total == 0:
        update_job_status(job_id, status="COMPLETED", current_stage="COMPLETED",
                          total_chunks=0, last_processed_chunk=0)
        logger.info("[Processing] %s: no text found, marked COMPLETED", pdf_name)
        return

    # ── CHUNKING (metadata recorded) ─────────────────────────────────
    try:
        logger.info("[Processing] %s: stage CHUNKING — %d chunks total",
                    pdf_name, total)
        update_job_status(job_id, current_stage="CHUNKING", total_chunks=total)
    except Exception as exc:
        logger.exception("[Processing] %s: CHUNKING metadata update failed", pdf_name)
        update_job_status(job_id, status="FAILED",
                          current_stage="CHUNKING",
                          error_message=f"CHUNKING failed: {exc}")
        return

    # ── Determine resume point ───────────────────────────────────────
    # Re-read the job in case a previous run left a checkpoint.
    fresh_job = get_job(job_id)
    last_idx = fresh_job.get("last_processed_chunk", 0) if fresh_job else 0
    faiss_count = get_pdf_chunk_count(pdf_name)

    if faiss_count != last_idx:
        logger.warning(
            "[Processing] %s: FAISS has %d vectors but checkpoint says %d. "
            "Cleaning inconsistent state.",
            pdf_name, faiss_count, last_idx,
        )
        if faiss_count > 0:
            remove_pdf(pdf_name)
        last_idx = 0
        update_job_status(job_id, last_processed_chunk=0)

    start_idx = last_idx

    if start_idx >= total:
        update_job_status(job_id, status="COMPLETED", current_stage="COMPLETED",
                          last_processed_chunk=total)
        logger.info("[Processing] %s: already fully indexed (%d chunks)",
                    pdf_name, total)
        return

    logger.info("[Processing] %s: %d total chunks, resuming from index %d",
                pdf_name, total, start_idx)

    # ── EMBED + INDEX in batches ─────────────────────────────────────
    for i in range(start_idx, total, EMBED_BATCH):
        batch = chunks[i: i + EMBED_BATCH]
        texts = [c["text"] for c in batch]

        # ── EMBEDDING ─────────────────────────────────────────────────
        try:
            logger.info("[Processing] %s: stage EMBEDDING — batch %d..%d (%d texts)",
                        pdf_name, i, min(i + EMBED_BATCH, total), len(texts))
            update_job_status(job_id, current_stage="EMBEDDING",
                              last_processed_chunk=i)
            embeddings = create_embeddings(texts)
            logger.info("[Processing] %s: embeddings generated (shape %s)",
                        pdf_name, embeddings.shape)
        except Exception as exc:
            logger.exception("[Processing] %s: EMBEDDING failed at chunk %d",
                             pdf_name, i)
            update_job_status(job_id, status="FAILED",
                              current_stage="EMBEDDING",
                              error_message=f"EMBEDDING failed at chunk {i}: {exc}")
            return

        # ── INDEXING ──────────────────────────────────────────────────
        try:
            logger.info("[Processing] %s: stage INDEXING — adding %d vectors",
                        pdf_name, len(batch))
            update_job_status(job_id, current_stage="INDEXING",
                              last_processed_chunk=i)
            add_documents(batch, embeddings)
            processed_up_to = i + len(batch)
            save_checkpoint(job_id, processed_up_to,
                            status="PROCESSING", current_stage="INDEXING")
            logger.info("[Processing] %s: %d / %d chunks indexed (%d total FAISS vectors)",
                        pdf_name, processed_up_to, total, total_vectors())
        except Exception as exc:
            logger.exception("[Processing] %s: INDEXING failed at chunk %d",
                             pdf_name, i)
            update_job_status(job_id, status="FAILED",
                              current_stage="INDEXING",
                              error_message=f"INDEXING failed at chunk {i}: {exc}")
            return

    # ── COMPLETED ────────────────────────────────────────────────────
    try:
        update_job_status(job_id, status="COMPLETED", current_stage="COMPLETED",
                          last_processed_chunk=total)
        logger.info("[Processing] %s: COMPLETED — total vectors = %d",
                    pdf_name, total_vectors())
    except Exception as exc:
        logger.exception("[Processing] %s: final status update failed", pdf_name)
        update_job_status(job_id, status="FAILED",
                          error_message=f"Final status update failed: {exc}")


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


def resume_processing():
    """Find all pending (non-terminal) jobs and resume each in a thread."""
    pending = get_pending_jobs()
    if not pending:
        logger.info("[Processing] No pending jobs to resume")
        return

    logger.info("[Processing] Resuming %d pending job(s) ...", len(pending))
    for job in pending:
        jid = job["job_id"]
        with _STATE_LOCK:
            if jid in _ACTIVE_JOBS:
                logger.info("[Processing]   %s: already active, skipping", jid)
                continue
            _ACTIVE_JOBS.add(jid)

        logger.info("[Processing]   %s: %s  (stage=%s, checkpoint=%d)",
                    jid, job["pdf_name"], job["current_stage"],
                    job["last_processed_chunk"])
        threading.Thread(target=_run_wrapper, args=(jid,), daemon=True).start()


def retry_failed_jobs():
    """Reset FAILED jobs to PROCESSING and retry them."""
    failed = get_failed_jobs()
    if not failed:
        logger.info("[Processing] No failed jobs to retry")
        return

    logger.info("[Processing] Retrying %d failed job(s) ...", len(failed))
    for job in failed:
        jid = job["job_id"]
        with _STATE_LOCK:
            if jid in _ACTIVE_JOBS:
                logger.info("[Processing]   %s: already active, skipping", jid)
                continue
            _ACTIVE_JOBS.add(jid)

        logger.info("[Processing]   %s: %s", jid, job["pdf_name"])
        update_job_status(jid, status="PROCESSING", error_message=None)
        threading.Thread(target=_run_wrapper, args=(jid,), daemon=True).start()


# ---------------------------------------------------------------------------
# Active-job tracking (shared across resume / retry / worker)
# ---------------------------------------------------------------------------

_ACTIVE_JOBS = set()
_STATE_LOCK = threading.Lock()


def _run_wrapper(job_id):
    """Wrap _process_job so ACTIVE_JOBS is always cleaned up."""
    logger.info("[Processing] Thread started for job %s", job_id)
    try:
        _process_job(job_id)
    except Exception as exc:
        # Safety net — anything that escapes _process_job's own handlers.
        logger.exception("[Processing] Unhandled exception in job %s", job_id)
        try:
            update_job_status(job_id, status="FAILED",
                              error_message=f"Unhandled: {exc}")
        except Exception:
            logger.exception("[Processing] Failed to mark job %s as FAILED", job_id)
    finally:
        with _STATE_LOCK:
            _ACTIVE_JOBS.discard(job_id)
            logger.info("[Processing] Thread finished for job %s", job_id)


# ---------------------------------------------------------------------------
# Background worker  (polling loop)
# ---------------------------------------------------------------------------


class BackgroundWorker:
    """Daemon thread that polls for pending jobs every *POLL_INTERVAL*
    seconds and dispatches each to a separate worker thread.

    Singleton — use the module-level ``background_worker`` instance.
    """

    def __init__(self):
        self._running = False

    def start(self):
        """Start the polling loop (idempotent)."""
        if self._running:
            logger.info("[Processing] Background worker already running, ignoring")
            return
        self._running = True
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()
        logger.info("[Processing] Background worker started (poll=%ds)", POLL_INTERVAL)

    def _loop(self):
        import time

        while True:
            try:
                pending = get_pending_jobs()
                if pending:
                    logger.debug("[Processing] Worker poll: %d pending job(s)",
                                 len(pending))
                for job in pending:
                    jid = job["job_id"]
                    with _STATE_LOCK:
                        if jid in _ACTIVE_JOBS:
                            continue
                        _ACTIVE_JOBS.add(jid)
                    threading.Thread(target=_run_wrapper, args=(jid,),
                                     daemon=True).start()
            except Exception as exc:
                logger.error("[Processing] Background worker error: %s", exc,
                             exc_info=True)
            time.sleep(POLL_INTERVAL)


# Singleton — importers get the same instance.
background_worker = BackgroundWorker()
