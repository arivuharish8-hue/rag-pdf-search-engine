"""Celery tasks — the PDF processing pipeline.

Message flow (each task enqueues the next one on success):

    process_pdf_job ──▶ download_pdf ──▶ extract_and_chunk ──▶ embed_chunks
        ──▶ index_chunks ──▶ update_metadata ──▶ finalize

Design notes
------------
- Fault tolerance: every stage task is registered with ``acks_late`` (see
  celery_config).  If a worker crashes mid-task, RabbitMQ redelivers the
  message and the stage re-runs.  Stages are idempotent: chunks are cached on
  disk, the FAISS/checkpoint consistency check clears partial state, and
  terminal-status guards stop late redeliveries from touching completed jobs.
- Duplicate protection: ``process_pdf_job`` atomically claims the job via
  ``try_claim_job`` (a non-terminal job can only be claimed once at a time).
  FAISS writes are additionally serialized by a cross-process file lock.
- Retries: a stage that fails calls ``self.retry()`` with exponential backoff.
  Once retries are exhausted the job is marked FAILED (with error_message)
  and the pipeline stops.
"""

import logging
import os
import pickle

from celery_app import celery_app
from utils.database import (
    get_job,
    save_checkpoint,
    try_claim_job,
    update_job_status,
)
from utils.embeddings import create_embeddings
from utils.faiss_db import (
    add_documents,
    get_pdf_chunk_count,
    remove_pdf,
    save_all,
    total_vectors,
)
from utils.pdf_utils import extract_text_from_pdf
from utils.supabase_storage import download_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_BATCH = 256
UPLOAD_FOLDER = "uploads"
CHUNKS_DIR = os.path.join("database", "chunks")
MAX_RETRIES = int(os.getenv("CELERY_TASK_MAX_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = 10  # base — doubles on every retry

TERMINAL_STAGES = ("COMPLETED", "FAILED")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks_path(job_id):
    return os.path.join(CHUNKS_DIR, f"{job_id}.pkl")


def _embeddings_path(job_id):
    return os.path.join(CHUNKS_DIR, f"{job_id}_emb.pkl")


def _load_chunks(job_id):
    path = _chunks_path(job_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"chunk cache missing for job {job_id}: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_embeddings(job_id):
    path = _embeddings_path(job_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"embedding cache missing for job {job_id}: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _cache_chunks(job_id, chunks):
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    path = _chunks_path(job_id)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(chunks, f)
    os.replace(tmp, path)


def _cache_embeddings(job_id, batches):
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    path = _embeddings_path(job_id)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(batches, f)
    os.replace(tmp, path)


def _cleanup_caches(job_id):
    for path in (_chunks_path(job_id), _embeddings_path(job_id)):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            logger.warning("[%s] cache cleanup failed for %s: %s",
                           job_id, path, exc)


def _job_alive(job_id):
    """Return the job dict if it may still be processed, else None."""
    job = get_job(job_id)
    if job is None:
        logger.warning("[%s] job not found — skipping", job_id)
        return None
    if job["status"] in TERMINAL_STAGES:
        logger.info("[%s] job already %s — skipping", job_id, job["status"])
        return None
    return job


def _retry_or_fail(self, job_id, stage, exc):
    """Retry the current stage with backoff, or mark the job FAILED when
    retries are exhausted (replacing the old manual background retry)."""
    logger.error("[%s] stage %s failed: %s", job_id, stage, exc, exc_info=True)

    attempts = self.request.retries + 1
    remaining = self.max_retries

    if remaining and attempts <= remaining:
        update_job_status(
            job_id,
            current_stage=stage,
            error_message=f"{stage} failed (attempt {attempts}/{remaining}): {exc}",
        )
        logger.info("[%s] retrying %s in %ss (attempt %d/%d)",
                    job_id, stage, RETRY_BACKOFF_SECONDS * (2 ** self.request.retries),
                    attempts, remaining)
        raise self.retry(exc=exc, countdown=RETRY_BACKOFF_SECONDS * (2 ** self.request.retries))

    update_job_status(
        job_id,
        status="FAILED",
        current_stage=stage,
        error_message=f"{stage} failed: {exc}",
    )
    raise


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@celery_app.task(bind=True, name="pdf.process")
def process_pdf_job(self, job_id):
    """Entry point published by the Flask upload / resume endpoints.

    Atomically claims the job (job_id lock).  On success dispatches the
    pipeline; on a failed claim the job is already being processed by another
    worker, so this run does nothing.
    """
    job = _job_alive(job_id)
    if job is None:
        return {"job_id": job_id, "status": "skipped"}

    if not try_claim_job(job_id):
        logger.info("[%s] job already claimed by another worker — skipping",
                    job_id)
        return {"job_id": job_id, "status": "claimed_elsewhere"}

    logger.info("[%s] job claimed — dispatching pipeline", job_id)
    download_pdf.delay(job_id)
    return {"job_id": job_id, "status": "dispatched"}


# ---------------------------------------------------------------------------
# Stage 1 — Download PDF
# ---------------------------------------------------------------------------


@celery_app.task(bind=True, name="pdf.download", max_retries=MAX_RETRIES)
def download_pdf(self, job_id):
    """Download the PDF from Supabase Storage to the shared uploads folder.

    Skips the download when the file is already present (upload flow stores a
    local copy, and crash-redelivery must be idempotent).
    """
    job = _job_alive(job_id)
    if job is None:
        return job_id

    pdf_name = job["storage_path"]
    local_path = os.path.join(UPLOAD_FOLDER, pdf_name)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    if os.path.exists(local_path):
        logger.info("[%s] download: %s already present locally", job_id, pdf_name)
    else:
        try:
            update_job_status(job_id, current_stage="DOWNLOADING")
            download_file(pdf_name, local_path)
            logger.info("[%s] download: fetched %s from Supabase", job_id, pdf_name)
        except Exception as exc:
            return _retry_or_fail(self, job_id, "DOWNLOADING", exc)

    update_job_status(job_id, current_stage="EXTRACTING")
    extract_and_chunk.delay(job_id)
    return job_id


# ---------------------------------------------------------------------------
# Stage 2 — Extract text + chunk
# ---------------------------------------------------------------------------


@celery_app.task(bind=True, name="pdf.extract_chunk", max_retries=MAX_RETRIES)
def extract_and_chunk(self, job_id):
    """Extract page text, chunk it and cache the chunks on disk.

    Extraction and chunking stay together (as in ``extract_text_from_pdf``)
    but the DB records both stage transitions.  A PDF with no extractable
    text completes immediately with zero chunks.
    """
    job = _job_alive(job_id)
    if job is None:
        return job_id

    pdf_name = job["storage_path"]
    local_path = os.path.join(UPLOAD_FOLDER, pdf_name)

    if os.path.exists(_chunks_path(job_id)):
        with open(_chunks_path(job_id), "rb") as f:
            chunks = pickle.load(f)
        logger.info("[%s] extract: using cached chunks (%d)", job_id, len(chunks))
    else:
        if not os.path.exists(local_path):
            return _retry_or_fail(
                self, job_id, "EXTRACTING",
                FileNotFoundError(f"local file missing: {local_path}"),
            )
        try:
            update_job_status(job_id, current_stage="EXTRACTING")
            chunks = extract_text_from_pdf(local_path, pdf_name)
            logger.info("[%s] extract: %d chunks from %s", job_id, len(chunks), pdf_name)

            update_job_status(job_id, current_stage="CHUNKING",
                              total_chunks=len(chunks))
            _cache_chunks(job_id, chunks)
            logger.info("[%s] extract: cached chunks to disk", job_id)
        except Exception as exc:
            return _retry_or_fail(self, job_id, "EXTRACTING", exc)

    if len(chunks) == 0:
        update_job_status(job_id, status="COMPLETED", current_stage="COMPLETED",
                          total_chunks=0, last_processed_chunk=0)
        logger.info("[%s] extract: no text found, marked COMPLETED", job_id)
        _cleanup_caches(job_id)
        return job_id

    embed_chunks.delay(job_id)
    return job_id


# ---------------------------------------------------------------------------
# Stage 3 — Embedding
# ---------------------------------------------------------------------------


@celery_app.task(bind=True, name="pdf.embed", max_retries=MAX_RETRIES)
def embed_chunks(self, job_id):
    """Generate embeddings for every chunk, in batches, and cache them.

    Batch-wise progress is recorded via last_processed_chunk.  Re-runs
    overwrite the cache (idempotent).
    """
    job = _job_alive(job_id)
    if job is None:
        return job_id

    try:
        chunks = _load_chunks(job_id)
    except Exception as exc:
        return _retry_or_fail(self, job_id, "EMBEDDING", exc)

    total = len(chunks)
    update_job_status(job_id, current_stage="EMBEDDING", total_chunks=total)
    logger.info("[%s] embed: %d chunks", job_id, total)

    batches = []
    try:
        for i in range(0, total, EMBED_BATCH):
            batch = chunks[i: i + EMBED_BATCH]
            texts = [c["text"] for c in batch]
            logger.info("[%s] embed: batch %d..%d (%d texts)",
                        job_id, i, min(i + EMBED_BATCH, total), len(texts))
            vectors = create_embeddings(texts)
            batches.append((i, vectors))
            update_job_status(job_id, current_stage="EMBEDDING",
                              last_processed_chunk=min(i + len(batch), total))
            logger.info("[%s] embed: batch done (shape %s)",
                        job_id, vectors.shape)

        _cache_embeddings(job_id, batches)
        logger.info("[%s] embed: cached %d embedding batch(es)",
                    job_id, len(batches))
    except Exception as exc:
        return _retry_or_fail(self, job_id, "EMBEDDING", exc)

    index_chunks.delay(job_id)
    return job_id


# ---------------------------------------------------------------------------
# Stage 4 — FAISS indexing
# ---------------------------------------------------------------------------


@celery_app.task(bind=True, name="pdf.index", max_retries=MAX_RETRIES)
def index_chunks(self, job_id):
    """Add chunk embeddings to the FAISS index, batch by batch.

    Resume/consistency logic (ported from the original pipeline): the number
    of vectors already in FAISS for this PDF must match the checkpoint.  Any
    mismatch (crash mid-index, redelivered batch, stale partial state) clears
    the PDF's vectors and restarts indexing from zero.

    The job is re-checked before every batch: if the PDF was deleted while
    this stage was running, the job row is already gone and further batches
    must NOT be added, otherwise the delete's FAISS removal would be undone.
    """
    job = _job_alive(job_id)
    if job is None:
        return job_id

    pdf_name = job["storage_path"]

    try:
        chunks = _load_chunks(job_id)
        batches = _load_embeddings(job_id)
    except Exception as exc:
        return _retry_or_fail(self, job_id, "INDEXING", exc)

    total = len(chunks)

    fresh = get_job(job_id)
    last_idx = fresh.get("last_processed_chunk", 0) if fresh else 0
    faiss_count = get_pdf_chunk_count(pdf_name)

    if faiss_count != last_idx:
        logger.warning(
            "[%s] index: FAISS has %d vectors but checkpoint says %d — "
            "cleaning inconsistent state for %s",
            job_id, faiss_count, last_idx, pdf_name,
        )
        if faiss_count > 0:
            remove_pdf(pdf_name)
        last_idx = 0
        update_job_status(job_id, last_processed_chunk=0)

    try:
        for start, vectors in batches:
            if start < last_idx:
                logger.info("[%s] index: batch %d already indexed, skipping",
                            job_id, start)
                continue

            # Stop if the job disappeared mid-indexing (e.g. the PDF was
            # deleted while this stage was running).  The delete pipeline
            # removes the processing_jobs row before touching FAISS, so a
            # missing job here means its vectors must not be re-added.
            if _job_alive(job_id) is None:
                logger.info("[%s] index: job deleted during indexing — "
                            "aborting before batch %d", job_id, start)
                return job_id

            batch = chunks[start: start + len(vectors)]
            update_job_status(job_id, current_stage="INDEXING",
                              last_processed_chunk=start)
            # Guard re-checks job aliveness *inside* the FAISS file lock, so
            # a delete that already removed this job's processing_jobs row
            # (before taking the same lock) cannot have its vectors re-added.
            added = add_documents(
                batch, vectors, guard=lambda: _job_alive(job_id) is not None
            )
            if added == 0:
                logger.info("[%s] index: job deleted during indexing — "
                            "aborted before batch %d", job_id, start)
                return job_id

            processed_up_to = start + len(batch)
            save_checkpoint(job_id, processed_up_to,
                            status="PROCESSING", current_stage="INDEXING")
            logger.info(
                "[%s] index: %d / %d chunks indexed (%d total FAISS vectors)",
                job_id, processed_up_to, total, total_vectors(),
            )
    except Exception as exc:
        return _retry_or_fail(self, job_id, "INDEXING", exc)

    update_metadata.delay(job_id)
    return job_id


# ---------------------------------------------------------------------------
# Stage 5 — Metadata update
# ---------------------------------------------------------------------------


@celery_app.task(bind=True, name="pdf.metadata", max_retries=MAX_RETRIES)
def update_metadata(self, job_id):
    """Flush and verify metadata.pkl + faiss.index to disk.

    The index/metadata are already written on every ``add_documents``; this
    stage guarantees an explicit, consistent final flush and records it.
    """
    job = _job_alive(job_id)
    if job is None:
        return job_id

    try:
        update_job_status(job_id, current_stage="METADATA")
        n_vectors, n_entries = save_all()
        logger.info("[%s] metadata: persisted %d vectors, %d entries",
                    job_id, n_vectors, n_entries)
    except Exception as exc:
        return _retry_or_fail(self, job_id, "METADATA", exc)

    finalize.delay(job_id)
    return job_id


# ---------------------------------------------------------------------------
# Stage 6 — Final status update
# ---------------------------------------------------------------------------


@celery_app.task(bind=True, name="pdf.finalize")
def finalize(self, job_id):
    """Mark the job COMPLETED and clean up the on-disk stage caches."""
    job = _job_alive(job_id)
    if job is None:
        return job_id

    total = job.get("total_chunks", 0) or 0
    update_job_status(
        job_id,
        status="COMPLETED",
        current_stage="COMPLETED",
        total_chunks=total,
        last_processed_chunk=total,
        error_message=None,
    )
    logger.info("[%s] COMPLETED — %d chunks indexed (total vectors=%d)",
                job_id, total, total_vectors())

    _cleanup_caches(job_id)
    return job_id
