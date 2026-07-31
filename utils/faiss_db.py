"""FAISS index management with rebuild and sync support.

Thread-safe (``_FAISS_LOCK``) AND process-safe (``FileLock``): every disk
read / write acquires the cross-process lock so the Flask process and
multiple Celery workers can coexist without corrupting ``faiss.index`` or
``metadata.pkl``.

Writes are atomic (temp file + ``os.replace``) and the in-memory copy is
refreshed from disk whenever another process has persisted newer data
(mtime-based reload).

Lock layering (always in this order, never nested in the other direction):

    _FAISS_LOCK          # in-process thread lock (reentrant)
        FileLock         # cross-process file lock
            _load/_save
"""

import logging
import os
import pickle
import threading

import faiss
import numpy as np

from utils.file_lock import FileLock

logger = logging.getLogger(__name__)

DATABASE_DIR = "database"
INDEX_FILE = os.path.join(DATABASE_DIR, "faiss.index")
METADATA_FILE = os.path.join(DATABASE_DIR, "metadata.pkl")
LOCK_FILE = os.path.join(DATABASE_DIR, ".faiss.lock")

os.makedirs(DATABASE_DIR, exist_ok=True)

DIMENSION = 384

index = None
metadata = []
_FAISS_LOCK = threading.RLock()
_LOADED_AT = 0.0  # mtime of INDEX_FILE when index/metadata were last loaded


# ---------------------------------------------------------------------------
# Internal helpers — assume the caller already holds the appropriate locks.
# ---------------------------------------------------------------------------


def _load():
    """Load index + metadata from disk into memory."""
    global index, metadata, _LOADED_AT
    if os.path.exists(INDEX_FILE):
        logger.debug("[FAISS] Loading index from %s", INDEX_FILE)
        index = faiss.read_index(INDEX_FILE)
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "rb") as f:
                metadata = pickle.load(f)
            logger.debug("[FAISS] Loaded %d metadata entries", len(metadata))
        else:
            metadata = []
            logger.debug("[FAISS] No metadata file, starting fresh")
    else:
        logger.debug("[FAISS] No index file, creating new empty index")
        index = faiss.IndexFlatIP(DIMENSION)
        metadata = []
    try:
        _LOADED_AT = os.path.getmtime(INDEX_FILE)
    except OSError:
        _LOADED_AT = 0.0


def _save():
    """Persist index + metadata atomically.

    Writes to temp files then ``os.replace`` so a concurrent reader always
    sees either the previous or the new complete state — never a partial one.
    """
    global index, metadata, _LOADED_AT
    logger.debug("[FAISS] Saving index (%d vectors) and %d metadata entries",
                 index.ntotal, len(metadata))

    tmp_index = INDEX_FILE + ".tmp"
    faiss.write_index(index, tmp_index)
    os.replace(tmp_index, INDEX_FILE)

    tmp_meta = METADATA_FILE + ".tmp"
    with open(tmp_meta, "wb") as f:
        pickle.dump(metadata, f)
    os.replace(tmp_meta, METADATA_FILE)

    _LOADED_AT = os.path.getmtime(INDEX_FILE)


def _maybe_reload():
    """Reload from disk if another process persisted newer data.

    Caller holds ``_FAISS_LOCK`` only.  Cheap no-op when nobody else wrote
    (mtime unchanged), so a worker does not re-read its own file each batch.
    """
    global _LOADED_AT
    try:
        mtime = os.path.getmtime(INDEX_FILE)
    except OSError:
        return
    if mtime > _LOADED_AT:
        with FileLock(LOCK_FILE):
            _load()


def _maybe_reload_under_lock():
    """Same as ``_maybe_reload`` but the caller already holds ``FileLock``."""
    global _LOADED_AT
    try:
        mtime = os.path.getmtime(INDEX_FILE)
    except OSError:
        return
    if mtime > _LOADED_AT:
        _load()


def _prepare_for_write():
    """Bring state current before mutating. Caller holds both locks."""
    if index is None:
        _load()
    else:
        _maybe_reload_under_lock()


def _remove_pdf_unlocked(pdf_name):
    """Remove one PDF's vectors + metadata. Caller holds both locks.

    The surviving vectors are copied from the existing index with
    ``reconstruct_n`` (no re-embedding), so removal keeps the exact stored
    embeddings, preserves index/metadata order, and runs in seconds even for
    large corpora.  Persists faiss.index + metadata.pkl atomically via
    ``_save()``.
    """
    global index, metadata
    before = len(metadata)
    keep_positions = [i for i, m in enumerate(metadata)
                      if m["pdf_name"] != pdf_name]
    removed = before - len(keep_positions)

    if not removed:
        logger.debug("[FAISS] remove_pdf: %s not found", pdf_name)
        return

    new_index = faiss.IndexFlatIP(DIMENSION)
    if index is not None and index.ntotal > 0:
        n = min(index.ntotal, before)
        positions = [i for i in keep_positions if i < n]
        if positions:
            vectors = index.reconstruct_n(0, n)[positions]
            new_index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    index = new_index
    keep_set = set(keep_positions)
    metadata = [m for i, m in enumerate(metadata) if i in keep_set]
    _save()
    logger.info(
        "[FAISS] remove_pdf: %s — removed %d entries, %d vectors remain",
        pdf_name, removed, index.ntotal,
    )


def _rebuild_from_metadata():
    """Rebuild index vectors from the in-memory metadata list."""
    global index, metadata
    index = faiss.IndexFlatIP(DIMENSION)

    if not metadata:
        _save()
        logger.info("[FAISS] Rebuilt empty index (no metadata)")
        return

    from utils.embeddings import create_embeddings

    texts = [m["text"] for m in metadata]
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        embeddings = create_embeddings(batch)
        index.add(np.asarray(embeddings, dtype=np.float32))

    _save()
    logger.info("[FAISS] Rebuilt index with %d vectors from %d chunks",
                index.ntotal, len(metadata))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_documents(chunks, embeddings):
    global index, metadata
    with _FAISS_LOCK:
        with FileLock(LOCK_FILE):
            _prepare_for_write()

            n_before = index.ntotal
            embeddings = np.asarray(embeddings, dtype=np.float32)
            index.add(embeddings)
            metadata.extend(chunks)
            _save()
            logger.info("[FAISS] add_documents: %d → %d vectors (+%d)",
                        n_before, index.ntotal, len(chunks))


def remove_pdf(pdf_name):
    with _FAISS_LOCK:
        with FileLock(LOCK_FILE):
            _prepare_for_write()
            _remove_pdf_unlocked(pdf_name)


def rebuild_index():
    with _FAISS_LOCK:
        with FileLock(LOCK_FILE):
            _prepare_for_write()
            _rebuild_from_metadata()


def search(query_embedding, top_k=3):
    with _FAISS_LOCK:
        if index is None:
            with FileLock(LOCK_FILE):
                _load()
        else:
            _maybe_reload()

        if index.ntotal == 0:
            return []

        query_embedding = np.asarray([query_embedding], dtype=np.float32)
        scores, ids = index.search(query_embedding, top_k * 5)

        results = []
        seen = set()

        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            if score < 0.40:
                continue
            # Guard against a momentary index/metadata mismatch from a
            # concurrent writer (index is always at least as new as metadata).
            if idx >= len(metadata):
                continue

            item = metadata[idx]
            key = (item["pdf_name"], item["page"], item["chunk"])

            if key in seen:
                continue
            seen.add(key)

            result = item.copy()
            result["score"] = round(float(score), 4)
            results.append(result)

            if len(results) >= top_k:
                break

        return results


def total_vectors():
    with _FAISS_LOCK:
        if index is None:
            with FileLock(LOCK_FILE):
                _load()
        else:
            _maybe_reload()
        return index.ntotal


def get_pdf_chunk_count(pdf_name):
    with _FAISS_LOCK:
        if index is None:
            with FileLock(LOCK_FILE):
                _load()
        else:
            _maybe_reload()
        return sum(1 for m in metadata if m["pdf_name"] == pdf_name)


def get_indexed_pdfs():
    with _FAISS_LOCK:
        if index is None:
            with FileLock(LOCK_FILE):
                _load()
        else:
            _maybe_reload()
        return list(set(m["pdf_name"] for m in metadata))


def sync_with_files(existing_files):
    """Remove FAISS entries for PDFs no longer in existing_files list."""
    with _FAISS_LOCK:
        with FileLock(LOCK_FILE):
            _prepare_for_write()

            existing_set = set(existing_files)
            indexed = set(m["pdf_name"] for m in metadata)
            orphaned = [p for p in indexed if p not in existing_set]

            for pdf_name in orphaned:
                logger.info("[FAISS] Removing orphaned PDF: %s", pdf_name)
                _remove_pdf_unlocked(pdf_name)

            return orphaned


def save_all():
    """Public helper: atomically persist current index + metadata to disk.

    Used by the Celery "metadata update" stage so metadata.pkl is guaranteed
    to be flushed and consistent with faiss.index.
    """
    with _FAISS_LOCK:
        with FileLock(LOCK_FILE):
            _prepare_for_write()
            _save()
            return index.ntotal, len(metadata)


_load()
