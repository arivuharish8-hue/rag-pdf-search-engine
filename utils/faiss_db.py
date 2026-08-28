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
import time

import faiss
import numpy as np

from utils.bm25 import search as bm25_search, build as bm25_build
from utils.file_lock import FileLock

logger = logging.getLogger(__name__)

# Number of times to retry an ``os.replace`` when the target file is briefly
# open by another process (Windows raises PermissionError in that case).
ATOMIC_REPLACE_RETRIES = int(os.getenv("FAISS_REPLACE_RETRIES", "10"))

# Verified-removal polling: after removing a PDF we release the file lock and
# re-check the persisted files, because a Celery worker that was mid-batch can
# re-add one batch after our save.  These constants bound how long we wait and
# re-remove before declaring the deletion successful.
REMOVE_VERIFY_RETRIES = int(os.getenv("FAISS_REMOVE_VERIFY_RETRIES", "5"))
REMOVE_VERIFY_DELAY = float(os.getenv("FAISS_REMOVE_VERIFY_DELAY", "0.6"))

# Hybrid (FAISS + BM25) retrieval settings.
FAISS_WEIGHT = float(os.getenv("FAISS_WEIGHT", "0.5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
FAISS_CANDIDATES = int(os.getenv("FAISS_CANDIDATES", "30"))
BM25_CANDIDATES = int(os.getenv("BM25_CANDIDATES", "30"))

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
_METADATA_VERSION = 0  # bumped on every metadata change; drives BM25 rebuilds


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

    _mark_metadata_changed()


def _mark_metadata_changed():
    """Bump the metadata version and rebuild the BM25 index from it.

    Called whenever ``metadata`` is reloaded or mutated (still under the
    FAISS / file locks held by the caller).  Keeping FAISS vectors, the
    metadata list and the BM25 index on the same version means a deleted PDF
    can never resurface through keyword search.
    """
    global _METADATA_VERSION
    _METADATA_VERSION += 1
    bm25_build(metadata, _METADATA_VERSION)


def _atomic_replace(src, dst):
    """``os.replace`` that survives transient Windows file-lock contention.

    On Windows ``os.replace`` fails with ``PermissionError`` if another
    process has ``dst`` open at that instant (e.g. a Celery worker calling
    ``total_vectors()``/``search()``, which open the index file via
    ``os.path.getmtime`` / ``faiss.read_index``).  A stale ``os.replace``
    failure is fatal for a delete: ``faiss.index``/``metadata.pkl`` would
    simply not be rewritten.  Retrying closes that window.
    """
    for attempt in range(1, ATOMIC_REPLACE_RETRIES + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == ATOMIC_REPLACE_RETRIES:
                raise
            logger.debug("[FAISS] os.replace(%s) blocked by an open handle — "
                         "retry %d/%d", dst, attempt, ATOMIC_REPLACE_RETRIES)
            time.sleep(0.05 * attempt)


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
    _atomic_replace(tmp_index, INDEX_FILE)

    tmp_meta = METADATA_FILE + ".tmp"
    with open(tmp_meta, "wb") as f:
        pickle.dump(metadata, f)
    _atomic_replace(tmp_meta, METADATA_FILE)

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
        # faiss.index no longer exists on disk — the authoritative on-disk
        # state is empty.  Reload (→ empty) instead of serving the stale
        # in-memory index forever: a delete that emptied the database would
        # otherwise keep reporting phantom vectors and old search hits.
        with FileLock(LOCK_FILE):
            _load()
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
        # See _maybe_reload: a missing index file means empty on disk.
        _load()
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

    Operates on a FRESH read of the persisted state (``_load()``) rather than
    a worker's possibly-stale in-memory copy, so the removal is authoritative
    even with multiple gunicorn/Celery processes — a stale worker would
    otherwise compute ``removed == 0`` and never persist anything.

    When the index and metadata counts match 1:1, the surviving vectors are
    copied from the existing index with ``reconstruct_n`` (no re-embedding).
    When they do NOT match (stale / corrupt state where individual removal
    cannot be trusted), the whole index is rebuilt from the remaining
    metadata instead.  Both paths persist faiss.index + metadata.pkl
    atomically.

    Returns the number of metadata entries removed (0 when the PDF had no
    indexed chunks — callers should not treat that as a failure by itself).
    """
    global index, metadata
    _load()  # authoritative on-disk state — never remove against stale memory
    before = len(metadata)
    keep_positions = [i for i, m in enumerate(metadata)
                      if m["pdf_name"] != pdf_name]
    removed = before - len(keep_positions)

    if not removed:
        logger.debug("[FAISS] remove_pdf: %s not found in persisted state",
                     pdf_name)
        return 0

    keep_set = set(keep_positions)
    if index is not None and index.ntotal == before:
        new_index = faiss.IndexFlatIP(DIMENSION)
        if index.ntotal > 0:
            positions = [i for i in keep_positions if i < index.ntotal]
            if positions:
                vectors = index.reconstruct_n(0, index.ntotal)[positions]
                new_index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        index = new_index
        metadata = [m for i, m in enumerate(metadata) if i in keep_set]
        _mark_metadata_changed()
        _save()
    else:
        metadata = [m for i, m in enumerate(metadata) if i in keep_set]
        _rebuild_from_metadata()

    logger.info(
        "[FAISS] remove_pdf: %s — removed %d entries, %d vectors remain",
        pdf_name, removed, index.ntotal,
    )
    return removed


def _rebuild_from_metadata():
    """Rebuild index vectors from the in-memory metadata list."""
    global index, metadata
    index = faiss.IndexFlatIP(DIMENSION)

    if not metadata:
        _save()
        logger.info("[FAISS] Rebuilt empty index (no metadata)")
        _mark_metadata_changed()
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
    _mark_metadata_changed()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_documents(chunks, embeddings, guard=None):
    """Add chunk embeddings + metadata, optionally gated by ``guard``.

    ``guard`` is a zero-arg callable evaluated *while holding both locks*,
    right before the write.  A Celery worker uses it to re-check job
    aliveness so a PDF deleted concurrently cannot have its vectors
    re-introduced after the delete's FAISS removal (delete removes the
    processing_jobs row *before* acquiring the FAISS locks, so the guard and
    the removal serialize on the file lock).

    Returns the number of chunks added (0 when ``guard`` aborted the write).
    """
    global index, metadata
    with _FAISS_LOCK:
        with FileLock(LOCK_FILE):
            _prepare_for_write()

            if guard is not None and not guard():
                logger.info("[FAISS] add_documents aborted by guard "
                            "(no vectors written)")
                return 0

            n_before = index.ntotal
            embeddings = np.asarray(embeddings, dtype=np.float32)
            index.add(embeddings)
            metadata.extend(chunks)
            _mark_metadata_changed()
            _save()
            logger.info("[FAISS] add_documents: %d → %d vectors (+%d)",
                        n_before, index.ntotal, len(chunks))
            return len(chunks)


def _persisted_chunks_for(pdf_name):
    """Count a PDF's chunks by reading metadata.pkl straight from disk."""
    if not os.path.exists(METADATA_FILE):
        return 0
    try:
        with open(METADATA_FILE, "rb") as f:
            persisted = pickle.load(f)
    except Exception:
        logger.error("[FAISS] metadata.pkl unreadable — cannot verify removal")
        return -1
    return sum(1 for m in persisted if m["pdf_name"] == pdf_name)


def _verify_persisted_clean(pdf_name):
    """Poll metadata.pkl over a bounded window, re-removing late re-adds.

    A Celery worker that passed ``_job_alive()`` before a delete can still be
    mid-batch: after our save it re-adds one batch of this PDF via
    ``add_documents``.  Release the lock and poll the persisted files across
    the FULL window (even when the first poll is clean) so a batch that lands
    shortly after our write is caught and re-removed.  Caller holds
    ``_FAISS_LOCK``.
    """
    for attempt in range(1, REMOVE_VERIFY_RETRIES + 1):
        time.sleep(REMOVE_VERIFY_DELAY)
        with FileLock(LOCK_FILE):
            _load()  # fresh from disk (worker may have re-added)
            if sum(1 for m in metadata if m["pdf_name"] == pdf_name):
                logger.warning(
                    "[FAISS] remove_pdf: %s re-appeared (late worker batch) — "
                    "re-removing (verify %d/%d)",
                    pdf_name, attempt, REMOVE_VERIFY_RETRIES,
                )
                _remove_pdf_unlocked(pdf_name)

    present = _persisted_chunks_for(pdf_name)
    if present > 0:
        logger.error(
            "[FAISS] remove_pdf: %s still has %d persisted chunks after %d "
            "verify passes — deletion did NOT persist reliably",
            pdf_name, present, REMOVE_VERIFY_RETRIES,
        )
    return present


def persisted_chunk_count(pdf_name):
    """Number of a PDF's chunks that are still on disk (direct metadata.pkl read).

    Used after a delete to prove the removal actually persisted, independent of
    the in-memory copy.
    """
    return _persisted_chunks_for(pdf_name)


def remove_pdf(pdf_name):
    """Remove one PDF's vectors + metadata from disk.

    Re-checks the persisted files after removal and re-removes anything a
    mid-batch worker re-added, so the delete is actually durable — not just
    applied to the in-memory copy.

    Returns the number of metadata entries removed (0 if ``pdf_name`` has no
    indexed chunks).
    """
    with _FAISS_LOCK:
        with FileLock(LOCK_FILE):
            _prepare_for_write()
            removed = _remove_pdf_unlocked(pdf_name)

        # Always run the verification window: a late worker batch can re-add
        # chunks even when the in-memory removal found nothing (it may have
        # landed between our _prepare_for_write and this call).
        _verify_persisted_clean(pdf_name)
        return removed


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


def _faiss_candidates(query_embedding, top_k):
    """FAISS semantic candidates as ``(raw_score, metadata_index)`` pairs.

    Preserves the existing search semantics (IndexFlatIP dot product,
    dedupe by chunk key) but returns more candidates than the final top-k
    so BM25-only hits are not starved out.

    The similarity floor (0.01) is intentionally very low to allow the hybrid
    scorer and reranker to make the final relevance decision.  A strict
    floor here starves paraphrase queries whose FAISS embedding similarity
    is low but whose BM25 keyword overlap is strong.  The cross-encoder
    reranker (rerank_score > 0) is the true relevance gate.
    """
    query_embedding = np.asarray([query_embedding], dtype=np.float32)
    scores, ids = index.search(query_embedding, top_k * 5)

    results = []
    seen = set()

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        if score < 0.01:
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

        results.append((float(score), idx))
        if len(results) >= top_k:
            break

    return results


def _normalize_scores(merged):
    """Min-max normalize each modality over its own candidate scores.

    Returns a list of result dicts with normalized ``faiss_score`` /
    ``bm25_score`` and the weighted ``hybrid_score`` (also exposed as
    ``score`` for the existing UI), sorted by hybrid score descending.
    """
    faiss_vals = [v["faiss_raw"] for v in merged.values()
                  if v["faiss_raw"] is not None]
    bm25_vals = [v["bm25_raw"] for v in merged.values()
                 if v["bm25_raw"] is not None]

    def norm(value, values):
        if value is None or not values:
            return 0.0
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return 1.0
        return (value - lo) / (hi - lo)

    results = []
    for v in merged.values():
        faiss_score = norm(v["faiss_raw"], faiss_vals)
        bm25_score = norm(v["bm25_raw"], bm25_vals)
        hybrid_score = FAISS_WEIGHT * faiss_score + BM25_WEIGHT * bm25_score

        result = v["item"].copy()
        result["faiss_score"] = round(faiss_score, 4)
        result["bm25_score"] = round(bm25_score, 4)
        result["hybrid_score"] = round(hybrid_score, 4)
        result["score"] = result["hybrid_score"]
        results.append(result)

    results.sort(key=lambda r: r["hybrid_score"], reverse=True)
    return results


_BM25_STOPWORDS = frozenset(
    "a an and are as at be but by for if in is it of on or so the to was "
    "we what when where which who whom why how do does did have has had "
    "this that these those i you he she they them your my our their his her "
    "me us not no nor too very can could should would may might must will "
    "shall just also into over under about from with without give tell show "
    "name list describe explain know".split()
)


def _bm25_query_tokens(query_text):
    """Return query tokens with stopwords stripped for BM25 scoring.

    Stopwords like ``what``, ``are``, ``the``, ``of`` appear in nearly every
    prose document and artificially inflate their BM25 scores, pushing
    short / structured documents (resumes, lists) below the top-k cutoff.
    Stripping them lets content words (``achievements``, ``haridass``) drive
    ranking exclusively.
    """
    from utils.bm25 import tokenize as _tok
    tokens = _tok(query_text)
    return [t for t in tokens if t not in _BM25_STOPWORDS and len(t) >= 2]


def hybrid_search(query_text, query_embedding, top_k=5):
    """Hybrid retrieval: FAISS semantic + BM25 keyword, fused by score.

    Retrieves ``FAISS_CANDIDATES`` (default 10) semantic hits and
    ``BM25_CANDIDATES`` (default 10) keyword hits, merges them by chunk key
    ``(pdf_name, page, page, chunk)``, min-max normalizes each modality, then
    combines them as ``FAISS_WEIGHT * norm_faiss + BM25_WEIGHT * norm_bm25``
    and returns the top ``top_k`` chunks.

    BM25 is kept consistent with FAISS by rebuilding from the exact metadata
    list whenever the collection changes (see ``_mark_metadata_changed``), so
    deleted PDFs never resurface through keyword search.
    """
    with _FAISS_LOCK:
        if index is None:
            with FileLock(LOCK_FILE):
                _load()
        else:
            _maybe_reload()

        if index.ntotal == 0:
            logger.debug("[Hybrid] query=%r — empty index, no candidates",
                         query_text)
            return []

        faiss_candidates = _faiss_candidates(query_embedding, FAISS_CANDIDATES)

        bm25_tokens = _bm25_query_tokens(query_text)
        bm25_query = " ".join(bm25_tokens) if bm25_tokens else query_text
        bm25_candidates = bm25_search(
            bm25_query, metadata, _METADATA_VERSION, BM25_CANDIDATES
        )
        logger.info(
            "[Hybrid] query=%r — FAISS=%d, BM25=%d raw candidates "
            "(bm25_tokens=%s)",
            query_text[:80], len(faiss_candidates), len(bm25_candidates),
            bm25_tokens,
        )
        if faiss_candidates:
            logger.debug("[Hybrid] FAISS top scores: %s",
                         [(round(s, 4), i) for s, i in faiss_candidates[:3]])
        if bm25_candidates:
            logger.debug("[Hybrid] BM25 top scores: %s",
                         [(round(s, 4), i) for s, i in bm25_candidates[:3]])

        merged = {}
        for score, idx in faiss_candidates:
            item = metadata[idx]
            key = (item["pdf_name"], item["page"], item["chunk"])
            merged[key] = {
                "item": item, "faiss_raw": score, "bm25_raw": None,
            }
        for score, idx in bm25_candidates:
            item = metadata[idx]
            key = (item["pdf_name"], item["page"], item["chunk"])
            if key in merged:
                merged[key]["bm25_raw"] = score
            else:
                merged[key] = {
                    "item": item, "faiss_raw": None, "bm25_raw": score,
                }

        scored = _normalize_scores(merged)
        logger.debug(
            "[Hybrid] normalized (faiss, bm25, hybrid): %s",
            [(r["pdf_name"], r["page"], r["chunk"], r["faiss_score"],
              r["bm25_score"], r["hybrid_score"]) for r in scored],
        )
        final = scored[:top_k]
        logger.info(
            "[Hybrid] query=%r — %d merged candidate(s), selected %d",
            query_text, len(scored), len(final),
        )
        return final


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
    """Remove FAISS entries for PDFs no longer in existing_files list.

    Uses verified removal per PDF so a mid-batch worker cannot re-introduce
    the deleted vectors after the persisted write.
    """
    orphaned = []
    with _FAISS_LOCK:
        with FileLock(LOCK_FILE):
            _prepare_for_write()

            existing_set = set(existing_files)
            indexed = set(m["pdf_name"] for m in metadata)
            orphaned = [p for p in indexed if p not in existing_set]

            for pdf_name in orphaned:
                logger.info("[FAISS] Removing orphaned PDF: %s", pdf_name)
                _remove_pdf_unlocked(pdf_name)

    for pdf_name in orphaned:
        _verify_persisted_clean(pdf_name)

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
