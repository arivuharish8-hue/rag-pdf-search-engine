"""FAISS index management with rebuild and sync support.

Thread-safe — all public operations acquire ``_FAISS_LOCK``.
"""

import logging
import os
import pickle
import threading

import faiss
import numpy as np

logger = logging.getLogger(__name__)

DATABASE_DIR = "database"
INDEX_FILE = os.path.join(DATABASE_DIR, "faiss.index")
METADATA_FILE = os.path.join(DATABASE_DIR, "metadata.pkl")

os.makedirs(DATABASE_DIR, exist_ok=True)

DIMENSION = 384

index = None
metadata = []
_FAISS_LOCK = threading.Lock()


def _load():
    global index, metadata
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


def _save():
    global index, metadata
    logger.debug("[FAISS] Saving index (%d vectors) and %d metadata entries",
                 index.ntotal, len(metadata))
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)


def add_documents(chunks, embeddings):
    global index, metadata
    with _FAISS_LOCK:
        if index is None:
            _load()

        n_before = index.ntotal
        embeddings = np.asarray(embeddings, dtype=np.float32)
        index.add(embeddings)
        metadata.extend(chunks)
        _save()
        logger.info("[FAISS] add_documents: %d → %d vectors (+%d)",
                    n_before, index.ntotal, len(chunks))


def remove_pdf(pdf_name):
    global index, metadata
    with _FAISS_LOCK:
        if index is None:
            _load()

        before = len(metadata)
        metadata = [m for m in metadata if m["pdf_name"] != pdf_name]
        removed = before - len(metadata)
        if removed:
            logger.info("[FAISS] remove_pdf: %s — removing %d entries",
                        pdf_name, removed)
            _rebuild_from_metadata()
        else:
            logger.debug("[FAISS] remove_pdf: %s not found", pdf_name)


def _rebuild_from_metadata():
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


def rebuild_index():
    with _FAISS_LOCK:
        if index is None:
            _load()
        _rebuild_from_metadata()


def search(query_embedding, top_k=3):
    with _FAISS_LOCK:
        if index is None:
            _load()

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
            _load()
        return index.ntotal


def get_pdf_chunk_count(pdf_name):
    with _FAISS_LOCK:
        if index is None:
            _load()
        return sum(1 for m in metadata if m["pdf_name"] == pdf_name)


def get_indexed_pdfs():
    with _FAISS_LOCK:
        if index is None:
            _load()
        return list(set(m["pdf_name"] for m in metadata))


def sync_with_files(existing_files):
    """Remove FAISS entries for PDFs no longer in existing_files list."""
    with _FAISS_LOCK:
        if index is None:
            _load()

        existing_set = set(existing_files)
        indexed = get_indexed_pdfs()
        orphaned = [p for p in indexed if p not in existing_set]

        for pdf_name in orphaned:
            logger.info("[FAISS] Removing orphaned PDF: %s", pdf_name)
            remove_pdf(pdf_name)

        return orphaned


_load()
