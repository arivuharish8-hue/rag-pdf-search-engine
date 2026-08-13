"""RAG index health report -- READ-ONLY diagnostic.

Prints the state of the FAISS index, metadata.pkl, the local chunk cache,
Supabase Storage and the processing_jobs table, then states the most likely
failure point.

This script NEVER modifies anything:
- it does not delete files, rebuild FAISS, or touch metadata.pkl
- it does not update Supabase, RabbitMQ, Celery or processing_jobs

Usage:
    python check_index_health.py
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "database")
META_PATH = os.path.join(DB_DIR, "metadata.pkl")
INDEX_PATH = os.path.join(DB_DIR, "faiss.index")
CHUNKS_DIR = os.path.join(DB_DIR, "chunks")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

BAR = "=" * 72
SUB = "-" * 72


def load_metadata():
    if not os.path.exists(META_PATH):
        return None
    try:
        with open(META_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"   ERROR loading metadata.pkl: {exc}")
        return None


def load_index():
    if not os.path.exists(INDEX_PATH):
        return None
    try:
        import faiss
        return faiss.read_index(INDEX_PATH)
    except Exception as exc:
        print(f"   ERROR loading faiss.index: {exc}")
        return None


def get_storage_files():
    """Return (files, error).  Uses the storage client directly so a listing
    failure is reported instead of being silently swallowed."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, ".env"))
        from utils.supabase_storage import get_client, get_bucket
        files = get_client().storage.from_(get_bucket()).list()
        return (files or []), None
    except Exception as exc:
        return [], f"storage listing failed: {exc}"


def get_jobs():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, ".env"))
        from utils.database import get_all_jobs
        return get_all_jobs(), None
    except Exception as exc:
        return [], f"processing_jobs query failed: {exc}"


def main():
    print(BAR)
    print("RAG INDEX HEALTH REPORT")
    print(BAR)

    metadata = load_metadata()
    index = load_index()

    # -- 1. FAISS index exists -----------------------------------------
    print("\n1. FAISS index exists:")
    if index is not None:
        print("   YES")
    else:
        print("   NO" + ("  (file missing)" if not os.path.exists(INDEX_PATH) else ""))

    # -- 2. FAISS vector count -----------------------------------------
    print("\n2. FAISS vector count:")
    print(f"   {index.ntotal if index is not None else 'N/A'}")

    # -- 3. metadata.pkl exists ----------------------------------------
    print("\n3. metadata.pkl exists:")
    if metadata is not None:
        print("   YES")
    else:
        print("   NO" + ("  (file missing)" if not os.path.exists(META_PATH) else ""))

    # -- 4. Metadata entry count ---------------------------------------
    print("\n4. Metadata entry count:")
    n_meta = len(metadata) if isinstance(metadata, list) else 0
    print(f"   {n_meta}")

    # -- 5. FAISS vs metadata ------------------------------------------
    print("\n5. FAISS vs metadata:")
    if index is not None and isinstance(metadata, list):
        if index.ntotal == n_meta:
            print("   MATCH")
        else:
            print(f"   MISMATCH  (faiss={index.ntotal}, metadata={n_meta})")
    else:
        print("   N/A (one side unavailable)")

    # -- 6. PDFs represented in metadata -------------------------------
    print("\n6. PDFs represented in metadata:")
    from collections import OrderedDict
    per_pdf = OrderedDict()
    if isinstance(metadata, list):
        for m in metadata:
            if not isinstance(m, dict):
                continue
            key = m.get("pdf_name", "<unknown>")
            per_pdf.setdefault(key, []).append(m)
    if not per_pdf:
        print("   (none)")
    for name, entries in per_pdf.items():
        pages = [e.get("page") for e in entries if e.get("page") is not None]
        print(f"   {name}")
        print(f"      chunks: {len(entries)}   pages: first={min(pages) if pages else 'n/a'}"
              f"  last={max(pages) if pages else 'n/a'}")

    # -- 7. Local chunk cache ------------------------------------------
    print("\n7. Local chunk cache (database/chunks):")
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    cache_files = sorted(f for f in os.listdir(CHUNKS_DIR)
                         if f.endswith(".pkl"))
    if not cache_files:
        print("   (no cached chunk/embedding files -- normal for COMPLETED jobs)")
    for fname in cache_files:
        path = os.path.join(CHUNKS_DIR, fname)
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                detail = f"{len(obj)} chunks"
            else:
                detail = f"{len(obj) if hasattr(obj, '__len__') else '?'} entries"
        except Exception as exc:
            detail = f"unreadable: {exc}"
        print(f"   {fname}  ({detail})")

    # -- 8. Processing jobs --------------------------------------------
    print("\n8. Processing jobs (processing_jobs):")
    jobs, jobs_err = get_jobs()
    if jobs_err:
        print(f"   {jobs_err}")
    elif not jobs:
        print("   (no jobs)")
    for j in jobs:
        print(f"   job_id   : {j.get('job_id')}")
        print(f"   pdf_name : {j.get('pdf_name')}")
        print(f"   status   : {j.get('status')}")
        print(f"   stage    : {j.get('current_stage')}")
        print(f"   chunks   : total={j.get('total_chunks')} "
              f"done={j.get('last_processed_chunk')}")
        if j.get("error_message"):
            print(f"   error    : {j.get('error_message')}")
        print(SUB)

    # -- 9. Missing PDFs -----------------------------------------------
    print("\n9. Missing PDFs (present/claimed but absent from FAISS metadata):")
    storage, storage_err = get_storage_files()
    if storage_err:
        print(f"   {storage_err}")
    storage_names = {f.get("name") for f in storage
                     if f.get("name") and f["name"].lower().endswith(".pdf")}
    jobs_by_name = {}
    for j in jobs:
        jobs_by_name.setdefault(j.get("pdf_name"), []).append(j)

    issues = []
    for j in jobs:
        name = j.get("pdf_name")
        in_meta = name in per_pdf
        if not in_meta:
            if j.get("total_chunks", 0) > 0 and j.get("status") == "COMPLETED":
                issues.append(f"STALE: job COMPLETED with {j['total_chunks']} chunk(s) "
                              f"for '{name}' but FAISS has no vectors "
                              f"(PDF no longer in storage: {name in storage_names})")
            elif j.get("status") in ("COMPLETED",) and j.get("total_chunks", 0) == 0:
                issues.append(f"EMPTY: '{name}' COMPLETED with 0 chunks "
                              f"(no extractable text -- image-based/empty PDF)")
            else:
                issues.append(f"STALLED: '{name}' is {j.get('status')} "
                              f"(stage={j.get('current_stage')}) but has no vectors")
    for name in sorted(storage_names):
        if name not in per_pdf and not jobs_by_name.get(name):
            issues.append(f"ORPHANED-UPLOAD: '{name}' is in storage but has no job row "
                          f"and no vectors (upload never enqueued)")
    for name in sorted(per_pdf):
        if name not in storage_names and not any(j.get("pdf_name") == name
                                                 for j in jobs):
            issues.append(f"INDEX-ONLY: '{name}' has vectors but is not in storage")

    local_names = set()
    if os.path.isdir(UPLOAD_DIR):
        local_names = {f for f in os.listdir(UPLOAD_DIR)
                       if f.lower().endswith(".pdf")}
    for name in sorted(local_names - storage_names):
        if name not in per_pdf and name not in jobs_by_name:
            issues.append(f"LOCAL-ONLY: '{name}' exists in uploads/ but is NOT in "
                          f"storage and has no job -- failed upload artifact")

    if not issues:
        print("   (none -- every job/PDF is consistent)")
    for msg in issues:
        print(f"   - {msg}")

    # -- 10. Final diagnosis -------------------------------------------
    print("\n10. Final diagnosis:")
    diagnosed = False
    if index is not None and isinstance(metadata, list) and index.ntotal == n_meta:
        print("   - FAISS index and metadata.pkl are CONSISTENT "
              f"(both {n_meta}).")
    else:
        print("   - FAISS and metadata counts DISAGREE -- an index write was "
              "partial or interrupted.")
        diagnosed = True
    if any("STALLED" in i for i in issues):
        print("   - Jobs are stuck (UPLOADED/PROCESSING) -- the Celery worker is "
              "not consuming or a stage keeps failing.  Check worker logs and "
              "RabbitMQ.")
        diagnosed = True
    if any("STALE" in i for i in issues):
        print("   - COMPLETED jobs claim chunks that have no FAISS vectors -- "
              "the PDFs were removed from storage outside the app's delete flow "
              "or the index was reconciled without clearing these rows.")
        diagnosed = True
    if any("EMPTY" in i for i in issues):
        print("   - PDFs completed with 0 chunks -- no extractable text "
              "(image-based/scanned PDF).  Text indexing needs an OCR step; "
              "this is expected behaviour for such PDFs.")
        diagnosed = True
    if any("LOCAL-ONLY" in i for i in issues) or any("ORPHANED-UPLOAD" in i for i in issues):
        print("   - Some PDFs never reached the pipeline: upload to Supabase "
              "Storage failed silently or the object/job was removed.  Their "
              "files remain in uploads/ as artifacts.")
        diagnosed = True
    if not diagnosed:
        print("   - No failure found: the index is consistent and every "
              "non-empty COMPLETED job has vectors.")

    print(BAR)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
