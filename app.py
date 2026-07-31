"""Flask application — PDF RAG Search Engine with fault-tolerant batch processing."""

import logging
import os

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from utils.supabase_storage import list_files, delete_file
from utils.embeddings import create_query_embedding
from utils.faiss_db import (
    get_pdf_chunk_count,
    remove_pdf,
    search,
    sync_with_files,
    total_vectors,
)
from utils.gemini import GeminiGenerationError, generate_answer
from utils.database import get_job, get_all_jobs, delete_jobs_for_pdf
from utils.processing import start_processing, resume_pending_jobs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Flask(__name__)

load_dotenv()

app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "change-this")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("database", exist_ok=True)


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"


def _get_status_for_pdf(pdf_name):
    all_jobs = get_all_jobs()
    pdf_jobs = [j for j in all_jobs if j["pdf_name"] == pdf_name]

    if pdf_jobs:
        latest = max(pdf_jobs, key=lambda j: j["created_at"])
        s = latest["status"]
        if s == "COMPLETED":
            return "indexed"
        elif s == "FAILED":
            return "failed"
        return "processing"

    if get_pdf_chunk_count(pdf_name) > 0:
        return "indexed"
    return "processing"


def get_uploaded_pdfs():
    files = list_files()
    pdfs = []
    for f in files:
        name = f.get("name", "")
        if not name.lower().endswith(".pdf"):
            continue
        pdfs.append({
            "name": name,
            "created_at": f.get("created_at", ""),
            "status": _get_status_for_pdf(name),
            "chunks": get_pdf_chunk_count(name),
        })
    pdfs.sort(key=lambda p: p["created_at"], reverse=True)
    return pdfs


class DeleteNotFound(Exception):
    """Raised when the PDF exists nowhere that we could delete it from."""


class DeleteFailed(Exception):
    """Raised when a critical delete step fails before any partial change."""


def delete_pdf(pdf_name):
    """Remove a PDF end-to-end: Supabase Storage, processing_jobs, FAISS, disk.

    Ordered for atomicity / rollback:
      1. Supabase Storage (source of truth) — on failure nothing has changed
         yet, so the delete aborts cleanly with a meaningful error.
      2. processing_jobs rows — removed BEFORE the index so an in-flight
         Celery stage finds no job and stops instead of re-adding vectors.
      3. FAISS vectors + metadata (persists faiss.index + metadata.pkl).
      4. Local download-cache copy.

    Returns a summary dict.  Failures in the non-critical steps 2-4 are
    logged with the exact error and collected in summary["errors"] so the
    caller can still report a partial deletion with the right HTTP status.
    """
    storage_names = {f.get("name", "") for f in list_files()}
    storage_exists = pdf_name in storage_names
    vectors_before = get_pdf_chunk_count(pdf_name)
    job_ids = [j["job_id"] for j in get_all_jobs()
               if j["pdf_name"] == pdf_name]

    if not storage_exists and vectors_before == 0 and not job_ids:
        raise DeleteNotFound(f"PDF '{pdf_name}' not found or already deleted")

    errors = []

    if storage_exists:
        try:
            delete_file(pdf_name)
        except Exception as exc:
            logger.error("[Delete] Supabase deletion failed for %s: %s",
                         pdf_name, exc, exc_info=True)
            raise DeleteFailed(
                f"Supabase deletion failed for '{pdf_name}': {exc}"
            ) from exc
    else:
        logger.info("[Delete] %s already absent from Supabase — continuing "
                    "cleanup", pdf_name)

    try:
        delete_jobs_for_pdf(pdf_name)
    except Exception as exc:
        logger.error("[Delete] processing_jobs deletion failed for %s: %s",
                     pdf_name, exc, exc_info=True)
        errors.append(f"processing_jobs: {exc}")

    try:
        remove_pdf(pdf_name)
    except Exception as exc:
        logger.error("[Delete] FAISS removal failed for %s: %s",
                     pdf_name, exc, exc_info=True)
        errors.append(f"faiss: {exc}")

    local_path = os.path.join(UPLOAD_FOLDER, pdf_name)
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
    except OSError as exc:
        logger.error("[Delete] local file removal failed for %s: %s",
                     pdf_name, exc, exc_info=True)
        errors.append(f"local_file: {exc}")

    return {
        "pdf_name": pdf_name,
        "deleted": True,
        "storage_deleted": storage_exists,
        "jobs_deleted": len(job_ids),
        "vectors_removed": vectors_before - get_pdf_chunk_count(pdf_name),
        "total_vectors": total_vectors(),
        "errors": errors,
    }


def sync_storage_and_index():
    files = list_files()
    supabase_names = [f["name"] for f in files if f.get("name", "").endswith(".pdf")]
    return sync_with_files(supabase_names)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/pdfs")
def list_pdfs_json():
    return jsonify(get_uploaded_pdfs())


@app.route("/queue_status/<job_id>")
def queue_status(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify({"status": "unknown"})
    return jsonify({
        "status": "done"
        if job["status"] == "COMPLETED"
        else "error"
        if job["status"] == "FAILED"
        else "processing",
        "pdf_name": job["pdf_name"],
        "chunks": job["total_chunks"],
        "processed": job["last_processed_chunk"],
        "stage": job["current_stage"],
    })


@app.route("/uploads/<filename>")
def serve_pdf(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/delete/<pdf_name>", methods=["DELETE", "POST"])
def delete_pdf_route(pdf_name):
    """Delete a PDF end-to-end and return JSON (no redirect, no swallowing).

    Runs synchronously and only reports success once every step finished.
    HTTP status is meaningful:
      200  fully deleted (body: status=ok, vectors_removed, total_vectors, ...)
      404  PDF does not exist anywhere (idempotent re-delete)
      500  deletion failed / partial (details in body["error"])
    """
    try:
        summary = delete_pdf(pdf_name)
    except DeleteNotFound as exc:
        logger.info("[Delete] %s", exc)
        return jsonify({"status": "not_found", "deleted": False,
                        "error": str(exc)}), 404
    except DeleteFailed as exc:
        logger.error("[Delete] %s", exc)
        return jsonify({"status": "error", "deleted": False,
                        "error": str(exc)}), 500
    except Exception as exc:
        logger.error("[Delete] Unexpected error deleting %s: %s",
                     pdf_name, exc, exc_info=True)
        return jsonify({"status": "error", "deleted": False,
                        "error": str(exc)}), 500

    if summary["errors"]:
        logger.error("[Delete] Partial deletion of %s: %s",
                     pdf_name, "; ".join(summary["errors"]))
        return jsonify({"status": "partial", **summary,
                        "error": "; ".join(summary["errors"])}), 500

    logger.info("[Delete] Deleted %s (jobs=%d, vectors=%d, total_vectors=%d)",
                pdf_name, summary["jobs_deleted"], summary["vectors_removed"],
                summary["total_vectors"])
    return jsonify({"status": "ok", **summary}), 200


@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    answer = None
    query = ""

    if request.method == "POST":
        if "pdf" in request.files:
            pdf = request.files["pdf"]
            if pdf.filename and is_allowed_file(pdf.filename):
                try:
                    start_processing(pdf, pdf.filename)
                except Exception as e:
                    logger.error("[Upload] Error: %s", e)
                return redirect(url_for("home"))

        if "query" in request.form:
            query = request.form["query"].strip()
            if query:
                try:
                    query_embedding = create_query_embedding(query)
                    results = search(query_embedding, top_k=1)
                    if results:
                        contexts = [r["text"] for r in results]
                        try:
                            answer = generate_answer(query, contexts)
                        except GeminiGenerationError:
                            answer = contexts[0][:300]
                except Exception as e:
                    logger.error("[Search] Error: %s", e)

    return render_template(
        "index.html",
        results=results,
        answer=answer,
        query=query,
        uploaded_pdfs=get_uploaded_pdfs(),
        total_vectors=total_vectors(),
    )


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


@app.route("/health")
def health():
    """Liveness + dependency health check.

    Returns 200 when the app, database backend and RabbitMQ broker are all
    reachable; 503 otherwise.  ``celery_workers`` reports how many workers
    answered the ping (informational — 0 is allowed, e.g. during deploy).
    """
    checks = {
        "status": "ok",
        "app": "ok",
        "database": "ok",
        "broker": "ok",
        "celery_workers": 0,
    }
    status_code = 200

    try:
        get_all_jobs()
    except Exception as exc:
        checks["database"] = f"error: {exc}"
        checks["status"] = "error"
        status_code = 503

    try:
        from celery_app import celery_app

        with celery_app.connection() as conn:
            conn.ensure_connection(max_retries=1, timeout=2)
        try:
            workers = celery_app.control.ping(timeout=2)
            checks["celery_workers"] = len(workers)
        except Exception as exc:
            logger.warning("[Health] worker ping failed: %s", exc)
    except Exception as exc:
        checks["broker"] = f"error: {exc}"
        checks["status"] = "error"
        status_code = 503

    return jsonify(checks), status_code


# ---------------------------------------------------------------------------
# Debug / verification endpoints
# ---------------------------------------------------------------------------


@app.route("/debug")
def debug_index():
    return render_template("debug.html")


@app.route("/debug/metadata")
def debug_metadata():
    """Inspect metadata.pkl and compare with the FAISS index."""
    import pickle
    import os
    import traceback

    from utils.faiss_db import INDEX_FILE, METADATA_FILE, DIMENSION
    import faiss
    import numpy as np

    lines = []
    ok = True

    def out(msg):
        lines.append(msg)

    meta_path = METADATA_FILE
    idx_path = INDEX_FILE

    out("=" * 56)
    out("metadata.pkl Integrity Report")
    out("=" * 56)

    # ── 1. Check file exists ───────────────────────────────────────────
    out("")
    out("1. metadata.pkl exists")
    meta_exists = os.path.exists(meta_path)
    out(f"   YES" if meta_exists else "   NO")
    if not meta_exists:
        out("   (file not found at database/metadata.pkl)")
        ok = False
    else:
        out(f"   Size: {os.path.getsize(meta_path)} bytes")

    # ── 2. Load via pickle ─────────────────────────────────────────────
    out("")
    out("2. Load metadata.pkl")
    metadata = None
    if meta_exists:
        try:
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            out(f"   Loaded: {len(metadata)} entries")
        except Exception as e:
            out(f"   FAILED: {e}")
            out(traceback.format_exc())
            ok = False

    # ── 3. Type checks ─────────────────────────────────────────────────
    out("")
    out("3. Type information")
    if metadata is not None:
        out(f"   metadata object type : {type(metadata).__name__}")
        if isinstance(metadata, list) and metadata:
            out(f"   first entry type    : {type(metadata[0]).__name__}")
            if isinstance(metadata[0], dict):
                out(f"   first entry keys    : {list(metadata[0].keys())}")

    # ── 4. First / last entry ──────────────────────────────────────────
    out("")
    out("4. Sample entries")
    if isinstance(metadata, list):
        out(f"   Total entries: {len(metadata)}")
        if metadata:
            first = metadata[0]
            if isinstance(first, dict):
                out("   First entry:")
                for k, v in first.items():
                    out(f"      {k}: {str(v)[:160]}")
            else:
                out(f"   First entry (not a dict): {first}")
            last = metadata[-1]
            if isinstance(last, dict):
                out("   Last entry:")
                for k, v in last.items():
                    out(f"      {k}: {str(v)[:160]}")
            else:
                out(f"   Last entry (not a dict): {last}")

    # ── 5. Load FAISS and compare counts ──────────────────────────────
    out("")
    out("5. FAISS index comparison")
    n_meta = len(metadata) if isinstance(metadata, list) else 0
    try:
        index = faiss.read_index(idx_path)
        n_vec = index.ntotal
        out(f"   FAISS vectors      : {n_vec}")
        out(f"   Metadata entries   : {n_meta}")
        if n_meta == n_vec:
            out("   COUNT MATCH        : PASS")
        else:
            out("   COUNT MATCH        : FAIL")
            ok = False
    except Exception as e:
        out(f"   FAISS load failed  : {e}")
        ok = False

    # ── 6. Integrity checks on every entry ─────────────────────────────
    out("")
    out("6. Entry integrity")
    if isinstance(metadata, list):
        required = {"pdf_name", "page", "chunk", "text"}
        bad_type = 0
        missing = 0
        empty_text = 0
        for entry in metadata:
            if not isinstance(entry, dict):
                bad_type += 1
                continue
            miss = required - set(entry.keys())
            if miss:
                missing += 1
            if not entry.get("text"):
                empty_text += 1
        out(f"   Non-dict entries  : {bad_type}" + (" (OK)" if bad_type == 0 else " (FAIL)"))
        out(f"   Missing fields    : {missing}" + (" (OK)" if missing == 0 else " (FAIL)"))
        out(f"   Empty text fields : {empty_text}" + (" (OK)" if empty_text == 0 else " (FAIL)"))
        if bad_type or missing or empty_text:
            ok = False

  
    out("")
    out("=" * 56)
    if ok:
        out(" metadata.pkl verification PASSED")
        out(" All checks OK")
    else:
        out(" metadata.pkl verification FAILED")
        out(" See issues above")
    out("=" * 56)

    return "<pre>" + "\n".join(lines) + "</pre>", 200 if ok else 500



def start_reconciliation():
    """Re-enqueue any jobs left unfinished by a previous run (crash recovery).

    Runs once at startup.  Idempotent — the atomic job claim in the Celery
    orchestrator prevents duplicates.  If RabbitMQ is down, jobs stay
    non-terminal in the DB and are picked up on the next restart.
    """
    logger.info("[Startup] Starting queue reconciliation ...")
    try:
        n = resume_pending_jobs()
        logger.info("[Startup] Reconciliation done: %d job(s) re-enqueued", n)
    except Exception as exc:
        logger.error("[Startup] Reconciliation failed: %s", exc, exc_info=True)
    logger.info("[Startup] Reconciliation finished")


_is_reloader_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
_is_reloader_parent = not _is_reloader_child and os.environ.get("WERKZEUG_SERVER_FD")

if _is_reloader_child or not _is_reloader_parent:
    start_reconciliation()

if __name__ == "__main__":
    app.run(debug=True)
