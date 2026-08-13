"""Flask application — PDF RAG Search Engine with fault-tolerant batch processing."""

import logging
import os
import re
import threading
import time

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

from utils.supabase_storage import list_files, delete_file
from utils.embeddings import create_query_embedding
from utils.citations import (
    build_citation_map,
    extract_cited_ids,
    render_answer_links,
    validate_answer,
)
from utils.faiss_db import (
    get_pdf_chunk_count,
    hybrid_search,
    persisted_chunk_count,
    remove_pdf,
    sync_with_files,
    total_vectors,
)
from utils.gemini import GeminiGenerationError, generate_answer
from utils.query_normalizer import normalize_query
from utils.reranker import rerank
from utils.bm25 import tokenize as _bm25_tokenize
from utils.database import get_job, get_all_jobs, delete_jobs_for_pdf
from utils.processing import (
    start_processing,
    resume_pending_jobs,
    resume_stuck_uploads,
)

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

# Second-stage reranker: hybrid_search retrieves a larger candidate pool,
# then a local cross-encoder re-ranks it before Gemini.  The model is
# pre-cached in the image (see Dockerfile) and loaded lazily on first search.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

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


# Internal storage keys look like "<32-hex-uuid>_<original-filename>.pdf".
# The full value is always used for storage/lookup; only the displayed form is
# cleaned up.
_UUID_HASH_PREFIX = re.compile(r"^[0-9a-f]{32}_")


def display_pdf_name(pdf_name):
    """Human-readable filename for display, keeping the stored key untouched."""
    return _UUID_HASH_PREFIX.sub("", pdf_name) or pdf_name


def pdf_url_for(result):
    """Browser-accessible PDF URL for a retrieved source (existing /uploads route).

    Uses the actual pdf_name from the retrieved chunk metadata; never hardcoded.
    """
    return url_for("serve_pdf", filename=result["pdf_name"])


# User-friendly explanations keyed by GeminiGenerationError.kind.  These never
# contain the API key or raw provider internals; the full error is logged.
GEMINI_FALLBACK_REASONS = {
    "rate_limit": "Gemini is temporarily rate-limited. Please try again shortly.",
    "quota": (
        "Gemini API quota has been exhausted. Please check the Gemini API "
        "quota or use an available model/API key."
    ),
    "auth": "Gemini API authentication failed. Please check the API key configuration.",
}


def _fallback_answer(results, exc):
    """Build a clearly-labeled, non-AI fallback when Gemini cannot answer.

    Intentionally does NOT dump raw retrieved text into the user-facing answer;
    that would expose unrelated candidate chunks verbatim.  Instead we emit a
    clean status message and, if the strongest reranked result has a valid
    citation_id, append a single inline citation so the Sources section can
    render the correct page link without exposing raw content.
    """
    reason = GEMINI_FALLBACK_REASONS.get(
        exc.kind, "The Gemini API request failed. Please check the server logs."
    )
    message = "Gemini answer unavailable. " + reason

    # Attach a citation to the strongest reranked result only, so the Sources
    # section shows the single most-relevant page.  No raw text is exposed.
    if results:
        cid = results[0].get("citation_id")
        if cid is not None:
            message += f" [{ cid }]"

    return message


# Query tokens that carry no topical content.  A candidate whose only lexical
# overlap with the query is one of these is NOT corroborated.
_FUNCTION_WORDS = frozenset(
    "a an and are as at be but by for if in is it of on or so the to was "
    "we what when where which who whom why how do does did have has had "
    "this that these those i you he she they them your my our their his her "
    "me us not no nor too very can could should would may might must will "
    "shall just also into over under about from with without".split()
)


def _rescue_topically_corroborated(query, candidates, results, top_k):
    """Restore a genuinely relevant chunk that the reranker's strict relevance
    gate discarded, then order it ahead of non-corroborated results.

    The cross-encoder's relevance gate (``rerank_score > 0``) can reject the
    exact section page: ms-marco-MiniLM-L-6-v2 under-scores verbatim section
    text, e.g. query "haridass achievements" vs the Achievements chunk scores
    ~ -8 while an unrelated resume-header page scores ~ +0.8, so the correct
    page is dropped from the Gemini context and the wrong page is cited.
    A candidate whose text literally contains a non-stopword query token is
    verified against the live corpus BM25 index (true corpus-IDF weighting)
    and must never be dropped: it is restored and ordered first.

    Only the "final Gemini context" (the list handed to Gemini, and therefore
    the citation order / fallback source pick) is affected.  Retrieval,
    hybrid scoring, the reranker, Gemini, citation rendering and the fallback
    message logic are all left untouched.
    """
    if not candidates:
        return results

    topical = " ".join(
        t for t in _bm25_tokenize(query)
        if len(t) >= 3 and t not in _FUNCTION_WORDS
    )
    if not topical:
        return results

    try:
        from utils.bm25 import search as _bm25_search
        from utils import faiss_db
        hits = _bm25_search(
            topical,
            faiss_db.metadata,
            faiss_db._METADATA_VERSION,
            top_k=max(len(candidates), 1),
        )
    except Exception:
        logger.error(
            "[Search] Topical-corroboration rescue failed", exc_info=True
        )
        return results

    corroborated = {
        (faiss_db.metadata[idx]["pdf_name"],
         faiss_db.metadata[idx]["page"],
         faiss_db.metadata[idx]["chunk"]): round(score, 4)
        for score, idx in hits
    }

    result_keys = {
        (r["pdf_name"], r["page"], r["chunk"]) for r in results
    }
    rescued = []
    for candidate in candidates:
        key = (candidate["pdf_name"], candidate["page"], candidate["chunk"])
        if key not in result_keys and key in corroborated:
            item = candidate.copy()
            item["rerank_score"] = None
            item["topical_bm25_score"] = corroborated[key]
            rescued.append(item)

    merged = list(results) + rescued
    for r in merged:
        r["topical_bm25_score"] = corroborated.get(
            (r["pdf_name"], r["page"], r["chunk"]), 0.0
        )

    corroborated_results = [
        r for r in merged if r["topical_bm25_score"] > 0.0
    ]
    others = [r for r in merged if r["topical_bm25_score"] <= 0.0]

    corroborated_results.sort(
        key=lambda r: (
            r["topical_bm25_score"],
            r["rerank_score"]
            if r["rerank_score"] is not None else float("-inf"),
            r.get("hybrid_score", 0.0),
        ),
        reverse=True,
    )
    others.sort(
        key=lambda r: (
            r["rerank_score"]
            if r["rerank_score"] is not None else float("-inf"),
            r.get("hybrid_score", 0.0),
        ),
        reverse=True,
    )
    return (corroborated_results + others)[:top_k]


def _latest_job_for(all_jobs, pdf_name):
    jobs = [j for j in all_jobs if j.get("pdf_name") == pdf_name]
    return max(jobs, key=lambda j: j.get("created_at", "")) if jobs else None


def _status_from_job(job, pdf_name):
    if job is None:
        return "indexed" if get_pdf_chunk_count(pdf_name) > 0 else "processing"
    s = job["status"]
    if s == "COMPLETED":
        return "indexed"
    elif s == "FAILED":
        return "failed"
    return "processing"


def _get_status_for_pdf(pdf_name):
    all_jobs = get_all_jobs()
    return _status_from_job(_latest_job_for(all_jobs, pdf_name), pdf_name)


def get_uploaded_pdfs():
    files = list_files()
    all_jobs = get_all_jobs()

    pdfs = []
    for f in files:
        name = f.get("name", "")
        if not name.lower().endswith(".pdf"):
            continue
        job = _latest_job_for(all_jobs, name)

        # Authoritative chunk count: a COMPLETED job's total_chunks is what
        # the pipeline actually indexed.  Fall back to the FAISS index (live
        # progress, or indexed PDFs without a job row) when not completed.
        chunks = get_pdf_chunk_count(name)
        if job is not None and job["status"] == "COMPLETED" and job.get("total_chunks"):
            chunks = job["total_chunks"]

        # A COMPLETED job's error_message is an informational note (e.g. the
        # PDF had no extractable text) rather than a processing failure.
        message = ""
        if job is not None and job["status"] == "COMPLETED":
            message = job.get("error_message") or ""

        pdfs.append({
            "name": name,
            "created_at": f.get("created_at", ""),
            "status": _status_from_job(job, name),
            "chunks": chunks,
            "message": message,
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
         Celery stage finds no job and stops instead of re-adding vectors
         (tasks.py also re-checks job aliveness inside its indexing loop).
      3. FAISS vectors + metadata — the index is reconciled against the
         remaining Supabase files (persists faiss.index + metadata.pkl).
         The reconciliation removes the deleted PDF's vectors *and* any other
         orphaned vectors (e.g. a re-upload that changed the object name, or
         a stale entry from an earlier partial delete).  When the last PDF is
         deleted this leaves an empty index + empty metadata.
      4. Local download-cache copy.

    The FAISS reconciliation only runs when the storage listing was proven
    usable (the file existed and was deleted from Supabase).  ``list_files()``
    returns ``[]`` on a Supabase error, so without that guard a transient
    outage could wrongly empty the whole index.

    Returns a summary dict.  Failures in the non-critical steps 2-4 are
    logged with the exact error and collected in summary["errors"] so the
    caller can still report a partial deletion with the right HTTP status.
    """
    storage_names = {f.get("name", "") for f in list_files()}
    storage_exists = pdf_name in storage_names
    vectors_before = total_vectors()
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
        if storage_exists:
            # The storage listing we captured was proven correct (the file
            # existed and was deleted), so it is safe to use it as the
            # source of truth for what must remain in the index.
            remaining_pdfs = sorted(
                name for name in storage_names
                if name != pdf_name and name.lower().endswith(".pdf")
            )
            orphaned = sync_with_files(remaining_pdfs)
            if orphaned:
                logger.info("[Delete] Reconcile removed %d orphaned PDF(s): %s",
                            len(orphaned), orphaned)
        else:
            # Storage state is not positively confirmed (the listing may have
            # failed); remove only by exact name so a bogus empty listing can
            # never wipe unrelated vectors.
            remove_pdf(pdf_name)

        # Prove the removal actually persisted to disk.  A failed os.replace
        # (Windows file-lock contention) used to leave the PDF's chunks in
        # metadata.pkl while the route still reported success — this check
        # makes that a loud failure instead of a silent one.
        remaining_chunks = persisted_chunk_count(pdf_name)
        if remaining_chunks > 0:
            raise RuntimeError(
                f"'{pdf_name}' still has {remaining_chunks} chunks persisted "
                f"after FAISS removal"
            )
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

    vectors_after = total_vectors()
    return {
        "pdf_name": pdf_name,
        "deleted": True,
        "storage_deleted": storage_exists,
        "jobs_deleted": len(job_ids),
        "vectors_removed": vectors_before - vectors_after,
        "total_vectors": vectors_after,
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

#post# 

@app.route("/search", methods=["POST"])
def search_route():
    """PRG entry point for search queries.

    The search form posts here instead of to "/".  The query is stored in the
    session (a tiny cookie) and the client is redirected with 303 to the home
    page, which consumes the session on the next GET.  This keeps the browser
    URL at "/" (no ``?q=...``) and, because the session value is consumed on
    the first render, a browser refresh returns to the normal page state
    without any "Confirm Form Resubmission" prompt.
    """
    query = request.form.get("query", "").strip()
    if query:
        session["last_query"] = query
        logger.info("[Search] PRG: stored query=%r", query)
    return redirect(url_for("home"), 303)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "pdf" in request.files:
            pdf = request.files["pdf"]
            if pdf.filename and is_allowed_file(pdf.filename):
                try:
                    start_processing(pdf, pdf.filename)
                except Exception as e:
                    logger.error("[Upload] Error: %s", e, exc_info=True)
                    return redirect(url_for(
                        "home",
                        error="Upload failed: the PDF could not be stored or "
                              "queued. Please try again.",
                    ))
                return redirect(url_for("home"))
        return redirect(url_for("home"))

    # PRG search flow: /search POST stores the query in the session, this GET
    # consumes it exactly once.  Falling back to ?q= keeps direct URLs /
    # bookmarks working, but the session path is preferred so the browser URL
    # stays clean and a refresh returns to the normal (query-less) page state.
    query = (session.pop("last_query", "").strip()
             or request.args.get("q", "").strip())
    raw_query = query
    results = []
    answer = None
    answer_html = None
    cited_sources = []

    if query:
        try:
            if total_vectors() == 0:
                answer = "No documents indexed."
            else:
                # Keep the ORIGINAL query for display and for Gemini.  Only
                # the minimally corrected form (if an actual typo was detected)
                # is used for retrieval / reranking.
                retrieval_query = normalize_query(query)
                logger.info(
                    "[Search] Retrieval flow: %r -> %r -> hybrid_search()",
                    raw_query, retrieval_query,
                )
                query_embedding = create_query_embedding(retrieval_query)
                candidates = hybrid_search(
                    retrieval_query, query_embedding, top_k=RERANK_CANDIDATES
                )
                logger.info("[Search] Pre-rerank candidates: %d", len(candidates))
                results = rerank(retrieval_query, candidates, top_k=RERANK_TOP_K)
                # Rescue any genuinely relevant chunk the reranker's strict
                # relevance gate dropped (see _rescue_topically_corroborated).
                results = _rescue_topically_corroborated(
                    retrieval_query, candidates, results, RERANK_TOP_K
                )
                logger.info(
                    "[Search] Post-rerank final: %d — top rerank scores: %s",
                    len(results), [r.get("rerank_score") for r in results],
                )
                if results:
                    # Stable citations: [1], [2], ... over exactly the final
                    # reranked chunks that are handed to Gemini.
                    results, citations = build_citation_map(results)
                    for result in results:
                        result["pdf_url"] = pdf_url_for(result)
                        result["display_pdf_name"] = display_pdf_name(
                            result["pdf_name"]
                        )
                    try:
                        answer = generate_answer(query, results)
                    except GeminiGenerationError as exc:
                        logger.error(
                            "[Search] Gemini generation failed (kind=%s): %s",
                            exc.kind,
                            exc,
                            exc_info=True,
                        )
                        answer = _fallback_answer(results, exc)
                    if answer:
                        answer = validate_answer(answer, citations)
                        answer_html = render_answer_links(
                            answer, citations, pdf_url_for
                        )
                        # The Sources section is built ONLY from the citations
                        # actually present in the validated answer, not from
                        # the full reranked candidate list.  A retrieved page
                        # that Gemini did not cite must not appear as a source.
                        cited_ids = extract_cited_ids(answer)
                        deduped = {}
                        for cit in citations:
                            if cit["citation_id"] in cited_ids:
                                key = (cit["pdf_name"], cit["page"])
                                if key not in deduped:
                                    deduped[key] = {
                                        "citation_ids": [str(cit["citation_id"])],
                                        "pdf_name": cit["pdf_name"],
                                        "display_pdf_name": display_pdf_name(cit["pdf_name"]),
                                        "page": cit["page"],
                                        "chunks": [str(cit["chunk"])],
                                        "pdf_url": pdf_url_for(cit),
                                    }
                                else:
                                    deduped[key]["citation_ids"].append(str(cit["citation_id"]))
                                    if str(cit["chunk"]) not in deduped[key]["chunks"]:
                                        deduped[key]["chunks"].append(str(cit["chunk"]))
                        cited_sources = list(deduped.values())
                    else:
                        answer_html = answer
                else:
                    answer = "No relevant information was found in the uploaded documents."
                    answer_html = answer
        except Exception as e:
            logger.error("[Search] Error: %s", e)

    return render_template(
        "index.html",
        results=results,
        answer=answer,
        answer_html=answer_html,
        cited_sources=cited_sources,
        query=query,
        error=request.args.get("error", ""),
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


STUCK_UPLOAD_RECONCILE_INTERVAL = 15


def _stuck_upload_reconciler_loop():
    """Periodically re-enqueue jobs stuck in UPLOADED (publish failed).

    Runs in a daemon thread so an upload whose RabbitMQ publish failed (see
    ``utils.processing.enqueue_job``) is automatically retried instead of
    showing "Processing / 0 Chunks" forever.  Only UPLOADED jobs are touched,
    and each is claimed atomically before publishing, so a backed-up queue can
    never cause a duplicate.
    """
    logger.info("[Startup] Stuck-upload reconciler thread started "
                "(interval=%ds)", STUCK_UPLOAD_RECONCILE_INTERVAL)
    while True:
        try:
            n = resume_stuck_uploads()
            if n:
                logger.info("[Startup] Stuck-upload pass re-enqueued %d job(s)", n)
        except Exception as exc:
            logger.error("[Startup] Stuck-upload reconciliation failed: %s",
                         exc, exc_info=True)
        time.sleep(STUCK_UPLOAD_RECONCILE_INTERVAL)


def start_stuck_upload_reconciler():
    threading.Thread(
        target=_stuck_upload_reconciler_loop,
        name="stuck-upload-reconciler",
        daemon=True,
    ).start()


_is_reloader_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
_is_reloader_parent = not _is_reloader_child and os.environ.get("WERKZEUG_SERVER_FD")

if _is_reloader_child or not _is_reloader_parent:
    start_reconciliation()
    start_stuck_upload_reconciler()

if __name__ == "__main__":
    app.run(debug=True)
