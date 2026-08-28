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
from utils.query_normalizer import normalize_query, _COMMON_WORDS
from utils.reranker import rerank
from utils.bm25 import tokenize as _bm25_tokenize
from utils.database import get_job, get_all_jobs, delete_jobs_for_pdf
from utils.processing import (
    start_processing,
    resume_pending_jobs,
    resume_stuck_uploads,
)
import utils.chat_db as chat_db

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
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))

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
    "empty": (
        "Gemini did not return a usable answer for this query. "
        "Please try rephrasing your question."
    ),
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
_FUNCTION_WORDS = _COMMON_WORDS


def _generate_query_variants(query):
    """Generate retrieval variants from a single user query.

    Returns a dict with keys:
      - original: the raw user query (for display and Gemini)
      - normalized: typo-corrected form (for semantic retrieval)
      - keywords: stopwords-stripped content words (for keyword retrieval)
      - rewrites: list of Gemini-rewritten paraphrases (may be empty)

    All variants are derived from the same original query; no external
    knowledge is injected except for optional Gemini rewrites.
    """
    normalized = normalize_query(query)

    # Extract content keywords: drop stopwords and very short tokens
    # Use the NORMALIZED query for keywords so typo corrections are included
    kw_tokens = [
        t for t in _bm25_tokenize(normalized)
        if len(t) >= 2 and t not in _FUNCTION_WORDS
    ]
    keywords = " ".join(kw_tokens) if kw_tokens else normalized

    return {
        "original": query,
        "normalized": normalized,
        "keywords": keywords,
    }


def _gemini_query_rewrites(query, max_rewrites=2):
    """Use Gemini to generate paraphrase rewrites of the query for recall.

    Returns a list of rewritten query strings (may be empty if Gemini is
    unavailable, rate-limited, or returns garbage).  Each rewrite is a
    self-contained rephrasing that preserves the original intent.

    This is an OPTIONAL enhancement — the retrieval pipeline must work
    without it.  Failures are silently swallowed and logged.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return []

    rewrite_prompt = f"""Rewrite the following question into {max_rewrites} different
paraphrases that a person might ask to find the same information. Each
paraphrase should use different wording but preserve the exact same meaning.

Return ONLY the rewritten questions, one per line, with no numbering or
bullets. Do not answer the question.

Question: {query}

Paraphrases:"""

    try:
        from utils.gemini import _get_client
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=rewrite_prompt,
            config={"max_output_tokens": 128},
        )
        text = ""
        try:
            text = response.text
        except Exception:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                parts = getattr(candidates[0].content, "parts", None) or []
                texts = [p.text for p in parts if getattr(p, "text", None)]
                text = "\n".join(texts)

        rewrites = []
        for line in (text or "").strip().split("\n"):
            line = line.strip()
            # Strip numbering like "1. " or "- "
            line = re.sub(r"^[\d\-\*\.\)]+\s*", "", line).strip()
            if line and len(line) > 5 and line.lower() != query.lower():
                rewrites.append(line)
            if len(rewrites) >= max_rewrites:
                break

        if rewrites:
            logger.info(
                "[Search] Gemini rewrites for %r: %s", query[:60], rewrites,
            )
        return rewrites
    except Exception as exc:
        logger.debug("[Search] Gemini rewrite failed: %s", exc)
        return []


def _multi_variant_search(query, top_k, skip_rewrites=False):
    """Run hybrid retrieval with multiple query variants and merge results.

    Uses up to five representations of the same query to maximise recall:
      1. The normalized query — best for FAISS semantic match.
      2. The keywords-only form — best for BM25 exact-keyword match.
      3. The original query (if different from normalized after typos).
      4-5. Gemini-generated paraphrase rewrites (optional, for recall).

    Candidates from all variants are merged by chunk key and deduplicated.
    The merged pool is returned (not yet reranked) so the caller can apply
    its own reranker + rescue logic.

    When ``skip_rewrites`` is True, Gemini paraphrase rewrites are skipped
    to save an API call.  This is safe for simple standalone queries where
    the normalized + keyword variants already cover sufficient recall.
    """
    variants = _generate_query_variants(query)

    # Optional: Gemini paraphrase rewrites for semantic recall
    rewrites = [] if skip_rewrites else _gemini_query_rewrites(query)

    logger.info(
        "[Search] Query variants: original=%r, normalized=%r, keywords=%r, "
        "rewrites=%d",
        variants["original"], variants["normalized"], variants["keywords"],
        len(rewrites),
    )

    all_candidates = {}

    # --- Variant 1: normalized query (semantic + keyword) ---
    emb_norm = create_query_embedding(variants["normalized"])
    cands_norm = hybrid_search(
        variants["normalized"], emb_norm, top_k=top_k
    )
    for c in cands_norm:
        key = (c["pdf_name"], c["page"], c["chunk"])
        if key not in all_candidates:
            all_candidates[key] = c

    # --- Variant 2: keywords-only (BM25-dominant) ---
    # Only if keywords differ from normalized to avoid duplicate work
    if variants["keywords"] != variants["normalized"]:
        emb_kw = create_query_embedding(variants["keywords"])
        cands_kw = hybrid_search(
            variants["keywords"], emb_kw, top_k=top_k
        )
        for c in cands_kw:
            key = (c["pdf_name"], c["page"], c["chunk"])
            if key not in all_candidates:
                all_candidates[key] = c

    # --- Variant 3: original query (may differ from normalized after typos) ---
    if variants["original"] != variants["normalized"]:
        emb_orig = create_query_embedding(variants["original"])
        cands_orig = hybrid_search(
            variants["original"], emb_orig, top_k=top_k
        )
        for c in cands_orig:
            key = (c["pdf_name"], c["page"], c["chunk"])
            if key not in all_candidates:
                all_candidates[key] = c

    # --- Variants 4+: Gemini paraphrase rewrites (optional) ---
    for rewrite in rewrites:
        emb_rw = create_query_embedding(rewrite)
        cands_rw = hybrid_search(rewrite, emb_rw, top_k=top_k)
        for c in cands_rw:
            key = (c["pdf_name"], c["page"], c["chunk"])
            if key not in all_candidates:
                all_candidates[key] = c

    merged = list(all_candidates.values())
    logger.info(
        "[Search] Multi-variant merge: %d candidates from %d variant(s)",
        len(merged), 3 + len(rewrites),
    )
    return merged


# Query tokens that carry no topical content.  A candidate whose only lexical
# overlap with the query is one of these is NOT corroborated.
def _rescue_topically_corroborated(query, candidates, results, top_k,
                                   normalized_query=None):
    """Restore genuinely relevant chunks that the reranker's strict relevance
    gate discarded, then order them ahead of non-corroborated results.

    The cross-encoder's relevance gate (``rerank_score > 0``) can reject the
    exact section page: ms-marco-MiniLM-L-6-v2 under-scores verbatim section
    text, e.g. query "haridass achievements" vs the Achievements chunk scores
    ~ -8 while an unrelated resume-header page scores ~ +0.8, so the correct
    page is dropped from the Gemini context and the wrong page is cited.
    A candidate whose text literally contains a non-stopword query token is
    verified against the live corpus BM25 index (true corpus-IDF weighting)
    and must never be dropped: it is restored and ordered first.

    When ``results`` is the same object as ``candidates`` (pre-rerank call),
    this function acts as a filter that keeps corroborated candidates and
    drops uncorroborated ones, expanding the pool for the reranker.

    ``normalized_query``: if provided, used for topical keyword extraction
    instead of ``query``.  This ensures typo-corrected terms are used for
    BM25 corroboration (e.g. "achivements" → "achievements").

    Only the "final Gemini context" (the list handed to Gemini, and therefore
    the citation order / fallback source pick) is affected.  Retrieval,
    hybrid scoring, the reranker, Gemini, citation rendering and the fallback
    message logic are all left untouched.
    """
    if not candidates:
        return results

    # Build a key->text lookup dict once to avoid O(n) metadata scans
    _meta_lookup = {}
    try:
        for m in faiss_db.metadata:
            k = (m["pdf_name"], m["page"], m["chunk"])
            _meta_lookup[k] = m.get("text", "")
    except Exception:
        pass

    # Use normalized query for topical extraction if available
    topical_source = normalized_query or query
    topical = " ".join(
        t for t in _bm25_tokenize(topical_source)
        if len(t) >= 3 and t not in _FUNCTION_WORDS
    )
    if not topical:
        logger.debug("[Rescue] No topical tokens in query %r", query)
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
            "[Rescue] Topical-corroboration rescue failed", exc_info=True
        )
        return results

    corroborated = {
        (faiss_db.metadata[idx]["pdf_name"],
         faiss_db.metadata[idx]["page"],
         faiss_db.metadata[idx]["chunk"]): round(score, 4)
        for score, idx in hits
    }

    # Local query expansion: when the full topical query yields sparse
    # corroboration, also search with individual content keywords to catch
    # paraphrases/synonyms (e.g. "accomplishments" ≠ "achievements").
    if len(corroborated) < 2:
        individual_keywords = [
            t for t in _bm25_tokenize(topical_source)
            if len(t) >= 3 and t not in _FUNCTION_WORDS
        ]
        for kw in individual_keywords:
            try:
                kw_hits = _bm25_search(
                    kw,
                    faiss_db.metadata,
                    faiss_db._METADATA_VERSION,
                    top_k=max(len(candidates), 1),
                )
                for score, idx in kw_hits:
                    key = (faiss_db.metadata[idx]["pdf_name"],
                           faiss_db.metadata[idx]["page"],
                           faiss_db.metadata[idx]["chunk"])
                    if key not in corroborated:
                        corroborated[key] = round(score, 4)
                    else:
                        corroborated[key] = max(corroborated[key], round(score, 4))
            except Exception:
                pass
        logger.info(
            "[Rescue] Expanded with %d individual keywords → %d corroborated chunk(s)",
            len(individual_keywords), len(corroborated),
        )

    logger.info(
        "[Rescue] Topical=%r — BM25 corroborated %d chunk(s) out of %d candidates",
        topical, len(corroborated), len(candidates),
    )

    # Pre-rerank call: results IS candidates — filter to keep only
    # corroborated candidates (acts as a broad recall filter for the reranker)
    if results is candidates:
        candidate_keys = {
            (c["pdf_name"], c["page"], c["chunk"]) for c in candidates
        }
        rescued_keys = set()
        for c in candidates:
            key = (c["pdf_name"], c["page"], c["chunk"])
            if key in corroborated:
                rescued_keys.add(key)

        # Pull in corroborated chunks from FULL metadata that aren't in the
        # candidate pool — this catches chunks the multi-variant search missed
        # but BM25 topical corroboration found (e.g. paraphrase/synonym hits).
        for key, score in corroborated.items():
            if key not in candidate_keys:
                pdf_name, page, chunk = key
                item = {
                    "pdf_name": pdf_name,
                    "page": page,
                    "chunk": chunk,
                    "text": _meta_lookup.get(key, ""),
                    "hybrid_score": 0.0,
                    "topical_bm25_score": score,
                }
                candidates.append(item)
                rescued_keys.add(key)

        logger.info(
            "[Rescue] Pre-rerank filter: %d / %d candidates corroborated (+ %d pulled from full metadata)",
            len(rescued_keys) - (len(candidates) - len(candidate_keys) - (len(rescued_keys) - len(candidate_keys))),
            len(candidate_keys),
            len(candidates) - len(candidate_keys),
        )

        # If very few candidates are corroborated, keep all to avoid
        # starving the reranker.  Only filter when we have a solid pool.
        if len(rescued_keys) >= 2:
            filtered = []
            for c in candidates:
                key = (c["pdf_name"], c["page"], c["chunk"])
                item = c.copy()
                item["topical_bm25_score"] = corroborated.get(key, 0.0)
                if key in corroborated:
                    filtered.append(item)
            # Always include top hybrid-scored candidates even if not corroborated
            by_hybrid = sorted(
                [c for c in candidates
                 if (c["pdf_name"], c["page"], c["chunk"]) not in rescued_keys],
                key=lambda c: c.get("hybrid_score", 0.0),
                reverse=True,
            )
            for c in by_hybrid[:3]:
                key = (c["pdf_name"], c["page"], c["chunk"])
                item = c.copy()
                item["topical_bm25_score"] = corroborated.get(key, 0.0)
                filtered.append(item)

            logger.info(
                "[Rescue] Pre-rerank kept %d candidates (%d corroborated + %d top hybrid)",
                len(filtered), len(rescued_keys),
                min(len(by_hybrid), 3),
            )
            return filtered

        # Few corroborated — return all candidates unchanged
        for c in candidates:
            c["topical_bm25_score"] = corroborated.get(
                (c["pdf_name"], c["page"], c["chunk"]), 0.0
            )
        return candidates

    # Post-rerank call: restore dropped-but-relevant chunks
    result_keys = {
        (r["pdf_name"], r["page"], r["chunk"]) for r in results
    }
    candidate_keys = {
        (c["pdf_name"], c["page"], c["chunk"]) for c in candidates
    }
    rescued = []
    for candidate in candidates:
        key = (candidate["pdf_name"], candidate["page"], candidate["chunk"])
        if key not in result_keys and key in corroborated:
            item = candidate.copy()
            item["rerank_score"] = None
            item["topical_bm25_score"] = corroborated[key]
            rescued.append(item)

    # Also pull in corroborated chunks from FULL metadata not in candidates
    for key, score in corroborated.items():
        if key not in candidate_keys and key not in result_keys:
            pdf_name, page, chunk = key
            item = {
                "pdf_name": pdf_name,
                "page": page,
                "chunk": chunk,
                "text": _meta_lookup.get(key, ""),
                "rerank_score": None,
                "hybrid_score": 0.0,
                "topical_bm25_score": score,
            }
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

    final = (corroborated_results + others)[:top_k]
    logger.info(
        "[Rescue] Post-rerank: %d original + %d rescued = %d → final %d",
        len(results), len(rescued), len(merged), len(final),
    )
    return final


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
                logger.info("[Search] === PIPELINE START === query=%r", query)

                # Multi-variant retrieval: original + normalized + keywords + rewrites
                candidates = _multi_variant_search(query, RERANK_CANDIDATES)
                logger.info(
                    "[Search] Pre-rerank candidates: %d — top hybrid scores: %s",
                    len(candidates),
                    [round(c.get("hybrid_score", 0), 4) for c in candidates[:5]],
                )

                # Rescue topically corroborated chunks from ALL candidates
                # BEFORE reranking — the cross-encoder's strict relevance gate
                # can discard genuinely relevant chunks (negative scores for
                # short answer passages).  Rescue ensures these chunks survive
                # into the reranking pool.
                variants = _generate_query_variants(query)
                candidates = _rescue_topically_corroborated(
                    query, candidates, candidates, RERANK_CANDIDATES,
                    normalized_query=variants["normalized"],
                )

                results = rerank(query, candidates, top_k=RERANK_TOP_K)
                logger.info(
                    "[Search] After rerank: %d results — scores: %s",
                    len(results),
                    [round(r.get("rerank_score", 0), 4) for r in results[:5]],
                )

                # Second rescue pass: restore any chunk that the reranker
                # dropped but that is topically corroborated by BM25.
                results = _rescue_topically_corroborated(
                    query, candidates, results, RERANK_CANDIDATES,
                    normalized_query=variants["normalized"],
                )
                logger.info(
                    "[Search] Post-rerank final: %d — top rerank scores: %s",
                    len(results), [r.get("rerank_score") for r in results],
                )
                for i, r in enumerate(results[:3]):
                    logger.info(
                        "[Search]   [%d] page=%s chunk=%s hybrid=%.4f rerank=%s",
                        i, r.get("page"), r.get("chunk"),
                        r.get("hybrid_score", 0), r.get("rerank_score"),
                    )
                logger.info("[Search] === PIPELINE END ===")
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
# Chat page
# ---------------------------------------------------------------------------


@app.route("/chat.html")
@app.route("/chat")
def chat_page():
    """Redirect legacy /chat URLs to the unified home page."""
    return redirect(url_for("home"), 302)


# ---------------------------------------------------------------------------
# Chat API endpoints
# ---------------------------------------------------------------------------

CONTEXT_WINDOW = 10  # max recent messages for conversational context

# Reference words that indicate a follow-up query needing context resolution.
# Only call Gemini for context resolution when the query contains one of these.
_FOLLOWUP_MARKERS = frozenset(
    "he she him his her hers they them their theirs it its "
    "that this these those the above the former the latter "
    "himself herself itself themselves "
    "there where when how who whom whose".split()
)


def _resolve_standalone_query(retrieval_query, conversation_history):
    """Use Gemini to resolve follow-up references into a standalone query.

    If conversation_history is empty or very short, return the query as-is.
    Also skips the Gemini call when the query contains no pronouns or reference
    words (he, she, it, that, this, etc.), since such queries are already
    standalone and do not need context resolution.
    """
    if not conversation_history or len(conversation_history) < 2:
        return retrieval_query

    # Fast check: if the query has no pronouns/references, skip Gemini entirely
    query_tokens = set(retrieval_query.lower().split())
    if not query_tokens & _FOLLOWUP_MARKERS:
        return retrieval_query

    # Build a compact conversation context (last N message pairs)
    recent = conversation_history[-(CONTEXT_WINDOW * 2):]
    conv_lines = []
    for msg in recent:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long assistant answers for context
        text = msg["content"]
        if len(text) > 300:
            text = text[:300] + "..."
        conv_lines.append(f"{role_label}: {text}")

    conv_text = "\n".join(conv_lines)

    resolve_prompt = f"""Given the following conversation, rewrite the latest
user question as a standalone, self-contained question that can be understood
without the conversation history. Resolve all pronouns and references (he, she,
they, his, her, that, this, the above person, etc.) to their specific entities.

Do NOT answer the question. Only rewrite it as a standalone question.
If the question is already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation:
{conv_text}

Latest user question: {retrieval_query}

Standalone question:"""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return retrieval_query

    try:
        from utils.gemini import _get_client
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=resolve_prompt,
            config={"max_output_tokens": 128},
        )
        # Extract text safely
        text = ""
        try:
            text = response.text
        except Exception:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                parts = getattr(candidates[0].content, "parts", None) or []
                texts = [p.text for p in parts if getattr(p, "text", None)]
                text = "\n".join(texts)

        resolved = (text or "").strip()
        if resolved and len(resolved) > 5:
            if "The model API is currently overloaded" in resolved:
                raise Exception("API overloaded during context resolution")
            logger.info(
                "[Chat] Context resolve: %r -> %r", retrieval_query, resolved
            )
            return resolved
    except Exception as exc:
        logger.warning("[Chat] Context resolution failed: %s — using raw query", exc)

    return retrieval_query


@app.route("/chat", methods=["POST"])
def chat_send():
    """Process a chat message and return the assistant response.

    Request JSON:
        session_id (optional): existing session UUID
        message: user's question text

    Response JSON:
        session_id: session UUID (new or existing)
        answer: assistant response text (with inline citations)
        answer_html: response with clickable citation links
        sources: list of cited source dicts
    """
    _t_total = time.time()
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = (data.get("session_id") or "").strip() or None

    if not message:
        return jsonify({"error": "Message is required"}), 400

    _t_hist = time.time()
    # --- Session handling ---
    if session_id:
        existing_session = chat_db.get_session(session_id)
        if existing_session is None:
            session_id = chat_db.create_session(
                title=message[:60] + ("..." if len(message) > 60 else "")
            )
    else:
        title = message[:60] + ("..." if len(message) > 60 else "")
        session_id = chat_db.create_session(title=title)

    chat_db.add_message(session_id, "user", message)
    _perf_hist = round((time.time() - _t_hist) * 1000)
    logger.info("[PERF] session/history: %dms", _perf_hist)

    # --- RAG pipeline ---
    answer = None
    answer_html = None
    cited_sources = []
    gemini_calls = 0

    try:
        if total_vectors() == 0:
            answer = "No documents indexed. Please upload a PDF first."
        else:
            _t_ctx = time.time()
            conv_history = chat_db.get_recent_messages(session_id, limit=CONTEXT_WINDOW)
            _perf_ctx = round((time.time() - _t_ctx) * 1000)

            _t_norm = time.time()
            resolved_query = _resolve_standalone_query(message, conv_history)
            retrieval_query = normalize_query(resolved_query)
            original_normalized = normalize_query(message)
            _perf_norm = round((time.time() - _t_norm) * 1000)

            # Determine if this is a follow-up query needing Gemini context resolution
            query_tokens = set(message.lower().split())
            is_followup = bool(query_tokens & _FOLLOWUP_MARKERS)
            if is_followup:
                gemini_calls += 1  # context resolution used Gemini

            logger.info(
                "[Chat] Session=%s original=%r resolved=%r normalized=%r followup=%s",
                session_id, message, resolved_query, retrieval_query, is_followup,
            )

            _t_retrieval = time.time()
            # Determine if we can skip Gemini query rewrites (simple standalone query)
            # Skip rewrites when: no follow-up AND original == normalized (no typos)
            skip_rewrites = (not is_followup and original_normalized == retrieval_query)
            candidates = _multi_variant_search(
                retrieval_query, RERANK_CANDIDATES, skip_rewrites=skip_rewrites
            )
            if not skip_rewrites:
                gemini_calls += 1  # query rewrites used Gemini

            # Also search with original query if it differs from resolved
            if original_normalized != retrieval_query:
                emb_orig = create_query_embedding(original_normalized)
                cands_orig = hybrid_search(
                    original_normalized, emb_orig, top_k=RERANK_CANDIDATES
                )
                seen = {(c["pdf_name"], c["page"], c["chunk"]) for c in candidates}
                for c in cands_orig:
                    key = (c["pdf_name"], c["page"], c["chunk"])
                    if key not in seen:
                        candidates.append(c)
                        seen.add(key)
            _perf_retrieval = round((time.time() - _t_retrieval) * 1000)

            # Cap the candidate pool to avoid excessive reranker/rescue processing.
            # Keep top_k candidates from multi-variant merge (sorted by hybrid_score).
            # This ensures the reranker never sees hundreds of candidates.
            _RERANKER_MAX = RERANK_CANDIDATES * 3  # generous headroom for rescue
            if len(candidates) > _RERANKER_MAX:
                candidates.sort(key=lambda c: c.get("hybrid_score", 0.0), reverse=True)
                candidates = candidates[:_RERANKER_MAX]

            _t_rescue1 = time.time()
            candidates = _rescue_topically_corroborated(
                retrieval_query, candidates, candidates, RERANK_CANDIDATES,
                normalized_query=retrieval_query,
            )
            _perf_rescue1 = round((time.time() - _t_rescue1) * 1000)

            _t_rerank = time.time()
            results = rerank(retrieval_query, candidates, top_k=RERANK_TOP_K)
            _perf_rerank = round((time.time() - _t_rerank) * 1000)

            _t_rescue2 = time.time()
            results = _rescue_topically_corroborated(
                retrieval_query, candidates, results, RERANK_CANDIDATES,
                normalized_query=retrieval_query,
            )
            _perf_rescue2 = round((time.time() - _t_rescue2) * 1000)

            _t_gemini = time.time()
            if results:
                results, citations = build_citation_map(results)
                for result in results:
                    result["pdf_url"] = pdf_url_for(result)
                    result["display_pdf_name"] = display_pdf_name(
                        result["pdf_name"]
                    )
                try:
                    answer = generate_answer(message, results)
                    gemini_calls += 1  # answer generation
                except GeminiGenerationError as exc:
                    logger.error(
                        "[Chat] Gemini generation failed (kind=%s): %s",
                        exc.kind, exc, exc_info=True,
                    )
                    answer = _fallback_answer(results, exc)
                if answer:
                    answer = validate_answer(answer, citations)
                    answer_html = render_answer_links(
                        answer, citations, pdf_url_for
                    )
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
                answer = (
                    "I couldn't find relevant information in the uploaded "
                    "documents for this question."
                )
            _perf_gemini = round((time.time() - _t_gemini) * 1000)
    except Exception as exc:
        logger.error("[Chat] Error processing message: %s", exc, exc_info=True)
        answer = "An error occurred while processing your question. Please try again."
        _perf_gemini = 0
        _perf_retrieval = 0
        _perf_rerank = 0
        _perf_rescue1 = 0
        _perf_rescue2 = 0
        _perf_norm = 0
        _perf_ctx = 0

    _t_save = time.time()
    chat_db.add_message(session_id, "assistant", answer or "", cited_sources)
    _perf_save = round((time.time() - _t_save) * 1000)
    _perf_total = round((time.time() - _t_total) * 1000)

    logger.info(
        "[PERF] ctx=%dms norm=%dms retrieval=%dms rescue1=%dms reranker=%dms "
        "rescue2=%dms gemini=%dms save=%dms total=%dms gemini_calls=%d",
        _perf_ctx, _perf_norm, _perf_retrieval, _perf_rescue1, _perf_rerank,
        _perf_rescue2, _perf_gemini, _perf_save, _perf_total, gemini_calls,
    )

    return jsonify({
        "session_id": session_id,
        "answer": answer,
        "answer_html": answer_html or answer,
        "sources": cited_sources,
    })


@app.route("/chat/sessions", methods=["GET"])
def chat_sessions_list():
    """Return all chat sessions ordered by most recent."""
    sessions = chat_db.list_sessions(limit=100)
    return jsonify(sessions)


@app.route("/chat/sessions/<session_id>", methods=["GET"])
def chat_session_messages(session_id):
    """Return all messages for a specific session."""
    session = chat_db.get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    messages = chat_db.get_messages(session_id, limit=200)
    return jsonify({
        "session": session,
        "messages": messages,
    })


@app.route("/chat/sessions/<session_id>", methods=["DELETE"])
def chat_session_delete(session_id):
    """Delete a chat session and all its messages."""
    chat_db.delete_session(session_id)
    return jsonify({"status": "ok"})


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
