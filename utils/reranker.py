"""Second-stage cross-encoder reranking over the hybrid candidate pool.

Runs AFTER hybrid_search (BM25 + FAISS fusion) and BEFORE Gemini.  A lazily
loaded singleton ``sentence_transformers.CrossEncoder`` re-scores the
retrieved candidates against the query, so the retrieval pool can be larger
than the final context budget without touching hybrid scoring.

Each returned candidate keeps every existing field (pdf_name, page, chunk,
text, faiss_score, bm25_score, hybrid_score, score) and gains a
``rerank_score``; ``hybrid_score`` is never modified.  Any loading/scoring
failure is logged and the original candidate order is returned so search
never breaks.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Return the shared CrossEncoder, loading it lazily on first use."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import CrossEncoder
                model_name = os.getenv("RERANK_MODEL", DEFAULT_MODEL)
                logger.info("[Rerank] Loading model %r (first use)", model_name)
                _model = CrossEncoder(model_name)
    return _model


def _as_scalar(score):
    """Coerce a CrossEncoder output to a plain float score."""
    if isinstance(score, (list, tuple)):
        return float(score[0]) if score else 0.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def rerank(query, candidates, top_k=5):
    """Rerank *candidates* against *query*, returning the top ``top_k``.

    Adds ``rerank_score`` to each candidate and sorts descending.  On any
    model/score failure, logs the error and falls back to the original
    candidate order (still truncated to ``top_k``).
    """
    if not candidates:
        return []

    try:
        model = _get_model()
        pairs = [(query, (c.get("text") or "")) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)  # type: ignore[reportUnknownMemberType]
    except Exception:
        logger.error(
            "[Rerank] Scoring failed — using original candidate order",
            exc_info=True,
        )
        return list(candidates[:top_k])

    ranked = []
    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        item["rerank_score"] = round(_as_scalar(score), 4)
        ranked.append(item)

    ranked.sort(key=lambda r: r["rerank_score"], reverse=True)
    # Filter out highly irrelevant candidates (negative cross-encoder scores)
    ranked = [r for r in ranked if r["rerank_score"] > 0.0]
    return ranked[:top_k]
