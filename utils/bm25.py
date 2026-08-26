"""BM25 keyword retrieval over the same chunk collection as FAISS.

The BM25 index is built from the metadata list's ``text`` field (the exact
chunk texts FAISS indexes) and is rebuilt by ``utils.faiss_db`` only when the
chunk collection changes — never per query — so keyword search can never go
stale relative to the semantic index.

Rebuilds are keyed on a monotonically increasing version integer that
``utils.faiss_db`` bumps whenever it reloads or mutates its metadata.
"""

import re
import threading

from rank_bm25 import BM25Okapi

# Tokens are whitespace-delimited, lower-cased, with surrounding punctuation
# stripped.  This keeps exact identifiers such as "CVE-2021-3560",
# "ID_POLKIT", "react.js" or "linkedin.com/in/shanmugam" as single tokens so
# keyword matches on them are strong, while ordinary prose still tokenizes
# on word boundaries.
_STRIP_CHARS = ".,;:!?()[]{}\"'`<>|/\\=+*^$#@%~"
_TOKEN_SPLIT = re.compile(r"\s+")

_bm25 = None
_built_version = -1
_LOCK = threading.Lock()


def tokenize(text):
    """Tokenize one text or query for BM25 indexing / scoring."""
    tokens = _TOKEN_SPLIT.split((text or "").lower())
    return [t.strip(_STRIP_CHARS) for t in tokens if t.strip(_STRIP_CHARS)]


def build(metadata, version):
    """(Re)build the BM25 index from ``metadata`` chunk texts."""
    global _bm25, _built_version
    corpus = [tokenize(m.get("text", "")) for m in metadata]
    with _LOCK:
        if any(corpus):
            _bm25 = BM25Okapi(corpus)
        else:
            _bm25 = None  # empty collection — rank_bm25 cannot index 0 docs
        _built_version = version


def _ensure(metadata, version):
    """Return the BM25 index, rebuilding it if the collection changed."""
    global _bm25, _built_version
    with _LOCK:
        if _bm25 is None or _built_version != version:
            corpus = [tokenize(m.get("text", "")) for m in metadata]
            if any(corpus):
                _bm25 = BM25Okapi(corpus)
            else:
                _bm25 = None
            _built_version = version
        return _bm25


def search(query_text, metadata, version, top_k=10):
    """Return up to ``top_k`` ``(raw_score, metadata_index)`` pairs by BM25.

    Metadata index positions line up with FAISS vector ids, so a BM25 hit can
    be merged with its FAISS counterpart by identity.  Returns ``[]`` when
    there is nothing indexed.
    """
    bm25 = _ensure(metadata, version)
    if bm25 is None:
        return []

    scores = bm25.get_scores(tokenize(query_text))
    results = []
    for idx in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
        if scores[idx] <= 0:
            continue
        results.append((float(scores[idx]), idx))
        if len(results) >= top_k:
            break
    return results
