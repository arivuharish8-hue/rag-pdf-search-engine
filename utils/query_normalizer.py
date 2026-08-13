"""Safe typo-tolerant query normalization for retrieval.

Runs *before* BM25 + FAISS retrieval.  It performs ONLY conservative
spelling/typo correction and NEVER rewrites, paraphrases or restructures a
valid question:

  * question words (``what``, ``who``, ``where``, ``when``, ``how``, ...) and
    ordinary English function words are never treated as typos -- a valid
    question such as "what are the achievements of haridass" passes through
    exactly as written,
  * a token is only rewritten when it is clearly NOT a valid word: not in the
    indexed corpus, not a common English word, a plain lower-case /
    capitalized word (no digits, no all-caps acronyms, no CVE / number / ID
    tokens), and it has exactly one close vocabulary match within a bounded
    edit distance that is strictly closer than every other candidate.

Anything else -- correct spellings, question words, function words, technical
terms, numbers, IDs, acronyms, names already present in the corpus -- is left
untouched.  When no token can be safely corrected the original query is
returned unchanged.  ``normalize_query`` returns the same object when nothing
changed, so callers can cheaply detect whether an actual typo was corrected.
The ORIGINAL query is preserved for display and for the answer generator; the
minimally corrected form is used only for retrieval.
"""

import re
import threading

from utils.bm25 import tokenize as bm25_tokenize

# Tokens shorter than this are too ambiguous to correct.
_MIN_TOKEN_LENGTH = 4
# Edit-distance budget: 1 for short tokens, 2 for longer ones.
_MAX_EDIT = 1
_MAX_EDIT_LONG = 2
_LONG_TOKEN_LENGTH = 8

# Only bare alpha words are considered as typos or as candidate corrections.
# Anything with digits, hyphens, underscores etc. is treated as an exact
# identifier (CVE-2021-3560, ID_POLKIT, v2.5) and never touched.
_PLAIN_WORD = re.compile(r"^[a-z]+$")
_LETTERS = re.compile(r"[a-zA-Z]+")

# Common English words that are NEVER treated as typos.  A token in this set
# is a valid word, so rewriting it (e.g. "what" -> "that") would corrupt the
# user's question rather than fix a typo.  This keeps every question word and
# function word intact ("what are the achievements of haridass" stays exactly
# as written) while still allowing genuine misspellings ("achivements",
# "shanumugam") to be corrected because those are not real words.
_COMMON_WORDS = frozenset(
    # question words
    "what who where when how why which whose whom whatever whoever"
    # articles / determiners
    " the a an this that these those some any no every each both either"
    " neither all few many much more most other another such several"
    # pronouns
    " i you he she it we they me him her us them my your our their his its"
    " mine yours ours theirs myself yourself himself herself itself"
    " ourselves themselves"
    # prepositions
    " of to in on at by for with without from into onto upon over under"
    " above below between among through during before after about around"
    " across along behind beyond inside outside until since towards within"
    # conjunctions
    " and but or nor so yet because although though while whereas unless"
    # auxiliaries / common verbs
    " is am are was were be been being do does did done doing have has had"
    " having can could will would shall should may might must need"
    # adverbs / the rest
    " not no too very also just only then than now here there well really"
    " already still always never often sometimes again ever even"
    # request phrasing often found in questions
    " please tell show give name list describe explain know"
    .split()
)

_vocab = frozenset()
_vocab_owner = None
_vocab_lock = threading.Lock()


def _levenshtein(a, b):
    """Standard Levenshtein distance (small buffers, non-recursive)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def build_vocabulary(metadata):
    """Known lower-case words from the indexed corpus (chunk text + pdf name)."""
    words = set()
    for m in metadata or []:
        text = "%s %s" % (m.get("text") or "", m.get("pdf_name") or "")
        for tok in bm25_tokenize(text):
            if _PLAIN_WORD.fullmatch(tok) and len(tok) >= _MIN_TOKEN_LENGTH:
                words.add(tok)
    return frozenset(words)


def _get_vocab(metadata):
    """Return the corpus vocabulary, cached per metadata snapshot.

    ``metadata`` is swapped wholesale by ``utils.faiss_db`` whenever it is
    loaded or mutated, so ``id(metadata)`` is a cheap freshness key.
    """
    global _vocab, _vocab_owner
    owner = id(metadata)
    with _vocab_lock:
        if _vocab_owner == owner and _vocab:
            return _vocab
        _vocab = build_vocabulary(metadata)
        _vocab_owner = owner
        return _vocab


def _best_correction(token, vocab):
    """Single best vocabulary word within the edit budget, else None.

    Returns None when there is no candidate, or when the two closest
    candidates are tied (ambiguous -- not confident enough to rewrite).
    """
    max_edit = _MAX_EDIT_LONG if len(token) >= _LONG_TOKEN_LENGTH else _MAX_EDIT
    best = None
    best_dist = max_edit + 1
    for word in vocab:
        if word == token or abs(len(word) - len(token)) > max_edit:
            continue
        dist = _levenshtein(token, word)
        if dist < best_dist:
            best, best_dist = word, dist
        elif dist == best_dist:
            best = None  # tie -> not a confident correction
    return best


def _correction_for(token, vocab):
    """Return the safe correction for one token, or None to keep it as-is."""
    if len(token) < _MIN_TOKEN_LENGTH:
        return None
    lower = token.lower()
    if not _PLAIN_WORD.fullmatch(lower):
        return None  # digits / symbols / hyphenated identifiers
    if lower in _COMMON_WORDS:
        return None  # valid question/function word -> never rewrite it
    if lower in vocab:
        return None  # already a known term -> spelled correctly
    if token.isupper():
        return None  # all-caps: acronym or identifier, do not touch
    best = _best_correction(lower, vocab)
    if best is None:
        return None
    if token[:1].isupper() and token[1:].islower():
        return best.capitalize()
    return best


def normalize_query_with_vocab(query, vocab):
    """Normalize *query* against an explicit vocabulary (testable / pure).

    Returns the corrected query string, or the original object when no token
    was safely corrected.
    """
    if not query:
        return query
    out = []
    last = 0
    changed = False
    for m in _LETTERS.finditer(query):
        out.append(query[last:m.start()])
        token = m.group(0)
        correction = _correction_for(token, vocab)
        if correction is None:
            out.append(token)
        else:
            out.append(correction)
            changed = True
        last = m.end()
    if not changed:
        return query
    out.append(query[last:])
    return "".join(out)


def normalize_query(query):
    """Normalize *query* against the live indexed corpus.

    Entry point used by the search route, before FAISS/BM25 retrieval.
    """
    if not query or not query.strip():
        return query
    from utils import faiss_db  # lazy: avoids import-time dependency on app
    vocab = _get_vocab(faiss_db.metadata)
    if not vocab:
        return query
    return normalize_query_with_vocab(query, vocab)
