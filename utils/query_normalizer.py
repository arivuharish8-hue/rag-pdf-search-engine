"""Safe typo-tolerant query normalization with abbreviation expansion.

Runs *before* BM25 + FAISS retrieval.  It performs:
  1. Abbreviation expansion: short tokens (2-3 chars) that look like initials
     are expanded to matching multi-word phrases from the corpus vocabulary.
     e.g. "ms" -> "mahendra singh", "ap" -> "andhra pradesh"
  2. Conservative spelling/typo correction: a token is only rewritten when it
     is clearly NOT a valid word and has exactly one close vocabulary match.

Question words and function words are NEVER rewritten.  The ORIGINAL query is
preserved for display and for the answer generator; the minimally corrected
form is used only for retrieval.
"""

import re
import threading
from itertools import product as _product

from utils.bm25 import tokenize as bm25_tokenize

# Tokens shorter than this are too ambiguous to correct.
_MIN_TOKEN_LENGTH = 4

# Abbreviation expansion: tokens of this length or shorter are candidates for
# expansion into multi-word phrases from the corpus.
_ABBREV_MAX_LENGTH = 3
# Minimum expansion length: only expand if the replacement has at least this
# many words (e.g. "ms" -> "mahendra singh" is 2 words, acceptable).
_ABBREV_MIN_WORDS = 2

# Only bare alpha words are considered as abbreviations, typos or candidates.
_PLAIN_WORD = re.compile(r"^[a-z]+$")
_LETTERS = re.compile(r"[a-zA-Z]+")

# Common English words that are NEVER treated as typos or abbreviations.
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


def _is_abbreviation_token(token):
    """Return True if *token* looks like an abbreviation/initials.

    Common English words (question words, prepositions, etc.) are NEVER
    treated as abbreviations even if they are short.
    """
    lower = token.lower()
    # Skip common English words — "who", "is", "what", etc. are not abbreviations
    if lower in _COMMON_WORDS:
        return False
    # All-caps acronyms like "USA" or "BJP" — treat as abbreviations
    if token.isupper() and len(token) <= _ABBREV_MAX_LENGTH:
        return True
    # Short lowercase/capitalized tokens like "ms", "Msk", "ap"
    if len(token) <= _ABBREV_MAX_LENGTH and _PLAIN_WORD.fullmatch(lower):
        return True
    return False


def _expand_abbreviations(query, vocab):
    """Expand short abbreviation tokens in *query* using corpus vocabulary.

    Only expands tokens that look like abbreviations (2-3 chars) and can be
    matched to multi-word phrases in the corpus.  Returns the expanded query
    or the original if nothing changed.

    Expansion requires that the candidate words appear ADJACENT in the corpus
    (as a name phrase), preventing random expansions like "ms" -> "management system".
    """
    if not query or not vocab:
        return query

    import logging
    _logger = logging.getLogger(__name__)

    from utils import faiss_db
    metadata = faiss_db.metadata or []

    # Build word list and adjacent word-pair set from corpus
    all_words = set()
    word_pairs = set()  # (word1, word2) when adjacent in a chunk
    word_freq = {}

    for m in metadata:
        text = m.get("text") or ""
        tokens = [t.lower() for t in bm25_tokenize(text) if _PLAIN_WORD.fullmatch(t)]
        all_words.update(tokens)
        for t in tokens:
            word_freq[t] = word_freq.get(t, 0) + 1
        for i in range(len(tokens) - 1):
            word_pairs.add((tokens[i], tokens[i + 1]))

    word_list = sorted(all_words)
    if not word_list:
        return query

    _logger.debug("[QueryNorm] vocab=%d, words=%d, pairs=%d",
                  len(vocab), len(word_list), len(word_pairs))

    out = []
    last = 0
    changed = False

    for m in _LETTERS.finditer(query):
        out.append(query[last:m.start()])
        token = m.group(0)

        if _is_abbreviation_token(token) and len(token) <= _ABBREV_MAX_LENGTH:
            initials = list(token.lower())
            candidates_per_letter = []
            valid = True
            for letter in initials:
                matching = [w for w in word_list if w.startswith(letter)]
                if not matching:
                    valid = False
                    break
                # Top 10 by frequency
                matching.sort(key=lambda w: (-word_freq.get(w, 0), w))
                candidates_per_letter.append(matching[:10])

            if valid and candidates_per_letter:
                best_expansion = None
                best_pair_score = -1

                for combo in _product(*candidates_per_letter):
                    if len(combo) < _ABBREV_MIN_WORDS:
                        continue
                    # ONLY accept expansions where adjacent words actually
                    # appear together in the corpus — prevents random pairings
                    pair_hits = sum(
                        1 for i in range(len(combo) - 1)
                        if (combo[i], combo[i + 1]) in word_pairs
                    )
                    if pair_hits > best_pair_score:
                        best_pair_score = pair_hits
                        best_expansion = combo

                if best_expansion and best_pair_score > 0:
                    expansion = " ".join(best_expansion)
                    _logger.info("[QueryNorm] Expanded %r -> %r (pair_hits=%d)",
                                 token, expansion, best_pair_score)
                    out.append(expansion)
                    changed = True
                    last = m.end()
                    continue
                else:
                    _logger.debug(
                        "[QueryNorm] No co-occurring expansion for %r "
                        "(best_pair_score=%d)", token, best_pair_score)

        out.append(token)
        last = m.end()

    if not changed:
        return query
    out.append(query[last:])
    return "".join(out)


# Edit-distance budget: 1 for short tokens, 2 for longer ones.
_MAX_EDIT = 1
_MAX_EDIT_LONG = 2
_LONG_TOKEN_LENGTH = 8

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
    First tries abbreviation expansion, then typo correction.
    """
    if not query or not query.strip():
        return query
    from utils import faiss_db  # lazy: avoids import-time dependency on app
    import logging
    _logger = logging.getLogger(__name__)

    vocab = _get_vocab(faiss_db.metadata)
    if not vocab:
        _logger.debug("[QueryNorm] No vocabulary, returning original query")
        return query

    _logger.debug("[QueryNorm] Input query: %r, vocab size: %d", query, len(vocab))

    # Step 1: Try abbreviation expansion (e.g. "ms dhoni" -> "mahendra singh dhoni")
    expanded = _expand_abbreviations(query, vocab)
    if expanded is not query:
        # Abbreviation was expanded — use expanded form for retrieval
        # but also run typo correction on the expanded form
        _logger.info("[QueryNorm] Abbreviation expanded: %r -> %r", query, expanded)
        return normalize_query_with_vocab(expanded, vocab)

    # Step 2: No abbreviation expansion — try typo correction
    result = normalize_query_with_vocab(query, vocab)
    if result is not query:
        _logger.info("[QueryNorm] Typo corrected: %r -> %r", query, result)
    return result
