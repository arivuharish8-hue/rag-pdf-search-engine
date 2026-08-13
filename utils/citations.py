"""Citation mapping, validation, renumbering, and clickable-link rendering.

Citations are assigned to the *final candidate* chunks that are actually passed
to Gemini, so a citation ID always maps to a real retrieved source
(citation_id, pdf_name, page, chunk, text).  The same IDs are used in the
Gemini context ([SOURCE n]) and in the raw answer ([n]).

Before the user sees the answer, citations are validated and *renumbered*:
only the IDs Gemini actually used survive, and they are remapped to a
contiguous [1], [2], ... sequence.  The internal [SOURCE n] numbering is never
exposed to the user, and uncited retrieved candidates never appear in the
Sources section.

Gemini may emit compound citations such as [1, 2], [1,2] or [1-3].  These are
parsed into individual citation IDs, validated, renumbered, and rendered as
separate clickable links ([1][2] or [1][2][3]).
"""

import html
import logging
import re

logger = logging.getLogger(__name__)

# A citation token is a bracket pair whose content is only digits, commas,
# spaces and hyphens (e.g. [1], [10], [1, 2], [1-3]).  Non-citation bracketed
# text such as "[Source 1]" is deliberately NOT matched.
CITATION_TOKEN = re.compile(r"\[([0-9][0-9,\s\-]*)\]")


def _token_ids(content):
    """Expand a citation token's content into individual citation IDs.

    "1, 2" -> {1, 2}; "1-3" -> {1, 2, 3}; "1" -> {1}.
    """
    compact = re.sub(r"\s+", "", content)
    ids = set()
    for part in compact.split(","):
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            if start.isdigit() and end.isdigit() and int(end) >= int(start):
                ids.update(range(int(start), int(end) + 1))
        elif part.isdigit():
            ids.add(int(part))
    return ids


def build_citation_map(results):
    """Assign stable citation IDs [1], [2], ... to the final reranked chunks.

    Each result dict gains a ``citation_id`` field (in place) so the frontend
    and Gemini context share the exact same mapping.  Returns
    ``(results, citations)`` where ``citations`` is a list of
    {citation_id, pdf_name, page, chunk, text} dicts.
    """
    citations = []
    for index, res in enumerate(results, start=1):
        res["citation_id"] = index
        citations.append({
            "citation_id": index,
            "pdf_name": res.get("pdf_name", "Unknown"),
            "page": res.get("page", 1),
            "chunk": res.get("chunk", 0),
            "text": res.get("text", ""),
        })
    return results, citations


def extract_cited_ids(answer):
    """Return the set of citation IDs referenced in *answer*."""
    ids = set()
    for match in CITATION_TOKEN.finditer(answer):
        ids.update(_token_ids(match.group(1)))
    return ids


def renumber_citations(answer, citations):
    """Renumber the citations actually used in *answer* to a clean [1], [2], ...

    Gemini (or the fallback path) may reference internal candidate IDs that
    start above [1] — e.g. only ``[2]`` was cited even though nothing else is
    displayed.  The user must never see those internal numbers: every cited ID
    is remapped to a fresh sequential ID (ascending by internal ID) and the
    answer text is rewritten so the answer and the Sources section always
    agree on the numbering.

    Non-citation brackets are left untouched.  IDs not present in ``citations``
    are dropped (call ``validate_answer`` first to do the same before
    renumbering).

    Returns ``(new_answer, cited_citations)``:
      * ``new_answer``: the answer with renumbered citation tokens.
      * ``cited_citations``: one entry per *cited* source, in the new citation
        order, with the new ``citation_id`` and the original
        pdf_name/page/chunk/text preserved.
    """
    by_id = {c["citation_id"]: c for c in citations}
    order = sorted(extract_cited_ids(answer))
    mapping = {old: new for new, old in enumerate(order, start=1)}

    def _replace(match):
        ids = sorted(_token_ids(match.group(1)))
        if not ids:
            return match.group(0)
        return "[%s]" % ", ".join(str(mapping[n]) for n in ids if n in mapping)

    new_answer = CITATION_TOKEN.sub(_replace, answer)
    cited = [
        dict(by_id[old], citation_id=new)
        for old, new in sorted(mapping.items(), key=lambda kv: kv[1])
    ]
    return new_answer, cited


def validate_answer(answer, citations):
    """Remove citation references to IDs that do not exist in the mapping.

    Non-citation brackets (e.g. "[Source 1]") are left untouched.  Invalid IDs
    are dropped from compound tokens (e.g. [1, 9] -> [1]); tokens with no
    valid IDs are removed entirely.  The answer text is otherwise unchanged.
    Removed IDs are logged for debugging.
    """
    valid_ids = {c["citation_id"] for c in citations}

    def _replace(match):
        ids = _token_ids(match.group(1))
        if not ids:
            return match.group(0)
        good = sorted(ids & valid_ids)
        bad = sorted(ids - valid_ids)
        if bad:
            logger.warning(
                "[Citations] Answer referenced invalid citation ID(s) %s — removed.",
                bad,
            )
        if not good:
            return ""
        return "[%s]" % ", ".join(str(num) for num in good)

    return CITATION_TOKEN.sub(_replace, answer)


def render_answer_links(answer, citations, url_builder):
    """Escape *answer* and wrap valid citations in clickable links.

    ``url_builder(citation)`` must return the browser-accessible PDF URL for a
    citation WITHOUT the page fragment; the ``#page=N`` fragment is appended
    here so each citation opens its exact source page.

    The answer text is HTML-escaped before linking, so Gemini output can never
    inject markup.  Each valid citation ID becomes its own link; compound
    tokens (e.g. [1, 2]) render as adjacent links ([1][2]).
    """
    by_id = {c["citation_id"]: c for c in citations}

    def _link(cit):
        base_url = url_builder(cit)
        href = html.escape(
            "%s#page=%s" % (base_url, cit["page"]), quote=True
        )
        title = html.escape(
            "%s — page %s" % (cit["pdf_name"], cit["page"]), quote=True
        )
        return (
            '<a class="citation-link" href="%s" target="_blank" '
            'rel="noopener" title="%s">[%s]</a>'
            % (href, title, cit["citation_id"])
        )

    def _replace(match):
        ids = _token_ids(match.group(1))
        if not ids:
            return match.group(0)
        links = []
        for num in sorted(ids):
            cit = by_id.get(num)
            if cit is not None:
                links.append(_link(cit))
        return "".join(links)

    escaped = html.escape(answer)
    return CITATION_TOKEN.sub(_replace, escaped)
