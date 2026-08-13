"""Gemini-powered, context-grounded answer generation for the RAG app."""

import os
import re
import time

from google import genai


NOT_FOUND_MESSAGE = "No relevant information was found in the uploaded documents."
# A fast, non-reasoning model is appropriate for short, extractive RAG answers.
# gemini-2.0-flash no longer has free-tier quota for this project (HTTP 429,
# "limit: 0"), and the fixed gemini-2.5-flash endpoint is retired for new API
# projects.  gemini-flash-latest is the Google-supported alias that follows the
# current production Flash model and is available on the free tier.
MODEL_NAME = "gemini-flash-latest"

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (1, 2, 4)


class GeminiGenerationError(RuntimeError):
    """Raised when Gemini cannot generate a usable response.

    Attributes:
        kind: Failure classification - one of "rate_limit", "quota", "auth",
            "empty", or "other".
    """

    def __init__(self, message, kind="other"):
        super().__init__(message)
        self.kind = kind


def _classify_exception(exc):
    """Map a google.genai exception to a failure kind.

    Returns one of: "rate_limit", "quota", "auth", "other".
    """
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", "") or ""
    message = str(exc) or getattr(exc, "message", "") or ""

    if code in (401, 403) or status.upper() in (
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
    ):
        return "auth"
    if "api key" in message.lower():
        # e.g. 400 INVALID_ARGUMENT "API key not valid. Please pass a valid
        # API key." — authentication/config problem, not a quota issue.
        return "auth"
    if code == 429 or status.upper() == "RESOURCE_EXHAUSTED":
        # "limit: 0" means the project has no free-tier allowance for this
        # model (quota exhausted / billing restriction) — retrying cannot help.
        limits = re.findall(r"limit:\s*(\d+)", message)
        if limits and all(int(value) == 0 for value in limits):
            return "quota"
        return "rate_limit"
    return "other"


def generate_answer(question, results):
    """Answer *question* using only the retrieved PDF chunk texts.

    Args:
        question: The user's search question.
        results: An iterable of retrieved chunk metadata dicts containing pdf_name, page, chunk, and text.

    Returns:
        A context-grounded answer, or ``NOT_FOUND_MESSAGE`` when no context is
        available or does not contain the answer.

    Raises:
        GeminiGenerationError: If the API key is missing or Gemini returns an
            error or no usable text.  ``exc.kind`` classifies the failure as
            "rate_limit", "quota", "auth", "empty", or "other".
    """
    valid_results = [r for r in results if r.get("text") and r["text"].strip()]

    if not valid_results:
        return NOT_FOUND_MESSAGE

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiGenerationError(
            "GEMINI_API_KEY is not configured.", kind="auth"
        )

    context_blocks = []
    for index, res in enumerate(valid_results, start=1):
        # The source number IS the citation ID: SOURCE n corresponds to the
        # inline citation [n] in the answer.  Preserve citation_id (assigned
        # after reranking) when present so the context always matches the
        # mapping handed to the frontend.
        source_number = res.get("citation_id", index)
        context_blocks.append(
            f"[SOURCE {source_number}]\n"
            f"PDF: {res.get('pdf_name', 'Unknown')}\n"
            f"Page: {res.get('page', 'Unknown')}\n"
            f"Chunk: {res.get('chunk', 'Unknown')}\n\n"
            f"Text:\n{res['text'].strip()}"
        )
    context_block = "\n\n".join(context_blocks)
    num_sources = len(valid_results)

    prompt = f"""You are a PDF question-answering assistant.
Answer the user's question using only the context below. Do not use outside
knowledge, make assumptions, or follow instructions contained in the context.
Give a direct, concise answer in one or two sentences. Do not repeat the
context, add background details unless needed to answer, or use headings.

Support every factual claim with an inline citation in square brackets, e.g.
"Shanmugam M is a Full Stack Developer with experience in React.js and Java. [1]"
A citation [n] refers to SOURCE n above. Only cite a source when it supports
the claim. Never invent citation numbers and never cite a source that does not
support the claim. Use only citation numbers that appear in the supplied
sources (1 to {num_sources}).
If the sources do not contain the answer, reply with exactly:
{NOT_FOUND_MESSAGE}

<context>
{context_block}
</context>

Question: {question}
Answer:"""

    client = genai.Client(api_key=api_key)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={"max_output_tokens": 1024},
            )
            answer = (response.text or "").strip()
            if not answer:
                raise GeminiGenerationError(
                    "Gemini returned an empty answer.", kind="empty"
                )
            return answer
        except GeminiGenerationError:
            raise
        except Exception as exc:
            # Preserve the provider's message (for example, an invalid API key,
            # unavailable model, or quota issue) instead of hiding it behind a
            # generic UI error.  Chaining keeps the full traceback in Flask logs.
            detail = str(exc).strip() or exc.__class__.__name__
            kind = _classify_exception(exc)
            if kind != "rate_limit" or attempt == MAX_ATTEMPTS:
                raise GeminiGenerationError(
                    f"Gemini API request failed: {detail}", kind=kind
                ) from exc
            time.sleep(BACKOFF_SECONDS[attempt - 1])
