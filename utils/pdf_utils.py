"""PDF text extraction and word-based chunking."""

import logging
import re

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def chunk_text(text, chunk_size=120, overlap=20):
    """Split text into small, overlapping word-based chunks."""

    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def extract_text_from_pdf(pdf_path, pdf_name):
    """Extract text page by page and create chunk metadata dicts.

    Args:
        pdf_path: Path to the PDF file on disk.
        pdf_name: Logical name (used in chunk metadata).

    Returns:
        List of dicts with keys: pdf_name, page, chunk, text.
        Returns an empty list if the PDF has no extractable text.
    """
    logger.info("[PDF] Opening %s (%s)", pdf_name, pdf_path)
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    logger.info("[PDF] %s: %d page(s)", pdf_name, total_pages)

    all_chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            logger.debug("[PDF] %s page %d: no text", pdf_name, page_no)
            continue

        page_chunks = chunk_text(text)
        logger.debug("[PDF] %s page %d: %d chunk(s)", pdf_name, page_no,
                     len(page_chunks))

        for chunk_no, chunk in enumerate(page_chunks, start=1):
            all_chunks.append({
                "pdf_name": pdf_name,
                "page": page_no,
                "chunk": chunk_no,
                "text": chunk,
            })

    logger.info("[PDF] %s: extracted %d chunk(s) from %d page(s)",
                pdf_name, len(all_chunks), total_pages)
    return all_chunks
