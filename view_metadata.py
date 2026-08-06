"""Debugging tool: view metadata.pkl contents stored on disk.

Read-only — never modifies metadata.pkl.
"""

import os
import pickle

# ── Paths ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, "database", "metadata.pkl")

SEPARATOR = "=" * 72


def main():
    if not os.path.exists(META_PATH):
        print("No metadata.pkl found.")
        return

    try:
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
    except Exception as exc:
        print(f"FAILED to load metadata.pkl: {exc}")
        return

    if not isinstance(metadata, list):
        print(f"Unexpected metadata type: {type(metadata).__name__}")
        return

    print(f"Total metadata entries: {len(metadata)}")
    print()

    query = input("Enter a PDF filename (or press Enter to view all): ").strip().lower()

    matched = 0
    for i, entry in enumerate(metadata, start=1):
        if not isinstance(entry, dict):
            print(f"Entry {i}: (not a dict: {type(entry).__name__})")
            print(SEPARATOR)
            continue

        pdf_name = entry.get("pdf_name", "")
        if query and query not in str(pdf_name).lower():
            continue

        matched += 1
        text = str(entry.get("text", ""))
        page = entry.get("page")
        chunk_no = entry.get("chunk")
        print(f"PDF Name    : {pdf_name}")
        print(f"Page        : {page if page is not None else 'n/a'}")
        print(f"Chunk Number: {chunk_no if chunk_no is not None else 'n/a'}")
        print(f"Text Length : {len(text)}")
        print(f"Chunk Text  : {text[:500]}")
        print(SEPARATOR)

    if matched == 0:
        print(f"No matching entries for '{query}'.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
    except Exception as exc:
        print(f"Unexpected error: {exc}")
