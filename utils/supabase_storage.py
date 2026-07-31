"""Supabase Storage helpers."""

import logging
import os

from supabase import create_client

logger = logging.getLogger(__name__)

_client = None


def get_client():
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = (
            os.getenv("SUPABASE_SECRET_KEY")
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )
        _client = create_client(url, key)
    return _client


def get_bucket():
    return os.getenv("SUPABASE_BUCKET", "pdfs")


def list_files():
    """Return list of file dicts from Supabase Storage.

    Each dict has: name, id, created_at, updated_at, metadata.
    Returns empty list on error.
    """
    try:
        client = get_client()
        files = client.storage.from_(get_bucket()).list()
        return files if files else []
    except Exception as e:
        print(f"[Supabase] list_files error: {e}")
        return []


def delete_file(path):
    """Delete a single file from Supabase Storage.

    Returns True on success.  Raises RuntimeError with a meaningful message
    when the deletion fails so callers never mistake a failed removal for
    success.  Supabase Storage remove() is idempotent for missing objects.
    """
    try:
        client = get_client()
        client.storage.from_(get_bucket()).remove([path])
        logger.info("[Supabase] Deleted %s from bucket %s", path, get_bucket())
        return True
    except Exception as e:
        logger.error("[Supabase] delete_file error for %s: %s",
                     path, e, exc_info=True)
        raise RuntimeError(
            f"Failed to delete '{path}' from Supabase Storage: {e}"
        ) from e


def upload_file(local_path, object_name):
    """Upload a file to Supabase Storage."""
    client = get_client()
    with open(local_path, "rb") as f:
        client.storage.from_(get_bucket()).upload(
            path=object_name,
            file=f.read(),
            file_options={"content-type": "application/pdf"},
        )


def download_file(object_name, local_path):
    """Download a file from Supabase Storage to a local path."""
    client = get_client()
    data = client.storage.from_(get_bucket()).download(object_name)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)
