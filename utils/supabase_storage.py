"""Supabase Storage helpers."""

import os
from supabase import create_client


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
    """Delete a single file from Supabase Storage."""
    try:
        client = get_client()
        client.storage.from_(get_bucket()).remove([path])
        return True
    except Exception as e:
        print(f"[Supabase] delete_file error: {e}")
        return False


def upload_file(local_path, object_name):
    """Upload a file to Supabase Storage."""
    client = get_client()
    with open(local_path, "rb") as f:
        client.storage.from_(get_bucket()).upload(
            path=object_name,
            file=f.read(),
            file_options={"content-type": "application/pdf"},
        )
