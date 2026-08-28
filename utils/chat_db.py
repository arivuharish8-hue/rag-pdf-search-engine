"""Persistent chat storage for the conversational Q&A chatbot.

Uses the Supabase PostgreSQL backend exclusively.
"""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from utils.supabase_storage import get_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend state
# ---------------------------------------------------------------------------
_SESSIONS_TABLE = get_client().table("chat_sessions")
_MESSAGES_TABLE = get_client().table("chat_messages")

def _now():
    return datetime.now(timezone.utc).isoformat()

# ---------------------------------------------------------------------------
# Session operations
# ---------------------------------------------------------------------------

def create_session(title="New Chat"):
    """Create a new chat session and return its id."""
    session_id = str(uuid4())
    now = _now()
    row = {
        "id": session_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
    }
    _SESSIONS_TABLE.insert(row).execute()
    logger.info("[ChatDB] Session %s created: %s", session_id, title)
    return session_id

def get_session(session_id):
    """Return a single session dict, or None."""
    result = _SESSIONS_TABLE.select("*").eq("id", session_id).execute()
    return result.data[0] if result.data else None

def list_sessions(limit=50):
    """Return sessions ordered by updated_at descending (most recent first)."""
    result = (
        _SESSIONS_TABLE.select("*")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []

def update_session_title(session_id, title):
    """Update a session's title."""
    _SESSIONS_TABLE.update({"title": title, "updated_at": _now()}).eq("id", session_id).execute()

def delete_session(session_id):
    """Delete a session and all its messages."""
    _MESSAGES_TABLE.delete().eq("session_id", session_id).execute()
    _SESSIONS_TABLE.delete().eq("id", session_id).execute()

# ---------------------------------------------------------------------------
# Message operations
# ---------------------------------------------------------------------------

def add_message(session_id, role, content, sources=None):
    """Add a message to a session.

    ``sources`` is a list of source dicts (citation_id, pdf_name, page, etc.)
    stored as JSONB. Only used for assistant messages.
    """
    now = _now()
    row = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "created_at": now,
    }
    res = _MESSAGES_TABLE.insert(row).execute()
    if res.data:
        return res.data[0].get("id")
    return None

def get_messages(session_id, limit=100):
    """Return messages for a session, ordered by created_at ascending."""
    result = (
        _MESSAGES_TABLE.select("*")
        .eq("session_id", session_id)
        .order("created_at")
        .limit(limit)
        .execute()
    )
    return result.data or []

def get_recent_messages(session_id, limit=10):
    """Return the last N messages for a session (for conversation context).

    Fetches messages in descending order by created_at, takes the last ``limit``
    messages, then reverses them so they are in chronological order.  This avoids
    loading all messages and slicing in Python.
    """
    result = (
        _MESSAGES_TABLE.select("*")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    msgs = result.data or []
    msgs.reverse()
    return msgs
