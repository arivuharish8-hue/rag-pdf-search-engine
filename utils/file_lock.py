"""Cross-process file lock used to serialize FAISS / metadata writes.

Flask and every Celery worker are separate OS processes.  The in-process
``threading.Lock`` in faiss_db only guards threads inside one process, so a
second lock is needed to keep concurrent processes from corrupting
``faiss.index`` / ``metadata.pkl`` while rebuilding or appending vectors.

Works on POSIX (``fcntl.flock``) and Windows (``msvcrt.locking``).
"""

import os

try:
    import fcntl  # POSIX only
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:
    import msvcrt  # Windows only
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


class FileLock:
    """Exclusive advisory lock backed by a small lock file.

    Usage::

        with FileLock("database/.faiss.lock"):
            ...  # mutate faiss.index / metadata.pkl
    """

    def __init__(self, path):
        self.path = path
        self._fh = None

    def __enter__(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fh = open(self.path, "a+b")
        # msvcrt.locking needs at least one byte at the locked offset.
        self._fh.seek(0, os.SEEK_END)
        if self._fh.tell() == 0:
            self._fh.write(b"\x00")
            self._fh.flush()
        self._fh.seek(0)
        if fcntl is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
        else:  # pragma: no cover - neither fcntl nor msvcrt
            raise RuntimeError("No file locking support on this platform")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if fcntl is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._fh.close()
            self._fh = None
