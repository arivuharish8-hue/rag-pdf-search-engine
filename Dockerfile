# Flask + Celery worker image for the PDF RAG search engine.
# The same image runs both services; docker-compose overrides the command.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# build-essential: psutil/gevent/… compilation on slim.  libgomp1: FAISS.
# curl: container healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Non-root runtime user.  Celery refuses to boot as root once the broker
# serializer is JSON/pickle-free, so every service runs as this user.
# /app/uploads and /app/database are bind-mounted at runtime; Docker Desktop
# (gRPC-FUSE) exposes them as writable to container users.
RUN useradd -ms /bin/bash appuser \
    && mkdir -p /app/uploads /app/database \
    && chown -R appuser:appuser /app

# Pre-cache the SentenceTransformer model under the non-root user's HF cache
# so worker startup never re-downloads (as root it used /root/.cache).
USER appuser
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 5000

# Overridden in docker-compose for the worker service.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "180", "app:app"]
